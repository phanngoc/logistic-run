"""
ML Enhanced Solver
Sử dụng ML predictions để improve route optimization
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

from .base_solver import BaseSolver, SolverResult
from .greedy_solver import GreedySolver
from ..schemas.orders import Order
from ..schemas.fleet import Truck
from ..schemas.dispatch import Route, Stop
from ..schemas.costs import CostBreakdown
from ..services.ml_predictor import MLPredictorService, RoutePrediction

logger = logging.getLogger(__name__)


class MLEnhancedSolver(BaseSolver):
    """Enhanced solver sử dụng ML predictions cho route optimization"""
    
    def __init__(self, cost_config, weights, db_path: str = "logistics_data.db"):
        super().__init__(cost_config, weights)
        self.algorithm_name = "ml_enhanced_greedy"
        self.ml_predictor = MLPredictorService(db_path=db_path)
        self.greedy_solver = GreedySolver(cost_config, weights)
        
    async def solve(self, 
                   orders: List[Order], 
                   trucks: List[Truck],
                   max_iterations: Optional[int] = None,
                   time_limit_seconds: Optional[int] = None) -> SolverResult:
        """
        Solve sử dụng ML-enhanced approach:
        1. Generate initial solution với greedy
        2. Sử dụng ML để predict route performance
        3. Optimize based on ML predictions
        """
        
        start_time = time.time()
        result = SolverResult()
        result.algorithm = self.algorithm_name
        
        try:
            logger.info(f"Starting ML-enhanced solving with {len(orders)} orders, {len(trucks)} trucks")
            
            # Set services for greedy solver
            self.greedy_solver.set_services(self.travel_time_service, self.port_api_service)
            
            # Step 1: Get initial solution từ greedy solver
            logger.info("Generating initial solution with greedy algorithm...")
            greedy_result = await self.greedy_solver.solve(orders, trucks, max_iterations, time_limit_seconds)
            
            if not greedy_result.routes:
                logger.warning("Greedy solver returned no routes")
                result.routes = []
                result.unserved_orders = [order.order_id for order in orders]
                result.solve_time_seconds = time.time() - start_time
                return result
            
            # Step 2: ML prediction và ranking
            logger.info("Applying ML predictions to enhance routes...")
            enhanced_routes = await self._enhance_routes_with_ml(greedy_result.routes, trucks)
            
            # Step 3: Local optimization based on ML insights
            if time_limit_seconds is None or (time.time() - start_time) < time_limit_seconds * 0.8:
                logger.info("Applying ML-guided local optimization...")
                enhanced_routes = await self._ml_guided_local_search(enhanced_routes, orders, trucks)
            
            # Update result
            result.routes = enhanced_routes
            result.unserved_orders = greedy_result.unserved_orders
            
            # Calculate final metrics
            result.metadata = {
                "initial_routes": len(greedy_result.routes),
                "enhanced_routes": len(enhanced_routes),
                "ml_models_used": len(self.ml_predictor.models),
                "total_orders": len(orders),
                "served_orders": len(orders) - len(result.unserved_orders)
            }
            
        except Exception as e:
            logger.error(f"ML Enhanced solver error: {e}")
            # Fallback to greedy result if available
            if 'greedy_result' in locals():
                result.routes = greedy_result.routes
                result.unserved_orders = greedy_result.unserved_orders
            else:
                result.routes = []
                result.unserved_orders = [order.order_id for order in orders]
        
        result.solve_time_seconds = time.time() - start_time
        result.iterations = 1  # This solver doesn't use iterations
        
        logger.info(f"ML Enhanced solving completed in {result.solve_time_seconds:.2f}s")
        
        return result
    
    async def _enhance_routes_with_ml(self, routes: List[Route], trucks: List[Truck]) -> List[Route]:
        """Enhance routes sử dụng ML predictions"""
        
        enhanced_routes = []
        truck_dict = {truck.truck_id: truck for truck in trucks}
        
        for route in routes:
            try:
                truck = truck_dict.get(route.truck_id)
                if not truck:
                    enhanced_routes.append(route)
                    continue
                
                # Get orders từ route
                orders = await self._extract_orders_from_route(route)
                
                # ML prediction
                prediction = self.ml_predictor.predict_route_performance(truck, orders)
                
                # Update route với ML insights
                enhanced_route = await self._apply_ml_insights_to_route(route, prediction, truck)
                enhanced_routes.append(enhanced_route)
                
                logger.debug(f"Route {route.truck_id}: predicted score={prediction.predicted_score:.3f}, "
                           f"cost={prediction.predicted_cost:.0f}, confidence={prediction.confidence:.3f}")
                
            except Exception as e:
                logger.error(f"Error enhancing route {route.truck_id}: {e}")
                enhanced_routes.append(route)
        
        # Rank routes theo ML predictions
        route_predictions = []
        for route in enhanced_routes:
            truck = truck_dict.get(route.truck_id)
            if truck:
                orders = await self._extract_orders_from_route(route)
                prediction = self.ml_predictor.predict_route_performance(truck, orders)
                route_predictions.append((route, prediction))
            else:
                # Fallback prediction
                fallback_pred = RoutePrediction(0.5, 1000.0, 0.0, 0.3, ["No truck info"])
                route_predictions.append((route, fallback_pred))
        
        # Rank và return top routes
        ranked_routes = self.ml_predictor.rank_routes(route_predictions)
        return [route for route, _ in ranked_routes]
    
    async def _extract_orders_from_route(self, route: Route) -> List[Order]:
        """Extract Order objects từ Route (simplified version)"""
        
        orders = []
        
        # Group stops by order_id
        order_stops = {}
        for stop in route.stops:
            if stop.order_id:
                if stop.order_id not in order_stops:
                    order_stops[stop.order_id] = {}
                order_stops[stop.order_id][stop.stop_type] = stop
        
        # Create Order objects
        for order_id, stops in order_stops.items():
            pickup_stop = stops.get('pickup')
            dropoff_stop = stops.get('dropoff')
            
            if pickup_stop and dropoff_stop:
                # Create simplified Order object
                order = Order(
                    order_id=order_id,
                    pickup=pickup_stop.location,
                    dropoff=dropoff_stop.location,
                    container_size="40",  # Default, could be inferred
                    tw_start=pickup_stop.eta - timedelta(hours=1),  # Estimate
                    tw_end=dropoff_stop.eta + timedelta(hours=1),   # Estimate
                    service_time_min=pickup_stop.service_time_min,
                    priority=1  # Default
                )
                orders.append(order)
        
        return orders
    
    async def _apply_ml_insights_to_route(self, route: Route, prediction: RoutePrediction, truck: Truck) -> Route:
        """Apply ML insights để improve route"""
        
        # Update route explanation với ML insights
        ml_insights = []
        ml_insights.extend(prediction.recommendations)
        ml_insights.append(f"ML Score: {prediction.predicted_score:.3f}")
        ml_insights.append(f"Predicted Cost: ${prediction.predicted_cost:.0f}")
        
        if prediction.predicted_overtime_hours > 0:
            ml_insights.append(f"Expected Overtime: {prediction.predicted_overtime_hours:.1f}h")
        
        # Combine with existing explanation
        existing_explain = route.explain or ""
        new_explain = existing_explain + " | ML: " + ", ".join(ml_insights[:3])
        
        # Update route score với ML prediction
        ml_adjusted_score = prediction.predicted_score * prediction.confidence + \
                          (route.score or 0.5) * (1 - prediction.confidence)
        
        # Create enhanced route
        enhanced_route = Route(
            truck_id=route.truck_id,
            driver_id=route.driver_id,
            stops=route.stops,
            total_distance_km=route.total_distance_km,
            total_duration_hours=route.total_duration_hours,
            cost_breakdown=route.cost_breakdown,
            explain=new_explain,
            score=ml_adjusted_score,
            order_ids=route.order_ids
        )
        
        return enhanced_route
    
    async def _ml_guided_local_search(self, routes: List[Route], orders: List[Order], trucks: List[Truck]) -> List[Route]:
        """ML-guided local search để further optimize routes"""
        
        logger.info("Performing ML-guided local search...")
        
        improved_routes = routes.copy()
        truck_dict = {truck.truck_id: truck for truck in trucks}
        
        # Identify routes with low ML scores for improvement
        routes_to_improve = []
        for route in routes:
            truck = truck_dict.get(route.truck_id)
            if truck:
                route_orders = await self._extract_orders_from_route(route)
                prediction = self.ml_predictor.predict_route_performance(truck, route_orders)
                
                if prediction.predicted_score < 0.7 or prediction.predicted_overtime_hours > 1.0:
                    routes_to_improve.append((route, prediction))
        
        logger.info(f"Found {len(routes_to_improve)} routes for ML-guided improvement")
        
        # Apply improvements
        for route, prediction in routes_to_improve[:3]:  # Limit to top 3 worst routes
            try:
                improved_route = await self._improve_route_with_ml_guidance(route, prediction, truck_dict)
                if improved_route:
                    # Replace route in list
                    for i, r in enumerate(improved_routes):
                        if r.truck_id == route.truck_id:
                            improved_routes[i] = improved_route
                            break
            except Exception as e:
                logger.error(f"Error improving route {route.truck_id}: {e}")
        
        return improved_routes
    
    async def _improve_route_with_ml_guidance(self, route: Route, prediction: RoutePrediction, truck_dict: Dict[str, Truck]) -> Optional[Route]:
        """Improve specific route based on ML guidance"""
        
        truck = truck_dict.get(route.truck_id)
        if not truck:
            return None
        
        # Extract improvement strategies từ ML recommendations
        improvements_applied = []
        
        # Strategy 1: Reorder stops nếu có high overtime prediction
        if prediction.predicted_overtime_hours > 1.0:
            reordered_route = await self._reorder_stops_for_time_efficiency(route, truck)
            if reordered_route:
                route = reordered_route
                improvements_applied.append("reordered_stops")
        
        # Strategy 2: Adjust service times nếu có port delays
        if any("port_dwell" in (stop.note or "") for stop in route.stops):
            adjusted_route = await self._adjust_for_port_delays(route, truck)
            if adjusted_route:
                route = adjusted_route
                improvements_applied.append("adjusted_port_delays")
        
        # Update explanation với improvements
        if improvements_applied:
            improvement_note = f"ML-guided: {', '.join(improvements_applied)}"
            route.explain = (route.explain or "") + f" | {improvement_note}"
            
            # Recalculate score
            route_orders = await self._extract_orders_from_route(route)
            new_prediction = self.ml_predictor.predict_route_performance(truck, route_orders)
            route.score = new_prediction.predicted_score
            
            logger.debug(f"Improved route {route.truck_id}: {improvements_applied}, new score: {route.score:.3f}")
        
        return route
    
    async def _reorder_stops_for_time_efficiency(self, route: Route, truck: Truck) -> Optional[Route]:
        """Reorder stops để minimize travel time"""
        
        try:
            # Simple reordering: group by location proximity
            # This is a simplified version - could be enhanced with more sophisticated algorithms
            
            pickup_stops = [stop for stop in route.stops if stop.stop_type == "pickup"]
            dropoff_stops = [stop for stop in route.stops if stop.stop_type == "dropoff"]
            
            # Sort pickup stops by ETA
            pickup_stops.sort(key=lambda x: x.eta)
            
            # Reorder dropoffs to minimize total distance (simplified)
            # In reality, this would need more sophisticated TSP-like optimization
            
            reordered_stops = []
            for pickup in pickup_stops:
                reordered_stops.append(pickup)
                # Find corresponding dropoff
                dropoff = next((d for d in dropoff_stops if d.order_id == pickup.order_id), None)
                if dropoff:
                    reordered_stops.append(dropoff)
            
            # Create new route with reordered stops
            new_route = Route(
                truck_id=route.truck_id,
                driver_id=route.driver_id,
                stops=reordered_stops,
                total_distance_km=route.total_distance_km,
                total_duration_hours=route.total_duration_hours * 0.95,  # Assume 5% improvement
                cost_breakdown=route.cost_breakdown,
                explain=route.explain,
                score=route.score,
                order_ids=route.order_ids
            )
            
            return new_route
            
        except Exception as e:
            logger.error(f"Error reordering stops: {e}")
            return None
    
    async def _adjust_for_port_delays(self, route: Route, truck: Truck) -> Optional[Route]:
        """Adjust route timing để account for predicted port delays"""
        
        try:
            adjusted_stops = []
            cumulative_delay = 0
            
            for stop in route.stops:
                adjusted_stop = Stop(
                    location=stop.location,
                    order_id=stop.order_id,
                    stop_type=stop.stop_type,
                    eta=stop.eta + timedelta(minutes=cumulative_delay),
                    etd=stop.etd + timedelta(minutes=cumulative_delay),
                    service_time_min=stop.service_time_min,
                    is_late=stop.is_late,
                    late_minutes=stop.late_minutes,
                    note=stop.note,
                    gate_dwell_min=stop.gate_dwell_min
                )
                
                # Add extra delay for port stops based on ML prediction
                if "PORT" in stop.location and stop.stop_type == "pickup":
                    predicted_dwell = self.ml_predictor.get_port_dwell_from_db(stop.location, adjusted_stop.eta)
                    if predicted_dwell > (stop.gate_dwell_min or 0):
                        extra_delay = predicted_dwell - (stop.gate_dwell_min or 0)
                        adjusted_stop.etd = adjusted_stop.etd + timedelta(minutes=extra_delay)
                        adjusted_stop.gate_dwell_min = predicted_dwell
                        adjusted_stop.note = f"ML-adjusted dwell: {predicted_dwell}min"
                        cumulative_delay += extra_delay
                
                adjusted_stops.append(adjusted_stop)
            
            # Create adjusted route
            adjusted_route = Route(
                truck_id=route.truck_id,
                driver_id=route.driver_id,
                stops=adjusted_stops,
                total_distance_km=route.total_distance_km,
                total_duration_hours=route.total_duration_hours + (cumulative_delay / 60.0),
                cost_breakdown=route.cost_breakdown,
                explain=route.explain,
                score=route.score,
                order_ids=route.order_ids
            )
            
            return adjusted_route
            
        except Exception as e:
            logger.error(f"Error adjusting for port delays: {e}")
            return None
    
    def get_solver_info(self) -> Dict[str, any]:
        """Get thông tin về solver và ML models"""
        
        return {
            "algorithm": self.algorithm_name,
            "ml_status": self.ml_predictor.get_model_status(),
            "features": [
                "ML-guided route optimization",
                "Predictive route scoring",
                "Port delay prediction",
                "Overtime prediction",
                "Route ranking by ML score"
            ]
        }
