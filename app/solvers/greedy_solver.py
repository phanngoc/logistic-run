"""Greedy solver với insertion heuristic"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
import logging

from .base_solver import BaseSolver, SolverResult
from ..schemas.orders import Order
from ..schemas.fleet import Truck
from ..schemas.dispatch import Route, Stop
from ..schemas.costs import CostBreakdown
from ..services.travel_time import TravelTimeResult
from ..utils.route_utils import RouteUtils
from ..utils.time_utils import TimeUtils

logger = logging.getLogger(__name__)


class GreedySolver(BaseSolver):
    """Greedy solver sử dụng best insertion heuristic"""
    
    def __init__(self, cost_config, weights):
        super().__init__(cost_config, weights)
        self.algorithm_name = "greedy_insertion"
    
    async def solve(self, 
                   orders: List[Order], 
                   trucks: List[Truck],
                   max_iterations: Optional[int] = None,
                   time_limit_seconds: Optional[int] = None) -> SolverResult:
        """
        Solve bằng greedy insertion
        
        1. Sort orders theo priority và urgency
        2. Với mỗi order, tìm truck + position tốt nhất (min cost increase)
        3. Insert order vào route tốt nhất
        """
        
        start_time = time.time()
        result = SolverResult()
        result.algorithm = self.algorithm_name
        
        try:
            # Store trucks for later use
            self.trucks = trucks
            
            # Precompute travel times matrix
            await self._precompute_travel_times(orders, trucks)
            
            # Initialize solution
            solution = self._initialize_solution(trucks)
            unserved_orders = orders.copy()
            
            # Sort orders by priority and urgency
            sorted_orders = self._sort_orders_for_insertion(unserved_orders)
            
            # Greedy insertion
            iteration = 0
            while sorted_orders and iteration < (max_iterations or 1000):
                iteration += 1
                
                # Check time limit
                if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
                    logger.info(f"Time limit reached after {iteration} iterations")
                    break
                
                order = sorted_orders.pop(0)
                
                # Find best insertion
                best_insertion = await self._find_best_insertion(order, solution)
                
                if best_insertion:
                    truck_id, insertion_pos = best_insertion
                    await self._insert_order(order, truck_id, insertion_pos, solution)
                    logger.debug(f"Inserted order {order.order_id} into truck {truck_id}")
                else:
                    # Cannot insert this order
                    result.unserved_orders.append(order.order_id)
                    logger.warning(f"Cannot insert order {order.order_id}")
            
            # Add remaining unserved orders
            for order in sorted_orders:
                result.unserved_orders.append(order.order_id)
            
            # Convert solution to routes
            result.routes = await self._solution_to_routes(solution, trucks)
            
        except Exception as e:
            logger.error(f"Greedy solver error: {e}")
            result.routes = []
            result.unserved_orders = [order.order_id for order in orders]
        
        result.solve_time_seconds = time.time() - start_time
        result.iterations = iteration
        result.metadata = {
            "total_orders": len(orders),
            "served_orders": len(orders) - len(result.unserved_orders),
            "trucks_used": len([r for r in result.routes if r.order_ids])
        }
        
        return result
    
    def _initialize_solution(self, trucks: List[Truck]) -> Dict[str, List[Order]]:
        """Initialize empty solution"""
        return {truck.truck_id: [] for truck in trucks}
    
    def _sort_orders_for_insertion(self, orders: List[Order]) -> List[Order]:
        """Sort orders theo priority và urgency"""
        
        def sort_key(order: Order) -> Tuple[int, float, str]:
            # Priority (higher priority first)
            priority = -order.priority
            
            # Urgency (tighter time window first)
            tw_duration = (order.tw_end - order.tw_start).total_seconds() / 3600  # hours
            urgency = tw_duration
            
            # Order ID for stable sort
            order_id = order.order_id
            
            return (priority, urgency, order_id)
        
        return sorted(orders, key=sort_key)
    
    async def _precompute_travel_times(self, orders: List[Order], trucks: List[Truck]):
        """Precompute travel time matrix cho các địa điểm cần thiết"""
        
        # Collect all unique locations
        locations = set()
        
        # Add truck start locations
        for truck in trucks:
            locations.add(truck.start_location)
        
        # Add order locations
        for order in orders:
            locations.add(order.pickup)
            locations.add(order.dropoff)
        
        locations_list = list(locations)
        
        # Get travel time matrix
        if self.travel_time_service:
            self.travel_time_matrix = await self.travel_time_service.get_travel_time_matrix(
                locations_list, datetime.now()
            )
        else:
            # Mock travel times for testing
            self.travel_time_matrix = {}
            for i, loc1 in enumerate(locations_list):
                for j, loc2 in enumerate(locations_list):
                    if i != j:
                        self.travel_time_matrix[(loc1, loc2)] = TravelTimeResult(
                            travel_time_seconds=1800,  # 30 min default
                            distance_meters=30000,     # 30 km default
                            highway_distance_meters=15000  # 50% highway
                        )
    
    async def _find_best_insertion(self, order: Order, solution: Dict[str, List[Order]]) -> Optional[Tuple[str, int]]:
        """
        Tìm vị trí insertion tốt nhất cho order
        
        Returns:
            (truck_id, position) hoặc None nếu không insert được
        """
        
        best_truck_id = None
        best_position = None
        best_cost_increase = float('inf')
        
        for truck_id, route_orders in solution.items():
            # Check if truck can handle this order
            truck = next((t for t in self.trucks if t.truck_id == truck_id), None)
            if not truck or not RouteUtils.can_truck_handle_order(truck, order):
                continue
            
            # Check route capacity
            if len(route_orders) >= truck.max_orders_per_day:
                continue
            
            # Try inserting at each position
            for pos in range(len(route_orders) + 1):
                cost_increase = await self._calculate_insertion_cost(order, truck_id, pos, route_orders)
                
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_truck_id = truck_id
                    best_position = pos
        
        if best_truck_id is not None:
            return (best_truck_id, best_position)
        
        return None
    
    async def _calculate_insertion_cost(self, 
                                      order: Order, 
                                      truck_id: str, 
                                      position: int, 
                                      current_route: List[Order]) -> float:
        """Tính cost increase khi insert order tại position trong truck route"""
        
        # Get truck info
        truck = next((t for t in self.trucks if t.truck_id == truck_id), None)
        if not truck:
            return float('inf')
        
        # Create new route with inserted order
        new_route = current_route.copy()
        new_route.insert(position, order)
        
        # Calculate cost for original route
        original_cost = await self._calculate_route_cost(truck, current_route)
        
        # Calculate cost for new route
        new_cost = await self._calculate_route_cost(truck, new_route)
        
        # Return cost increase
        return new_cost.total_cost - original_cost.total_cost
    
    async def _calculate_route_cost(self, truck: Truck, orders: List[Order]) -> CostBreakdown:
        """Tính cost breakdown cho route"""
        
        if not orders:
            return CostBreakdown()
        
        # Build route segments
        segments = []
        current_location = truck.start_location
        current_time = truck.shift_start
        
        for order in orders:
            # Segment to pickup
            pickup_travel_key = (current_location, order.pickup)
            pickup_travel = self.travel_time_matrix.get(pickup_travel_key)
            
            if pickup_travel:
                pickup_arrival = current_time + timedelta(seconds=pickup_travel.travel_time_seconds)
            else:
                pickup_arrival = current_time + timedelta(minutes=30)  # fallback
            
            # Wait for time window if needed
            pickup_arrival = max(pickup_arrival, order.tw_start)
            pickup_departure = pickup_arrival + timedelta(minutes=order.get_pickup_service_time())
            
            # Add gate dwell if pickup is at port
            if "PORT" in order.pickup:
                if self.port_api_service:
                    dwell_info = await self.port_api_service.get_gate_dwell_info(order.pickup, pickup_arrival)
                    pickup_departure += timedelta(minutes=dwell_info.expected_dwell_minutes)
                else:
                    pickup_departure += timedelta(minutes=30)  # default port dwell
            
            segments.append({
                "from": current_location,
                "to": order.pickup,
                "travel_result": pickup_travel or TravelTimeResult(1800, 30000),
                "departure_time": current_time,
                "order_id": order.order_id,
                "late_minutes": 0
            })
            
            # Segment to dropoff
            dropoff_travel_key = (order.pickup, order.dropoff)
            dropoff_travel = self.travel_time_matrix.get(dropoff_travel_key)
            
            if dropoff_travel:
                dropoff_arrival = pickup_departure + timedelta(seconds=dropoff_travel.travel_time_seconds)
            else:
                dropoff_arrival = pickup_departure + timedelta(minutes=45)  # fallback
            
            dropoff_departure = dropoff_arrival + timedelta(minutes=order.get_dropoff_service_time())
            
            # Calculate late minutes
            late_minutes = max(0, int((dropoff_arrival - order.tw_end).total_seconds() / 60))
            
            segments.append({
                "from": order.pickup,
                "to": order.dropoff,
                "travel_result": dropoff_travel or TravelTimeResult(2700, 45000),
                "departure_time": pickup_departure,
                "order_id": order.order_id,
                "late_minutes": late_minutes,
                "service_time_min": order.get_dropoff_service_time()
            })
            
            # Update current position and time
            current_location = order.dropoff
            current_time = dropoff_departure
        
        # Calculate cost using cost engine
        return self.cost_engine.calculate_route_cost(truck, segments, {order.order_id: order for order in orders})
    
    async def _insert_order(self, order: Order, truck_id: str, position: int, solution: Dict[str, List[Order]]):
        """Insert order vào solution"""
        solution[truck_id].insert(position, order)
    
    async def _solution_to_routes(self, solution: Dict[str, List[Order]], trucks: List[Truck]) -> List[Route]:
        """Convert solution dict thành list Route objects"""
        
        routes = []
        truck_dict = {truck.truck_id: truck for truck in trucks}
        
        for truck_id, orders in solution.items():
            if not orders:  # Skip empty routes
                continue
            
            truck = truck_dict[truck_id]
            
            # Calculate route details
            cost_breakdown = await self._calculate_route_cost(truck, orders)
            stops = await self._create_stops_for_route(truck, orders)
            
            # Calculate route metrics
            total_distance = cost_breakdown.distance_km
            total_duration = 0.0
            if stops:
                start_time = min(stop.eta for stop in stops)
                end_time = max(stop.etd for stop in stops)
                total_duration = (end_time - start_time).total_seconds() / 3600
            
            # Create explanation
            explain = self._generate_route_explanation(cost_breakdown, orders, stops)
            
            # Calculate score
            weights_dict = self._weights_to_dict()
            score = self.cost_engine.estimate_route_score(cost_breakdown, weights_dict)
            
            route = Route(
                truck_id=truck_id,
                driver_id=truck.driver_id,
                stops=stops,
                total_distance_km=total_distance,
                total_duration_hours=total_duration,
                cost_breakdown=cost_breakdown,
                explain=explain,
                score=score,
                order_ids=[order.order_id for order in orders]
            )
            
            routes.append(route)
        
        return routes
    
    async def _create_stops_for_route(self, truck: Truck, orders: List[Order]) -> List[Stop]:
        """Tạo stops cho route"""
        
        stops = []
        current_location = truck.start_location
        current_time = truck.shift_start
        
        for order in orders:
            # Pickup stop
            pickup_travel_key = (current_location, order.pickup)
            pickup_travel = self.travel_time_matrix.get(pickup_travel_key)
            
            if pickup_travel:
                pickup_arrival = current_time + timedelta(seconds=pickup_travel.travel_time_seconds)
            else:
                pickup_arrival = current_time + timedelta(minutes=30)
            
            pickup_arrival = max(pickup_arrival, order.tw_start)
            
            # Add gate dwell if needed
            gate_dwell_min = 0
            note = ""
            if "PORT" in order.pickup:
                if self.port_api_service:
                    dwell_info = await self.port_api_service.get_gate_dwell_info(order.pickup, pickup_arrival)
                    gate_dwell_min = dwell_info.expected_dwell_minutes
                    note = f"gate_dwell={gate_dwell_min}m"
                else:
                    gate_dwell_min = 30
                    note = "gate_dwell=30m(default)"
            
            pickup_departure = pickup_arrival + timedelta(minutes=order.get_pickup_service_time() + gate_dwell_min)
            
            pickup_stop = Stop(
                location=order.pickup,
                order_id=order.order_id,
                stop_type="pickup",
                eta=pickup_arrival,
                etd=pickup_departure,
                service_time_min=order.get_pickup_service_time(),
                is_late=pickup_arrival > order.tw_start,
                late_minutes=max(0, int((pickup_arrival - order.tw_start).total_seconds() / 60)),
                note=note,
                gate_dwell_min=gate_dwell_min if gate_dwell_min > 0 else None
            )
            stops.append(pickup_stop)
            
            # Dropoff stop
            dropoff_travel_key = (order.pickup, order.dropoff)
            dropoff_travel = self.travel_time_matrix.get(dropoff_travel_key)
            
            if dropoff_travel:
                dropoff_arrival = pickup_departure + timedelta(seconds=dropoff_travel.travel_time_seconds)
            else:
                dropoff_arrival = pickup_departure + timedelta(minutes=45)
            
            dropoff_departure = dropoff_arrival + timedelta(minutes=order.get_dropoff_service_time())
            
            dropoff_stop = Stop(
                location=order.dropoff,
                order_id=order.order_id,
                stop_type="dropoff",
                eta=dropoff_arrival,
                etd=dropoff_departure,
                service_time_min=order.get_dropoff_service_time(),
                is_late=dropoff_arrival > order.tw_end,
                late_minutes=max(0, int((dropoff_arrival - order.tw_end).total_seconds() / 60)),
                note="on-time" if dropoff_arrival <= order.tw_end else "late"
            )
            stops.append(dropoff_stop)
            
            # Update current position and time
            current_location = order.dropoff
            current_time = dropoff_departure
        
        return stops
    
    def _generate_route_explanation(self, cost_breakdown: CostBreakdown, orders: List[Order], stops: List[Stop]) -> str:
        """Tạo explanation cho route"""
        
        explanations = []
        
        # Time window compliance
        late_count = sum(1 for stop in stops if stop.is_late)
        if late_count == 0:
            explanations.append("meets all time windows")
        else:
            explanations.append(f"{late_count} late deliveries")
        
        # Overtime
        if cost_breakdown.overtime_hours == 0:
            explanations.append("no overtime")
        else:
            explanations.append(f"{cost_breakdown.overtime_hours:.1f}h overtime")
        
        # Route efficiency
        if cost_breakdown.distance_km < 100:
            explanations.append("short route")
        elif cost_breakdown.distance_km > 200:
            explanations.append("long route")
        else:
            explanations.append("moderate distance")
        
        # Highway usage
        highway_ratio = cost_breakdown.highway_km / cost_breakdown.distance_km if cost_breakdown.distance_km > 0 else 0
        if highway_ratio > 0.6:
            explanations.append("highway optimized")
        
        return ", ".join(explanations)
