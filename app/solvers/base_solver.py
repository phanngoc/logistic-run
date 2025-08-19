"""Base solver abstract class"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from ..schemas.orders import Order
from ..schemas.fleet import Truck
from ..schemas.costs import CostConfig
from ..schemas.dispatch import DispatchResponse, WeightConfig, Route, KPI
from ..services.travel_time import TravelTimeService
from ..services.port_api import PortApiService
from ..services.cost_engine import CostEngine

logger = logging.getLogger(__name__)


class SolverResult:
    """Kết quả từ solver"""
    
    def __init__(self):
        self.routes: List[Route] = []
        self.unserved_orders: List[str] = []
        self.solve_time_seconds: float = 0.0
        self.iterations: int = 0
        self.algorithm: str = "unknown"
        self.metadata: Dict[str, Any] = {}
    
    def calculate_kpi(self, total_orders: int) -> KPI:
        """Tính KPI từ solution"""
        served_orders = sum(len(route.order_ids) for route in self.routes)
        unserved_count = len(self.unserved_orders)
        
        total_distance = sum(route.total_distance_km for route in self.routes)
        total_cost = sum(route.cost_breakdown.total_cost for route in self.routes)
        
        late_orders = sum(
            1 for route in self.routes 
            for stop in route.stops 
            if stop.is_late and stop.order_id
        )
        
        overtime_hours = sum(route.cost_breakdown.overtime_hours for route in self.routes)
        
        # Utilization = orders served / total truck capacity
        utilization = served_orders / (len(self.routes) * 10) if self.routes else 0  # assume max 10 orders per truck
        utilization = min(1.0, utilization)
        
        avg_score = None
        if self.routes:
            scores = [route.score for route in self.routes if route.score is not None]
            if scores:
                avg_score = sum(scores) / len(scores)
        
        return KPI(
            total_orders=total_orders,
            served_orders=served_orders,
            unserved_orders=unserved_count,
            total_distance_km=total_distance,
            total_cost=total_cost,
            late_orders=late_orders,
            overtime_hours=overtime_hours,
            utilization_rate=utilization,
            avg_route_score=avg_score
        )


class BaseSolver(ABC):
    """Base abstract solver class"""
    
    def __init__(self, cost_config: CostConfig, weights: WeightConfig):
        self.cost_config = cost_config
        self.weights = weights
        self.cost_engine = CostEngine(cost_config)
        
        # Services will be injected
        self.travel_time_service: Optional[TravelTimeService] = None
        self.port_api_service: Optional[PortApiService] = None
    
    def set_services(self, travel_time_service: TravelTimeService, port_api_service: PortApiService):
        """Inject services dependencies"""
        self.travel_time_service = travel_time_service
        self.port_api_service = port_api_service
    
    @abstractmethod
    async def solve(self, 
                   orders: List[Order], 
                   trucks: List[Truck],
                   max_iterations: Optional[int] = None,
                   time_limit_seconds: Optional[int] = None) -> SolverResult:
        """
        Solve optimization problem
        
        Args:
            orders: Danh sách đơn hàng
            trucks: Danh sách xe
            max_iterations: Số iteration tối đa
            time_limit_seconds: Thời gian tối đa
            
        Returns:
            SolverResult
        """
        pass
    
    def _weights_to_dict(self) -> Dict[str, float]:
        """Convert WeightConfig to dict"""
        return {
            "lambda_late": self.weights.lambda_late,
            "lambda_ot": self.weights.lambda_ot, 
            "lambda_tw": self.weights.lambda_tw,
            "lambda_priority": self.weights.lambda_priority
        }
    
    def _calculate_objective_value(self, routes: List[Route]) -> float:
        """Tính objective value cho solution"""
        total_cost = sum(route.cost_breakdown.total_cost for route in routes)
        
        # Add penalty terms
        weights_dict = self._weights_to_dict()
        
        total_late_penalty = 0.0
        total_ot_penalty = 0.0
        
        for route in routes:
            # Late penalty
            total_late_penalty += weights_dict["lambda_late"] * route.cost_breakdown.late_minutes
            
            # Overtime penalty
            total_ot_penalty += weights_dict["lambda_ot"] * route.cost_breakdown.overtime_hours * 100
        
        return total_cost + total_late_penalty + total_ot_penalty
    
    async def create_dispatch_response(self, 
                                     solver_result: SolverResult, 
                                     total_orders: int) -> DispatchResponse:
        """Tạo DispatchResponse từ SolverResult"""
        
        kpi = solver_result.calculate_kpi(total_orders)
        
        return DispatchResponse(
            success=True,
            message=f"Optimization completed successfully using {solver_result.algorithm}",
            routes=solver_result.routes,
            unserved_orders=solver_result.unserved_orders,
            kpi=kpi,
            solve_time_seconds=solver_result.solve_time_seconds,
            iterations=solver_result.iterations,
            algorithm=solver_result.algorithm,
            metadata=solver_result.metadata
        )
    
    async def _validate_solution(self, routes: List[Route], orders: List[Order], trucks: List[Truck]) -> List[str]:
        """Validate solution constraints"""
        violations = []
        
        # Check all orders are served max once
        served_orders = set()
        for route in routes:
            for order_id in route.order_ids:
                if order_id in served_orders:
                    violations.append(f"Order {order_id} served multiple times")
                served_orders.add(order_id)
        
        # Check truck constraints
        truck_dict = {truck.truck_id: truck for truck in trucks}
        for route in routes:
            truck = truck_dict.get(route.truck_id)
            if not truck:
                violations.append(f"Route assigned to unknown truck {route.truck_id}")
                continue
            
            # Check max orders
            if len(route.order_ids) > truck.max_orders_per_day:
                violations.append(f"Truck {truck.truck_id} exceeds max orders: {len(route.order_ids)} > {truck.max_orders_per_day}")
        
        return violations
