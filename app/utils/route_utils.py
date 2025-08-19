"""Route utilities cho optimization"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
import logging

from ..schemas.orders import Order
from ..schemas.fleet import Truck
from ..schemas.dispatch import Stop
from ..services.travel_time import TravelTimeResult

logger = logging.getLogger(__name__)


class RouteUtils:
    """Helper functions cho route operations"""
    
    @staticmethod
    def can_truck_handle_order(truck: Truck, order: Order) -> bool:
        """Kiểm tra xe có thể xử lý order này không"""
        return order.container_size in truck.allowed_sizes
    
    @staticmethod
    def calculate_route_feasibility(truck: Truck, 
                                  orders: List[Order],
                                  current_time: Optional[datetime] = None) -> Dict[str, bool]:
        """
        Kiểm tra feasibility cho từng order với truck
        
        Returns:
            Dict với order_id -> feasible (True/False)
        """
        if current_time is None:
            current_time = datetime.now()
        
        feasibility = {}
        
        for order in orders:
            # Check container size compatibility
            if not RouteUtils.can_truck_handle_order(truck, order):
                feasibility[order.order_id] = False
                continue
            
            # Check time window feasibility (simple check)
            # Minimum time needed = current -> pickup -> dropoff + service times
            min_time_needed = timedelta(hours=2)  # Simplified estimation
            
            earliest_completion = current_time + min_time_needed
            if earliest_completion > order.tw_end:
                feasibility[order.order_id] = False
                continue
            
            # Check if truck shift can accommodate
            if order.tw_start > truck.shift_end or order.tw_end < truck.shift_start:
                feasibility[order.order_id] = False
                continue
            
            feasibility[order.order_id] = True
        
        return feasibility
    
    @staticmethod
    def create_stops_from_orders(orders: List[Order], 
                               travel_times: Dict[Tuple[str, str], TravelTimeResult],
                               start_location: str,
                               start_time: datetime) -> List[Stop]:
        """
        Tạo danh sách stops từ orders theo thứ tự
        
        Args:
            orders: Danh sách orders theo thứ tự trong route
            travel_times: Dict travel times giữa các địa điểm
            start_location: Địa điểm bắt đầu
            start_time: Thời gian bắt đầu
            
        Returns:
            List các Stop với timing calculated
        """
        stops = []
        current_location = start_location
        current_time = start_time
        
        for order in orders:
            # Pickup stop
            pickup_travel_key = (current_location, order.pickup)
            pickup_travel = travel_times.get(pickup_travel_key)
            
            if pickup_travel:
                pickup_arrival = current_time + timedelta(seconds=pickup_travel.travel_time_seconds)
            else:
                # Fallback estimation
                pickup_arrival = current_time + timedelta(minutes=30)
            
            # Adjust for time window
            pickup_arrival = max(pickup_arrival, order.tw_start)
            pickup_departure = pickup_arrival + timedelta(minutes=order.get_pickup_service_time())
            
            pickup_stop = Stop(
                location=order.pickup,
                order_id=order.order_id,
                stop_type="pickup",
                eta=pickup_arrival,
                etd=pickup_departure,
                service_time_min=order.get_pickup_service_time(),
                is_late=pickup_arrival > order.tw_start,
                late_minutes=max(0, int((pickup_arrival - order.tw_start).total_seconds() / 60))
            )
            stops.append(pickup_stop)
            
            # Dropoff stop
            dropoff_travel_key = (order.pickup, order.dropoff)
            dropoff_travel = travel_times.get(dropoff_travel_key)
            
            if dropoff_travel:
                dropoff_arrival = pickup_departure + timedelta(seconds=dropoff_travel.travel_time_seconds)
            else:
                # Fallback estimation
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
                late_minutes=max(0, int((dropoff_arrival - order.tw_end).total_seconds() / 60))
            )
            stops.append(dropoff_stop)
            
            # Update current position and time
            current_location = order.dropoff
            current_time = dropoff_departure
        
        return stops
    
    @staticmethod
    def validate_route_constraints(truck: Truck, orders: List[Order], stops: List[Stop]) -> List[str]:
        """
        Validate route constraints và return list of violations
        
        Returns:
            List of constraint violation messages
        """
        violations = []
        
        # Check container size constraints
        for order in orders:
            if not RouteUtils.can_truck_handle_order(truck, order):
                violations.append(f"Truck {truck.truck_id} cannot handle container size {order.container_size} for order {order.order_id}")
        
        # Check max orders per day
        if len(orders) > truck.max_orders_per_day:
            violations.append(f"Truck {truck.truck_id} exceeds max orders per day: {len(orders)} > {truck.max_orders_per_day}")
        
        # Check shift time constraints
        for stop in stops:
            if stop.eta < truck.shift_start:
                violations.append(f"Stop at {stop.location} scheduled before shift start")
            if stop.etd > truck.shift_end:
                violations.append(f"Stop at {stop.location} scheduled after shift end")
        
        # Check time window violations
        for stop in stops:
            if stop.is_late and stop.order_id:
                violations.append(f"Late delivery for order {stop.order_id} at {stop.location}: {stop.late_minutes} minutes late")
        
        return violations
    
    @staticmethod
    def calculate_route_statistics(stops: List[Stop]) -> Dict[str, float]:
        """Tính statistics cho route"""
        if not stops:
            return {}
        
        total_service_time = sum(stop.service_time_min for stop in stops)
        total_late_minutes = sum(stop.late_minutes for stop in stops)
        late_stops_count = sum(1 for stop in stops if stop.is_late)
        
        start_time = min(stop.eta for stop in stops)
        end_time = max(stop.etd for stop in stops)
        total_duration = (end_time - start_time).total_seconds() / 3600  # hours
        
        return {
            "total_stops": len(stops),
            "total_service_time_min": total_service_time,
            "total_late_minutes": total_late_minutes,
            "late_stops_count": late_stops_count,
            "late_stops_ratio": late_stops_count / len(stops) if stops else 0,
            "total_duration_hours": total_duration
        }
    
    @staticmethod
    def find_insertion_position(existing_orders: List[Order], 
                              new_order: Order,
                              travel_times: Dict[Tuple[str, str], TravelTimeResult]) -> int:
        """
        Tìm vị trí tốt nhất để insert order mới vào route
        
        Returns:
            Index position để insert (0 = đầu tiên)
        """
        if not existing_orders:
            return 0
        
        best_position = 0
        best_cost_increase = float('inf')
        
        for pos in range(len(existing_orders) + 1):
            # Calculate cost increase if inserting at this position
            cost_increase = RouteUtils._calculate_insertion_cost_at_position(
                existing_orders, new_order, pos, travel_times
            )
            
            if cost_increase < best_cost_increase:
                best_cost_increase = cost_increase
                best_position = pos
        
        return best_position
    
    @staticmethod
    def _calculate_insertion_cost_at_position(existing_orders: List[Order],
                                            new_order: Order,
                                            position: int,
                                            travel_times: Dict[Tuple[str, str], TravelTimeResult]) -> float:
        """Tính cost increase khi insert tại position"""
        
        # Simplified cost calculation - trong thực tế sẽ phức tạp hơn
        
        # Get neighboring locations
        if position == 0:
            prev_location = "DEPOT"  # Assume starting from depot
        else:
            prev_location = existing_orders[position - 1].dropoff
        
        if position >= len(existing_orders):
            next_location = "DEPOT"  # Assume ending at depot
        else:
            next_location = existing_orders[position].pickup
        
        # Calculate detour cost
        # Original: prev -> next
        original_key = (prev_location, next_location)
        original_travel = travel_times.get(original_key)
        original_cost = original_travel.distance_km if original_travel else 50.0
        
        # New route: prev -> pickup -> dropoff -> next
        pickup_key = (prev_location, new_order.pickup)
        delivery_key = (new_order.pickup, new_order.dropoff)
        continue_key = (new_order.dropoff, next_location)
        
        pickup_travel = travel_times.get(pickup_key)
        delivery_travel = travel_times.get(delivery_key)
        continue_travel = travel_times.get(continue_key)
        
        new_cost = 0.0
        if pickup_travel:
            new_cost += pickup_travel.distance_km
        else:
            new_cost += 30.0  # fallback
        
        if delivery_travel:
            new_cost += delivery_travel.distance_km
        else:
            new_cost += 40.0  # fallback
        
        if continue_travel:
            new_cost += continue_travel.distance_km
        else:
            new_cost += 30.0  # fallback
        
        return new_cost - original_cost
    
    @staticmethod
    def get_unique_locations(orders: List[Order]) -> Set[str]:
        """Lấy set tất cả locations unique từ orders"""
        locations = set()
        for order in orders:
            locations.add(order.pickup)
            locations.add(order.dropoff)
        return locations
    
    @staticmethod
    def estimate_route_duration(orders: List[Order], 
                              travel_times: Dict[Tuple[str, str], TravelTimeResult]) -> timedelta:
        """Ước tính tổng thời gian cho route"""
        
        if not orders:
            return timedelta(0)
        
        total_seconds = 0
        
        # Service times
        for order in orders:
            total_seconds += (order.get_pickup_service_time() + order.get_dropoff_service_time()) * 60
        
        # Travel times (simplified - assumes sequential order)
        for i in range(len(orders)):
            if i == 0:
                # From start to first pickup (estimate)
                total_seconds += 30 * 60  # 30 min default
            else:
                # From previous dropoff to current pickup
                prev_dropoff = orders[i-1].dropoff
                curr_pickup = orders[i].pickup
                travel_key = (prev_dropoff, curr_pickup)
                travel = travel_times.get(travel_key)
                if travel:
                    total_seconds += travel.travel_time_seconds
                else:
                    total_seconds += 30 * 60  # 30 min default
            
            # From pickup to dropoff
            pickup_to_dropoff_key = (orders[i].pickup, orders[i].dropoff)
            pickup_to_dropoff_travel = travel_times.get(pickup_to_dropoff_key)
            if pickup_to_dropoff_travel:
                total_seconds += pickup_to_dropoff_travel.travel_time_seconds
            else:
                total_seconds += 45 * 60  # 45 min default
        
        return timedelta(seconds=total_seconds)
