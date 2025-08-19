"""Cost Engine - tính toán chi phí cho routes"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

from ..schemas.costs import CostConfig, CostBreakdown
from ..schemas.fleet import Truck
from ..schemas.orders import Order
from .travel_time import TravelTimeResult

logger = logging.getLogger(__name__)


class CostEngine:
    """Engine tính toán chi phí tổng thể cho routes"""
    
    def __init__(self, cost_config: CostConfig):
        self.config = cost_config
    
    def calculate_fuel_cost(self, distance_km: float, truck: Optional[Truck] = None) -> float:
        """
        Tính chi phí nhiên liệu
        
        Args:
            distance_km: Khoảng cách (km)
            truck: Thông tin xe (để lấy fuel efficiency nếu có)
            
        Returns:
            Chi phí nhiên liệu (¥)
        """
        if distance_km <= 0:
            return 0.0
        
        # Sử dụng fuel efficiency riêng của xe nếu có
        if truck and truck.fuel_efficiency_km_per_liter:
            consumption_km_per_liter = truck.fuel_efficiency_km_per_liter
        else:
            consumption_km_per_liter = self.config.avg_consumption_km_per_liter
        
        # Tính lượng nhiên liệu cần (lít)
        fuel_liters = distance_km / consumption_km_per_liter
        
        # Convert fuel_cost_per_km to fuel_cost_per_liter
        fuel_cost_per_liter = self.config.fuel_cost_per_km * consumption_km_per_liter
        
        return fuel_liters * fuel_cost_per_liter
    
    def calculate_toll_cost(self, travel_result: TravelTimeResult, departure_time: datetime) -> float:
        """
        Tính chi phí toll
        
        Args:
            travel_result: Kết quả travel time với thông tin đường
            departure_time: Thời điểm khởi hành
            
        Returns:
            Chi phí toll (¥)
        """
        if travel_result.distance_km <= 0:
            return 0.0
        
        highway_km = travel_result.highway_distance_km
        urban_km = travel_result.distance_km - highway_km
        
        # Base toll cost
        highway_toll = highway_km * self.config.toll_per_km_highway
        urban_toll = urban_km * self.config.toll_per_km_urban
        base_toll = highway_toll + urban_toll
        
        # Time-of-day multiplier (peak hours)
        hour = departure_time.hour
        if self.config.peak_hours_start <= hour <= self.config.peak_hours_end:
            multiplier = self.config.toll_multiplier_peak
        else:
            multiplier = 1.0
        
        return base_toll * multiplier
    
    def calculate_overtime_cost(self, truck: Truck, total_working_minutes: int) -> Tuple[float, float]:
        """
        Tính chi phí overtime
        
        Args:
            truck: Thông tin xe và tài xế
            total_working_minutes: Tổng thời gian làm việc (phút)
            
        Returns:
            Tuple (overtime_cost, overtime_hours)
        """
        if total_working_minutes <= truck.overtime_threshold_min:
            return 0.0, 0.0
        
        overtime_minutes = total_working_minutes - truck.overtime_threshold_min
        overtime_hours = overtime_minutes / 60.0
        
        overtime_cost = overtime_hours * truck.overtime_rate_per_hour
        
        return overtime_cost, overtime_hours
    
    def calculate_late_penalty(self, late_minutes: int) -> float:
        """
        Tính chi phí phạt trễ
        
        Args:
            late_minutes: Số phút trễ
            
        Returns:
            Chi phí phạt (¥)
        """
        if late_minutes <= 0:
            return 0.0
        
        return late_minutes * self.config.late_penalty_per_min
    
    def calculate_route_cost(self, 
                           truck: Truck,
                           route_segments: List[Dict],
                           orders_info: Dict[str, Order]) -> CostBreakdown:
        """
        Tính chi phí tổng cho 1 route
        
        Args:
            truck: Thông tin xe
            route_segments: List các segment trong route
                Format: [{"from": "A", "to": "B", "travel_result": TravelTimeResult, 
                         "departure_time": datetime, "order_id": str, "late_minutes": int}]
            orders_info: Dict mapping order_id -> Order
            
        Returns:
            CostBreakdown chi tiết
        """
        
        total_fuel_cost = 0.0
        total_toll_cost = 0.0
        total_penalty_cost = 0.0
        total_distance_km = 0.0
        total_highway_km = 0.0
        total_late_minutes = 0
        
        # Tính chi phí từng segment
        for segment in route_segments:
            travel_result = segment["travel_result"]
            departure_time = segment["departure_time"]
            late_minutes = segment.get("late_minutes", 0)
            
            # Fuel cost
            fuel_cost = self.calculate_fuel_cost(travel_result.distance_km, truck)
            total_fuel_cost += fuel_cost
            
            # Toll cost  
            toll_cost = self.calculate_toll_cost(travel_result, departure_time)
            total_toll_cost += toll_cost
            
            # Penalty cost
            penalty_cost = self.calculate_late_penalty(late_minutes)
            total_penalty_cost += penalty_cost
            
            # Accumulate distances
            total_distance_km += travel_result.distance_km
            total_highway_km += travel_result.highway_distance_km
            total_late_minutes += late_minutes
        
        # Tính overtime cost
        total_working_minutes = self._calculate_total_working_time(route_segments, truck)
        overtime_cost, overtime_hours = self.calculate_overtime_cost(truck, total_working_minutes)
        
        # Total cost
        total_cost = total_fuel_cost + total_toll_cost + overtime_cost + total_penalty_cost
        
        return CostBreakdown(
            fuel_cost=total_fuel_cost,
            toll_cost=total_toll_cost,
            overtime_cost=overtime_cost,
            penalty_cost=total_penalty_cost,
            total_cost=total_cost,
            distance_km=total_distance_km,
            highway_km=total_highway_km,
            overtime_hours=overtime_hours,
            late_minutes=total_late_minutes,
            details={
                "segments_count": len(route_segments),
                "working_minutes": total_working_minutes,
                "truck_id": truck.truck_id
            }
        )
    
    def _calculate_total_working_time(self, route_segments: List[Dict], truck: Truck) -> int:
        """Tính tổng thời gian làm việc của route (từ start đến end)"""
        
        if not route_segments:
            return 0
        
        # Thời gian bắt đầu = shift start hoặc departure time của segment đầu
        start_time = min(truck.shift_start, route_segments[0]["departure_time"])
        
        # Thời gian kết thúc = arrival time + service time của segment cuối
        last_segment = route_segments[-1]
        travel_result = last_segment["travel_result"]
        departure_time = last_segment["departure_time"]
        
        end_time = departure_time + timedelta(seconds=travel_result.travel_time_seconds)
        
        # Add service time nếu có
        if "service_time_min" in last_segment:
            end_time += timedelta(minutes=last_segment["service_time_min"])
        
        # Tính tổng thời gian làm việc
        working_duration = end_time - start_time
        working_minutes = int(working_duration.total_seconds() / 60)
        
        return max(0, working_minutes)
    
    def calculate_insertion_cost(self,
                               truck: Truck,
                               existing_route_segments: List[Dict],
                               new_order: Order,
                               pickup_travel: TravelTimeResult,
                               delivery_travel: TravelTimeResult,
                               pickup_time: datetime,
                               delivery_time: datetime) -> CostBreakdown:
        """
        Tính chi phí khi insert order mới vào route hiện tại
        
        Args:
            truck: Xe
            existing_route_segments: Route hiện tại
            new_order: Order mới
            pickup_travel: Travel time đến pickup
            delivery_travel: Travel time từ pickup đến delivery
            pickup_time: Thời gian pickup dự kiến
            delivery_time: Thời gian delivery dự kiến
            
        Returns:
            Chi phí bổ sung khi thêm order này
        """
        
        # Tính late penalty cho order mới
        late_minutes = 0
        if delivery_time > new_order.tw_end:
            late_penalty_minutes = int((delivery_time - new_order.tw_end).total_seconds() / 60)
            late_minutes = max(0, late_penalty_minutes)
        
        # Tạo segments cho order mới
        new_segments = [
            {
                "from": "current_position",
                "to": new_order.pickup,
                "travel_result": pickup_travel,
                "departure_time": pickup_time,
                "order_id": new_order.order_id,
                "late_minutes": 0
            },
            {
                "from": new_order.pickup,
                "to": new_order.dropoff,
                "travel_result": delivery_travel,
                "departure_time": pickup_time + timedelta(minutes=new_order.get_pickup_service_time()),
                "order_id": new_order.order_id,
                "late_minutes": late_minutes,
                "service_time_min": new_order.get_dropoff_service_time()
            }
        ]
        
        # Tính cost cho order mới
        new_order_cost = self.calculate_route_cost(truck, new_segments, {new_order.order_id: new_order})
        
        return new_order_cost
    
    def estimate_route_score(self, cost_breakdown: CostBreakdown, weights: Dict[str, float]) -> float:
        """
        Tính điểm score cho route dựa trên cost và weights
        
        Args:
            cost_breakdown: Chi phí chi tiết
            weights: Trọng số cho từng thành phần
            
        Returns:
            Score (càng thấp càng tốt)
        """
        
        # Base score từ total cost
        score = cost_breakdown.total_cost
        
        # Penalty cho late delivery
        if cost_breakdown.late_minutes > 0:
            score += weights.get("lambda_late", 1.0) * cost_breakdown.late_minutes
        
        # Penalty cho overtime  
        if cost_breakdown.overtime_hours > 0:
            score += weights.get("lambda_ot", 1.0) * cost_breakdown.overtime_hours * 100
        
        # Bonus cho distance efficiency (lower distance = better score)
        distance_efficiency = 1.0 / (1.0 + cost_breakdown.distance_km / 100.0)
        score *= (2.0 - distance_efficiency)
        
        return score
    
    def compare_routes(self, route1_cost: CostBreakdown, route2_cost: CostBreakdown, weights: Dict[str, float]) -> int:
        """
        So sánh 2 routes
        
        Returns:
            -1 nếu route1 tốt hơn, 1 nếu route2 tốt hơn, 0 nếu bằng nhau
        """
        
        score1 = self.estimate_route_score(route1_cost, weights)
        score2 = self.estimate_route_score(route2_cost, weights)
        
        if score1 < score2:
            return -1
        elif score1 > score2:
            return 1
        else:
            return 0
