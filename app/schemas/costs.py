"""Schema cho cấu hình chi phí và breakdown"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class CostConfig(BaseModel):
    """Cấu hình chi phí"""
    fuel_cost_per_km: float = Field(0.25, ge=0, description="Chi phí nhiên liệu (¥/km)")
    avg_consumption_km_per_liter: float = Field(4.0, ge=0, description="Tiêu thụ trung bình (km/lít)")
    
    # Toll costs
    toll_per_km_highway: float = Field(0.15, ge=0, description="Phí cao tốc (¥/km)")
    toll_per_km_urban: float = Field(0.05, ge=0, description="Phí đường thành phố (¥/km)")
    toll_multiplier_peak: float = Field(1.2, ge=1.0, description="Hệ số giờ cao điểm")
    
    # Penalty costs
    late_penalty_per_min: float = Field(2.0, ge=0, description="Phạt trễ (¥/phút)")
    overtime_base_rate: float = Field(1500.0, ge=0, description="Mức overtime cơ bản (¥/giờ)")
    
    # Time-of-day multipliers
    peak_hours_start: int = Field(7, ge=0, le=23, description="Giờ bắt đầu cao điểm")
    peak_hours_end: int = Field(19, ge=0, le=23, description="Giờ kết thúc cao điểm")


class CostBreakdown(BaseModel):
    """Chi tiết breakdown chi phí"""
    fuel_cost: float = Field(0.0, ge=0, description="Chi phí nhiên liệu")
    toll_cost: float = Field(0.0, ge=0, description="Chi phí toll")
    overtime_cost: float = Field(0.0, ge=0, description="Chi phí overtime")
    penalty_cost: float = Field(0.0, ge=0, description="Chi phí phạt trễ")
    total_cost: float = Field(0.0, ge=0, description="Tổng chi phí")
    
    # Metrics
    distance_km: float = Field(0.0, ge=0, description="Tổng khoảng cách (km)")
    highway_km: float = Field(0.0, ge=0, description="Khoảng cách cao tốc (km)")
    overtime_hours: float = Field(0.0, ge=0, description="Số giờ overtime")
    late_minutes: int = Field(0, ge=0, description="Số phút trễ")
    
    # Additional breakdown
    details: Optional[Dict[str, Any]] = Field(None, description="Chi tiết bổ sung")
    
    def __add__(self, other: 'CostBreakdown') -> 'CostBreakdown':
        """Cộng 2 cost breakdown"""
        return CostBreakdown(
            fuel_cost=self.fuel_cost + other.fuel_cost,
            toll_cost=self.toll_cost + other.toll_cost,
            overtime_cost=self.overtime_cost + other.overtime_cost,
            penalty_cost=self.penalty_cost + other.penalty_cost,
            total_cost=self.total_cost + other.total_cost,
            distance_km=self.distance_km + other.distance_km,
            highway_km=self.highway_km + other.highway_km,
            overtime_hours=self.overtime_hours + other.overtime_hours,
            late_minutes=self.late_minutes + other.late_minutes
        )
