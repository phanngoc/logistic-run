"""Schema cho dispatch request/response"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from .orders import Order
from .fleet import Truck
from .costs import CostConfig, CostBreakdown


class WeightConfig(BaseModel):
    """Cấu hình trọng số cho optimization"""
    lambda_late: float = Field(1.0, ge=0, description="Trọng số phạt trễ")
    lambda_ot: float = Field(1.0, ge=0, description="Trọng số overtime")
    lambda_tw: float = Field(10.0, ge=0, description="Trọng số vi phạm time window")
    lambda_priority: float = Field(0.5, ge=0, description="Trọng số ưu tiên đơn hàng")


class DispatchRequest(BaseModel):
    """Request cho dispatch optimization"""
    orders: List[Order] = Field(..., min_items=1, description="Danh sách đơn hàng")
    fleet: List[Truck] = Field(..., min_items=1, description="Danh sách xe")
    costs: CostConfig = Field(default_factory=CostConfig, description="Cấu hình chi phí")
    weights: WeightConfig = Field(default_factory=WeightConfig, description="Trọng số optimization")
    
    # Optional solver settings
    max_iterations: Optional[int] = Field(None, ge=10, le=10000, description="Số iteration tối đa")
    time_limit_seconds: Optional[int] = Field(None, ge=5, le=300, description="Thời gian tối đa (giây)")


class Stop(BaseModel):
    """Điểm dừng trong route"""
    location: str = Field(..., description="Tên địa điểm")
    order_id: Optional[str] = Field(None, description="Mã đơn hàng (nếu có)")
    stop_type: str = Field(..., description="Loại: pickup, dropoff, depot")
    
    # Timing
    eta: datetime = Field(..., description="Thời gian đến dự kiến")
    etd: datetime = Field(..., description="Thời gian đi dự kiến")
    service_time_min: int = Field(0, ge=0, description="Thời gian xử lý (phút)")
    
    # Status
    is_late: bool = Field(False, description="Có bị trễ không")
    late_minutes: int = Field(0, ge=0, description="Số phút trễ")
    
    # Notes
    note: Optional[str] = Field(None, description="Ghi chú")
    gate_dwell_min: Optional[int] = Field(None, description="Thời gian chờ cảng (phút)")


class Route(BaseModel):
    """Route cho 1 xe"""
    truck_id: str = Field(..., description="Mã xe")
    driver_id: Optional[str] = Field(None, description="Mã tài xế")
    
    # Stops
    stops: List[Stop] = Field(..., description="Danh sách điểm dừng")
    
    # Metrics
    total_distance_km: float = Field(0.0, ge=0, description="Tổng khoảng cách")
    total_duration_hours: float = Field(0.0, ge=0, description="Tổng thời gian")
    
    # Costs
    cost_breakdown: CostBreakdown = Field(..., description="Chi tiết chi phí")
    
    # Explanation
    explain: Optional[str] = Field(None, description="Giải thích route")
    score: Optional[float] = Field(None, description="Điểm số route")
    
    # Orders served
    order_ids: List[str] = Field(default_factory=list, description="Danh sách đơn đã phục vụ")


class KPI(BaseModel):
    """KPI tổng quát của solution"""
    total_orders: int = Field(0, ge=0, description="Tổng số đơn")
    served_orders: int = Field(0, ge=0, description="Số đơn được phục vụ")
    unserved_orders: int = Field(0, ge=0, description="Số đơn chưa phục vụ")
    
    total_distance_km: float = Field(0.0, ge=0, description="Tổng khoảng cách")
    total_cost: float = Field(0.0, ge=0, description="Tổng chi phí")
    
    late_orders: int = Field(0, ge=0, description="Số đơn trễ")
    overtime_hours: float = Field(0.0, ge=0, description="Tổng overtime")
    
    utilization_rate: float = Field(0.0, ge=0, le=1, description="Tỷ lệ sử dụng xe")
    avg_route_score: Optional[float] = Field(None, description="Điểm trung bình route")


class DispatchResponse(BaseModel):
    """Response từ dispatch optimization"""
    success: bool = Field(..., description="Có thành công không")
    message: Optional[str] = Field(None, description="Thông báo")
    
    # Solution
    routes: List[Route] = Field(default_factory=list, description="Danh sách route")
    unserved_orders: List[str] = Field(default_factory=list, description="Đơn chưa phục vụ")
    
    # KPIs
    kpi: KPI = Field(..., description="Chỉ số KPI")
    
    # Solver info
    solve_time_seconds: float = Field(0.0, ge=0, description="Thời gian tính toán")
    iterations: int = Field(0, ge=0, description="Số iteration đã chạy")
    algorithm: str = Field("unknown", description="Thuật toán sử dụng")
    
    # Additional info
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata bổ sung")
