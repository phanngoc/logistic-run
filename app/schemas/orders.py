"""Schema cho đơn hàng (orders)"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator

from .common import Location, ContainerSize


class Order(BaseModel):
    """Đơn hàng cần vận chuyển"""
    order_id: str = Field(..., description="Mã đơn hàng")
    pickup: str = Field(..., description="Địa điểm lấy hàng")
    dropoff: str = Field(..., description="Địa điểm giao hàng") 
    container_size: ContainerSize = Field(..., description="Kích thước container (20/40)")
    
    # Time windows
    tw_start: datetime = Field(..., description="Thời điểm sớm nhất có thể pickup")
    tw_end: datetime = Field(..., description="Thời điểm muộn nhất phải delivery")
    
    # Service time
    service_time_min: int = Field(20, ge=0, le=240, description="Thời gian xử lý tại địa điểm (phút)")
    
    # Priority
    priority: int = Field(1, ge=1, le=5, description="Độ ưu tiên (1=thấp, 5=cao)")
    
    # Optional fields  
    pickup_service_time_min: Optional[int] = Field(None, ge=0, le=240, description="Thời gian pickup riêng")
    dropoff_service_time_min: Optional[int] = Field(None, ge=0, le=240, description="Thời gian dropoff riêng")
    
    @validator('order_id')
    def validate_order_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Mã đơn hàng không được rỗng")
        return v.strip().upper()
    
    @validator('pickup', 'dropoff')
    def validate_locations(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Địa điểm không được rỗng")
        return v.strip().upper()
        
    @validator('tw_end')
    def validate_tw_end_after_start(cls, v, values):
        if 'tw_start' in values and v <= values['tw_start']:
            raise ValueError("Thời điểm kết thúc phải sau thời điểm bắt đầu")
        return v
    
    def get_pickup_service_time(self) -> int:
        """Lấy thời gian pickup, fallback về service_time_min"""
        return self.pickup_service_time_min or self.service_time_min
    
    def get_dropoff_service_time(self) -> int:
        """Lấy thời gian dropoff, fallback về service_time_min"""
        return self.dropoff_service_time_min or self.service_time_min


class OrderRequest(BaseModel):
    """Request chứa danh sách đơn hàng"""
    orders: List[Order] = Field(..., min_items=1, description="Danh sách đơn hàng")
    
    @validator('orders')
    def validate_unique_order_ids(cls, v):
        order_ids = [order.order_id for order in v]
        if len(order_ids) != len(set(order_ids)):
            raise ValueError("Mã đơn hàng phải unique")
        return v
