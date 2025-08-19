"""Schema cho đội xe và tài xế (fleet)"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator

from .common import ContainerSize


class Truck(BaseModel):
    """Xe tải và tài xế"""
    truck_id: str = Field(..., description="Mã xe")
    driver_id: Optional[str] = Field(None, description="Mã tài xế")
    start_location: str = Field(..., description="Vị trí xuất phát")
    
    # Shift timing
    shift_start: datetime = Field(..., description="Bắt đầu ca làm")
    shift_end: datetime = Field(..., description="Kết thúc ca làm")
    
    # Overtime settings
    overtime_threshold_min: int = Field(600, ge=0, description="Ngưỡng overtime (phút)")
    overtime_rate_per_hour: float = Field(1500.0, ge=0, description="Giá overtime (¥/giờ)")
    
    # Capacity
    allowed_sizes: List[ContainerSize] = Field(..., description="Kích thước container được phép")
    max_orders_per_day: int = Field(10, ge=1, le=50, description="Số đơn tối đa/ngày")
    
    # Optional fields
    fuel_efficiency_km_per_liter: Optional[float] = Field(None, ge=0, description="Hiệu suất nhiên liệu")
    driver_skill_level: int = Field(3, ge=1, le=5, description="Trình độ tài xế (1-5)")
    
    @validator('truck_id')
    def validate_truck_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Mã xe không được rỗng")
        return v.strip().upper()
    
    @validator('start_location')
    def validate_start_location(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Vị trí xuất phát không được rỗng")
        return v.strip().upper()
        
    @validator('shift_end')
    def validate_shift_end_after_start(cls, v, values):
        if 'shift_start' in values and v <= values['shift_start']:
            raise ValueError("Kết thúc ca phải sau bắt đầu ca")
        return v
    
    @validator('allowed_sizes')
    def validate_allowed_sizes(cls, v):
        if not v:
            raise ValueError("Phải có ít nhất 1 kích thước container được phép")
        return v
        
    def can_handle_container(self, size: ContainerSize) -> bool:
        """Kiểm tra xe có thể xử lý container size này không"""
        return size in self.allowed_sizes
    
    def get_shift_duration_hours(self) -> float:
        """Tính thời gian ca làm (giờ)"""
        duration = self.shift_end - self.shift_start
        return duration.total_seconds() / 3600
    
    def get_overtime_threshold_hours(self) -> float:
        """Ngưỡng overtime (giờ)"""
        return self.overtime_threshold_min / 60


class Fleet(BaseModel):
    """Đội xe"""
    trucks: List[Truck] = Field(..., min_items=1, description="Danh sách xe")
    
    @validator('trucks')
    def validate_unique_truck_ids(cls, v):
        truck_ids = [truck.truck_id for truck in v]
        if len(truck_ids) != len(set(truck_ids)):
            raise ValueError("Mã xe phải unique")
        return v
