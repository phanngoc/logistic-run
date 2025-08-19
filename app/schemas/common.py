"""Common schemas cho các model chung"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator


class Location(BaseModel):
    """Địa điểm với tọa độ hoặc tên định danh"""
    name: str = Field(..., description="Tên địa điểm (VD: PORT_A_GATE_3, WAREHOUSE_X)")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Vĩ độ")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Kinh độ") 
    address: Optional[str] = Field(None, description="Địa chỉ chi tiết")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Tên địa điểm không được rỗng")
        return v.strip().upper()


class TimeWindow(BaseModel):
    """Khung thời gian cho pickup/delivery"""
    start: datetime = Field(..., description="Thời điểm sớm nhất")
    end: datetime = Field(..., description="Thời điểm muộn nhất")
    
    @validator('end')
    def validate_end_after_start(cls, v, values):
        if 'start' in values and v <= values['start']:
            raise ValueError("Thời điểm kết thúc phải sau thời điểm bắt đầu")
        return v


ContainerSize = Literal["20", "40"]
"""Kích thước container: 20ft hoặc 40ft"""
