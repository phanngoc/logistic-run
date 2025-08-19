"""Service cho Port API - lấy thông tin queue/gate/berth"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass
import logging
from statistics import mean

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class GateDwellInfo:
    """Thông tin thời gian chờ tại cảng"""
    port_name: str
    gate_id: Optional[str]
    expected_dwell_minutes: int
    confidence: float = 0.5  # 0-1, độ tin cậy của prediction
    source: str = "unknown"  # api, forecast, default
    queue_length: Optional[int] = None
    last_updated: Optional[datetime] = None
    
    def is_reliable(self) -> bool:
        """Kiểm tra thông tin có đáng tin cậy không"""
        return self.confidence >= 0.7


class PortApiService:
    """Service tích hợp với Port API và dự báo gate dwell"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Historical data for forecasting (sẽ load từ DB trong thực tế)
        self.historical_data: Dict[str, List[Dict]] = {}
        
        # Moving averages cache
        self.ma_cache: Dict[str, float] = {}
        
        # Initialize with mock historical data
        self._init_mock_historical_data()
    
    def _init_mock_historical_data(self):
        """Khởi tạo mock historical data cho forecasting"""
        # Mock data - trong thực tế sẽ load từ database
        base_data = {
            "PORT_A_GATE_1": [
                {"hour": h, "dow": d, "dwell_minutes": 20 + (h % 8) * 3 + (d % 3) * 5}
                for h in range(24) for d in range(7)
            ],
            "PORT_A_GATE_2": [
                {"hour": h, "dow": d, "dwell_minutes": 25 + (h % 6) * 4 + (d % 2) * 3}
                for h in range(24) for d in range(7)
            ],
            "PORT_A_GATE_3": [
                {"hour": h, "dow": d, "dwell_minutes": 30 + (h % 10) * 2 + (d % 4) * 4}
                for h in range(24) for d in range(7)
            ],
            "PORT_B_GATE_1": [
                {"hour": h, "dow": d, "dwell_minutes": 35 + (h % 7) * 3 + (d % 5) * 2}
                for h in range(24) for d in range(7)
            ]
        }
        self.historical_data = base_data
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.client.aclose()
    
    def _extract_port_gate(self, location: str) -> tuple[str, Optional[str]]:
        """Trích xuất port name và gate ID từ location string"""
        location = location.upper().strip()
        
        # Parse các format: PORT_A_GATE_1, PORT_A, PORT_B_GATE_2, etc.
        if "_GATE_" in location:
            parts = location.split("_GATE_")
            port_name = parts[0]
            gate_id = f"GATE_{parts[1]}" if len(parts) > 1 else None
        else:
            port_name = location
            gate_id = None
        
        return port_name, gate_id
    
    async def _call_port_api(self, port_name: str, gate_id: Optional[str] = None) -> Optional[GateDwellInfo]:
        """Gọi Port API để lấy thông tin real-time queue"""
        if not settings.port_api_key:
            logger.warning("Port API key not configured")
            return None
        
        try:
            url = f"{settings.port_api_base_url}/queue-status"
            params = {
                "api_key": settings.port_api_key,
                "port": port_name,
            }
            
            if gate_id:
                params["gate"] = gate_id
            
            response = await self.client.get(url, params=params)
            
            # Mock response since we don't have real Port API
            if response.status_code == 404:
                # API không có data cho port này
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            queue_length = data.get("queue_length", 0)
            expected_wait_min = data.get("expected_wait_minutes", 30)
            confidence = data.get("confidence", 0.8)
            
            return GateDwellInfo(
                port_name=port_name,
                gate_id=gate_id,
                expected_dwell_minutes=expected_wait_min,
                confidence=confidence,
                source="api",
                queue_length=queue_length,
                last_updated=datetime.now()
            )
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(f"No API data for {port_name}")
                return None
            logger.error(f"Port API HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"Port API error: {e}")
            return None
    
    def _get_historical_forecast(self, port_name: str, gate_id: Optional[str], target_time: datetime) -> Optional[GateDwellInfo]:
        """Dự báo dựa trên historical data với Moving Average"""
        
        # Tạo location key
        location_key = f"{port_name}_{gate_id}" if gate_id else port_name
        if location_key not in self.historical_data:
            # Try without gate_id
            if port_name not in self.historical_data:
                return None
            location_key = port_name
        
        # Extract time features
        hour_of_day = target_time.hour
        day_of_week = target_time.weekday()  # 0=Monday
        
        # Get historical data for similar time slots
        historical_records = self.historical_data[location_key]
        
        # Filter relevant records (same hour ± 1, same day of week ± 1)
        relevant_records = []
        for record in historical_records:
            if (abs(record["hour"] - hour_of_day) <= 1 and 
                abs(record["dow"] - day_of_week) <= 1):
                relevant_records.append(record["dwell_minutes"])
        
        if not relevant_records:
            # Fallback: use all records for that hour
            relevant_records = [r["dwell_minutes"] for r in historical_records if r["hour"] == hour_of_day]
        
        if not relevant_records:
            return None
        
        # Calculate moving average
        predicted_dwell = int(mean(relevant_records))
        
        # Adjust for time-of-day patterns
        if 6 <= hour_of_day <= 10:  # Morning rush
            predicted_dwell = int(predicted_dwell * 1.2)
        elif 16 <= hour_of_day <= 19:  # Evening rush
            predicted_dwell = int(predicted_dwell * 1.1)
        elif 22 <= hour_of_day or hour_of_day <= 5:  # Night
            predicted_dwell = int(predicted_dwell * 0.8)
        
        confidence = 0.6 if len(relevant_records) >= 5 else 0.4
        
        return GateDwellInfo(
            port_name=port_name,
            gate_id=gate_id,
            expected_dwell_minutes=predicted_dwell,
            confidence=confidence,
            source="forecast",
            last_updated=datetime.now()
        )
    
    def _get_default_dwell(self, port_name: str, gate_id: Optional[str]) -> GateDwellInfo:
        """Lấy default dwell time theo port"""
        
        # Check specific port settings
        default_minutes = settings.gate_dwell_by_port.get(port_name, settings.default_gate_dwell_min)
        
        # Adjust based on gate if available
        if gate_id and "GATE_1" in gate_id:
            default_minutes = int(default_minutes * 0.9)  # Gate 1 thường nhanh hơn
        elif gate_id and "GATE_3" in gate_id:
            default_minutes = int(default_minutes * 1.1)  # Gate 3 thường chậm hơn
        
        return GateDwellInfo(
            port_name=port_name,
            gate_id=gate_id,
            expected_dwell_minutes=default_minutes,
            confidence=0.3,
            source="default"
        )
    
    async def get_gate_dwell_info(self, location: str, expected_arrival_time: Optional[datetime] = None) -> GateDwellInfo:
        """
        Lấy thông tin gate dwell cho location tại thời điểm arrival_time
        
        Args:
            location: Tên địa điểm (VD: PORT_A_GATE_1)
            expected_arrival_time: Thời điểm đến dự kiến
            
        Returns:
            GateDwellInfo với prediction tốt nhất có thể
        """
        
        if expected_arrival_time is None:
            expected_arrival_time = datetime.now()
        
        port_name, gate_id = self._extract_port_gate(location)
        
        # Strategy 1: Try Port API (real-time)
        api_result = await self._call_port_api(port_name, gate_id)
        if api_result and api_result.is_reliable():
            logger.debug(f"Using API data for {location}")
            return api_result
        
        # Strategy 2: Historical forecast
        forecast_result = self._get_historical_forecast(port_name, gate_id, expected_arrival_time)
        if forecast_result and forecast_result.confidence >= 0.5:
            logger.debug(f"Using forecast for {location}")
            return forecast_result
        
        # Strategy 3: Use API result even if low confidence
        if api_result:
            logger.debug(f"Using low-confidence API data for {location}")
            return api_result
        
        # Strategy 4: Use forecast even if low confidence  
        if forecast_result:
            logger.debug(f"Using low-confidence forecast for {location}")
            return forecast_result
        
        # Strategy 5: Default fallback
        logger.debug(f"Using default values for {location}")
        return self._get_default_dwell(port_name, gate_id)
    
    async def get_bulk_gate_dwell_info(self, locations: List[str], expected_arrival_time: Optional[datetime] = None) -> Dict[str, GateDwellInfo]:
        """
        Lấy gate dwell info cho nhiều locations cùng lúc
        
        Args:
            locations: Danh sách địa điểm
            expected_arrival_time: Thời điểm đến dự kiến
            
        Returns:
            Dict với key là location và value là GateDwellInfo
        """
        
        # Filter unique port locations
        port_locations = [loc for loc in locations if "PORT" in loc.upper()]
        
        if not port_locations:
            return {}
        
        # Execute parallel requests
        tasks = [
            self.get_gate_dwell_info(location, expected_arrival_time)
            for location in port_locations
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        dwell_info = {}
        for location, result in zip(port_locations, results):
            if isinstance(result, Exception):
                logger.error(f"Error getting dwell info for {location}: {result}")
                # Use default fallback
                port_name, gate_id = self._extract_port_gate(location)
                result = self._get_default_dwell(port_name, gate_id)
            
            dwell_info[location] = result
        
        return dwell_info
    
    def add_historical_record(self, location: str, arrival_time: datetime, actual_dwell_minutes: int):
        """
        Thêm historical record để cải thiện forecasting
        
        Args:
            location: Địa điểm
            arrival_time: Thời điểm đến thực tế
            actual_dwell_minutes: Thời gian chờ thực tế
        """
        port_name, gate_id = self._extract_port_gate(location)
        location_key = f"{port_name}_{gate_id}" if gate_id else port_name
        
        if location_key not in self.historical_data:
            self.historical_data[location_key] = []
        
        record = {
            "timestamp": arrival_time.isoformat(),
            "hour": arrival_time.hour,
            "dow": arrival_time.weekday(),
            "dwell_minutes": actual_dwell_minutes
        }
        
        self.historical_data[location_key].append(record)
        
        # Keep only recent records (rolling window)
        max_records = 1000
        if len(self.historical_data[location_key]) > max_records:
            self.historical_data[location_key] = self.historical_data[location_key][-max_records:]
        
        logger.info(f"Added historical record for {location}: {actual_dwell_minutes}min")
    
    def get_location_statistics(self, location: str) -> Dict:
        """Lấy statistics cho location để debugging/monitoring"""
        port_name, gate_id = self._extract_port_gate(location)
        location_key = f"{port_name}_{gate_id}" if gate_id else port_name
        
        if location_key not in self.historical_data:
            return {"error": "No historical data"}
        
        records = self.historical_data[location_key]
        dwell_times = [r["dwell_minutes"] for r in records]
        
        if not dwell_times:
            return {"error": "No dwell time data"}
        
        stats = {
            "count": len(dwell_times),
            "mean": mean(dwell_times),
            "min": min(dwell_times),
            "max": max(dwell_times),
            "recent_avg": mean(dwell_times[-10:]) if len(dwell_times) >= 10 else mean(dwell_times)
        }
        
        return stats
