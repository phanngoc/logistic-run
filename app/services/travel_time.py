"""Service cho tính toán travel time với Azure Maps"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import logging

import redis.asyncio as aioredis
import httpx
from geopy.distance import geodesic

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class TravelTimeResult:
    """Kết quả travel time"""
    travel_time_seconds: int
    distance_meters: int
    highway_distance_meters: int = 0
    traffic_delay_seconds: int = 0
    route_summary: Optional[str] = None
    
    @property
    def travel_time_minutes(self) -> float:
        return self.travel_time_seconds / 60
    
    @property
    def distance_km(self) -> float:
        return self.distance_meters / 1000
    
    @property
    def highway_distance_km(self) -> float:
        return self.highway_distance_meters / 1000


class TravelTimeService:
    """Service tính travel time với Azure Maps + caching"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Location coordinates cache (tên địa điểm -> lat,lng)
        self.location_cache: Dict[str, Tuple[float, float]] = {}
        
        # Predefined locations (có thể load từ DB)
        self._init_predefined_locations()
    
    def _init_predefined_locations(self):
        """Khởi tạo coordinates cho các địa điểm định sẵn"""
        # Mock coordinates cho MVP - trong thực tế sẽ load từ DB
        self.location_cache.update({
            "PORT_A_GATE_1": (35.6762, 139.6503),  # Tokyo Port
            "PORT_A_GATE_2": (35.6760, 139.6505),
            "PORT_A_GATE_3": (35.6758, 139.6507),
            "PORT_B_GATE_1": (34.6937, 135.5023),  # Osaka Port
            "WAREHOUSE_X": (35.6895, 139.6917),   # Tokyo area
            "WAREHOUSE_Y": (35.6584, 139.7016),
            "DEPOT_1": (35.6812, 139.7671),       # Tokyo Depot
            "DEPOT_2": (34.6851, 135.5141),       # Osaka Depot
        })
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._init_redis()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.client.aclose()
        if self.redis_client:
            await self.redis_client.close()
    
    async def _init_redis(self):
        """Khởi tạo Redis connection"""
        try:
            if settings.redis_url:
                self.redis_client = aioredis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, caching disabled")
            self.redis_client = None
    
    def _get_cache_key(self, origin: str, destination: str, departure_time: datetime) -> str:
        """Tạo cache key cho travel time"""
        # Bucket time thành 15-minute intervals
        bucket_time = self._bucket_time(departure_time)
        key_data = f"{origin}|{destination}|{bucket_time}"
        return f"tt:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def _bucket_time(self, dt: datetime) -> str:
        """Bucket thời gian thành intervals 15 phút"""
        minute_bucket = (dt.minute // settings.time_bucket_minutes) * settings.time_bucket_minutes
        bucketed_dt = dt.replace(minute=minute_bucket, second=0, microsecond=0)
        return bucketed_dt.isoformat()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[TravelTimeResult]:
        """Lấy kết quả từ cache"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                # With decode_responses=True, cached_data is already a string
                data = json.loads(cached_data) if isinstance(cached_data, str) else json.loads(cached_data.decode())
                return TravelTimeResult(**data)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        
        return None
    
    async def _save_to_cache(self, cache_key: str, result: TravelTimeResult):
        """Lưu kết quả vào cache"""
        if not self.redis_client:
            return
        
        try:
            data = {
                "travel_time_seconds": result.travel_time_seconds,
                "distance_meters": result.distance_meters,
                "highway_distance_meters": result.highway_distance_meters,
                "traffic_delay_seconds": result.traffic_delay_seconds,
                "route_summary": result.route_summary
            }
            await self.redis_client.setex(
                cache_key, 
                settings.cache_ttl_seconds, 
                json.dumps(data)
            )
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def _get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Lấy coordinates từ tên địa điểm"""
        return self.location_cache.get(location.upper())
    
    async def _calculate_haversine_fallback(self, origin: str, destination: str) -> TravelTimeResult:
        """Fallback calculation using Haversine distance"""
        origin_coords = self._get_coordinates(origin)
        dest_coords = self._get_coordinates(destination)
        
        if not origin_coords or not dest_coords:
            # Default fallback nếu không có coordinates
            return TravelTimeResult(
                travel_time_seconds=3600,  # 1 hour default
                distance_meters=50000,     # 50km default
                route_summary="fallback_no_coords"
            )
        
        # Tính khoảng cách Haversine
        distance_km = geodesic(origin_coords, dest_coords).kilometers
        
        # Ước tính travel time (40 km/h average)
        travel_time_hours = distance_km / 40.0
        travel_time_seconds = int(travel_time_hours * 3600)
        
        # Ước tính 60% highway cho khoảng cách > 20km
        highway_ratio = 0.6 if distance_km > 20 else 0.2
        highway_distance_meters = int(distance_km * 1000 * highway_ratio)
        
        return TravelTimeResult(
            travel_time_seconds=travel_time_seconds,
            distance_meters=int(distance_km * 1000),
            highway_distance_meters=highway_distance_meters,
            route_summary="haversine_fallback"
        )
    
    async def _call_azure_maps_api(self, origin: str, destination: str, departure_time: datetime) -> Optional[TravelTimeResult]:
        """Gọi Azure Maps Route API"""
        if not settings.azure_maps_key:
            logger.warning("Azure Maps key not configured")
            return None
        
        origin_coords = self._get_coordinates(origin)
        dest_coords = self._get_coordinates(destination)
        
        if not origin_coords or not dest_coords:
            logger.warning(f"Missing coordinates for {origin} or {destination}")
            return None
        
        try:
            url = f"{settings.azure_maps_base_url}/route/directions/json"
            params = {
                "api-version": "1.0",
                "subscription-key": settings.azure_maps_key,
                "query": f"{origin_coords[0]},{origin_coords[1]}:{dest_coords[0]},{dest_coords[1]}",
                "departAt": departure_time.isoformat(),
                "traffic": "true",
                "routeType": "fastest",
                "travelMode": "truck"  # Truck-specific routing
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            routes = data.get("routes", [])
            
            if not routes:
                logger.warning("No routes returned from Azure Maps")
                return None
            
            route = routes[0]  # Lấy route đầu tiên
            summary = route.get("summary", {})
            
            travel_time_seconds = summary.get("travelTimeInSeconds", 0)
            distance_meters = summary.get("lengthInMeters", 0)
            traffic_delay_seconds = summary.get("trafficDelayInSeconds", 0)
            
            # Ước tính highway distance từ route legs (simplified)
            highway_distance_meters = self._estimate_highway_distance(route, distance_meters)
            
            return TravelTimeResult(
                travel_time_seconds=travel_time_seconds,
                distance_meters=distance_meters,
                highway_distance_meters=highway_distance_meters,
                traffic_delay_seconds=traffic_delay_seconds,
                route_summary="azure_maps"
            )
            
        except Exception as e:
            logger.error(f"Azure Maps API error: {e}")
            return None
    
    def _estimate_highway_distance(self, route_data: dict, total_distance: int) -> int:
        """Ước tính highway distance từ route data"""
        # Simplified estimation - trong thực tế sẽ parse route legs chi tiết
        legs = route_data.get("legs", [])
        if not legs:
            # Fallback: assume 40% highway for distance > 30km
            return int(total_distance * 0.4) if total_distance > 30000 else 0
        
        # Mock logic - thực tế sẽ phân tích road types trong legs
        highway_ratio = 0.5 if total_distance > 20000 else 0.1
        return int(total_distance * highway_ratio)
    
    async def get_travel_time(self, origin: str, destination: str, departure_time: datetime) -> TravelTimeResult:
        """
        Lấy travel time từ origin -> destination tại thời điểm departure_time
        
        Args:
            origin: Tên địa điểm xuất phát
            destination: Tên địa điểm đích
            departure_time: Thời điểm khởi hành
            
        Returns:
            TravelTimeResult với thông tin chi tiết
        """
        # Normalize tên địa điểm
        origin = origin.upper().strip()
        destination = destination.upper().strip()
        
        # Same location = 0 travel time
        if origin == destination:
            return TravelTimeResult(
                travel_time_seconds=0,
                distance_meters=0,
                route_summary="same_location"
            )
        
        # Check cache
        cache_key = self._get_cache_key(origin, destination, departure_time)
        cached_result = await self._get_from_cache(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for {origin} -> {destination}")
            return cached_result
        
        # Try Azure Maps API
        result = await self._call_azure_maps_api(origin, destination, departure_time)
        
        # Fallback to Haversine calculation
        if result is None:
            logger.info(f"Using fallback for {origin} -> {destination}")
            result = await self._calculate_haversine_fallback(origin, destination)
        
        # Cache result
        await self._save_to_cache(cache_key, result)
        
        return result
    
    async def get_travel_time_matrix(self, locations: List[str], departure_time: datetime) -> Dict[Tuple[str, str], TravelTimeResult]:
        """
        Lấy travel time matrix cho nhiều locations
        
        Args:
            locations: Danh sách tên địa điểm
            departure_time: Thời điểm khởi hành
            
        Returns:
            Dict với key (origin, destination) và value TravelTimeResult
        """
        matrix = {}
        
        # Tạo tất cả combinations
        tasks = []
        pairs = []
        
        for origin in locations:
            for destination in locations:
                if origin != destination:
                    task = self.get_travel_time(origin, destination, departure_time)
                    tasks.append(task)
                    pairs.append((origin, destination))
        
        # Execute parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for (origin, destination), result in zip(pairs, results):
            if isinstance(result, Exception):
                logger.error(f"Error getting travel time {origin}->{destination}: {result}")
                # Use fallback
                result = await self._calculate_haversine_fallback(origin, destination)
            
            matrix[(origin, destination)] = result
        
        return matrix
