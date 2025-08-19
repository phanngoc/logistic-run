"""Configuration settings cho ứng dụng MVP Logistics Run"""

import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    # API Keys
    azure_maps_key: Optional[str] = os.getenv("AZURE_MAPS_KEY")
    port_api_key: Optional[str] = os.getenv("PORT_API_KEY") 
    
    # Azure Maps settings
    azure_maps_base_url: str = "https://atlas.microsoft.com"
    
    # Port API settings
    port_api_base_url: str = "https://api.port-data.com"  # Mock URL
    
    # Cache settings
    redis_url: Optional[str] = os.getenv("REDIS_URL", "redis://localhost:6379")
    cache_ttl_seconds: int = 900  # 15 minutes for travel time cache
    
    # Solver settings
    solver_max_iterations: int = 1000
    solver_time_limit_seconds: int = 30
    
    # Cost defaults
    default_fuel_cost_per_km: float = 0.25
    default_toll_per_km_highway: float = 0.15
    default_late_penalty_per_min: float = 2.0
    
    # Gate dwell defaults (phút)
    default_gate_dwell_min: int = 30
    gate_dwell_by_port: dict = {
        "PORT_A": 30,
        "PORT_B": 45, 
        "PORT_C": 60
    }
    
    # Optimization weights
    default_lambda_late: float = 1.0
    default_lambda_ot: float = 1.0
    default_lambda_tw: float = 10.0
    
    # Time bucketing for cache (minutes)
    time_bucket_minutes: int = 15
    
    # Debug mode
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    class Config:
        env_file = ".env"


settings = Settings()
