"""FastAPI main application cho MVP Dispatch Optimization"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .schemas.dispatch import DispatchRequest, DispatchResponse
from .services.travel_time import TravelTimeService
from .services.port_api import PortApiService
from .services.cost_engine import CostEngine
from .solvers.local_search import LocalSearchSolver

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
travel_time_service: TravelTimeService = None
port_api_service: PortApiService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    
    # Startup
    logger.info("Starting MVP Dispatch Optimization Service")
    
    global travel_time_service, port_api_service
    
    # Initialize services
    travel_time_service = TravelTimeService()
    port_api_service = PortApiService()
    
    # Setup services
    try:
        await travel_time_service.__aenter__()
        await port_api_service.__aenter__()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down services")
    try:
        if travel_time_service:
            await travel_time_service.__aexit__(None, None, None)
        if port_api_service:
            await port_api_service.__aexit__(None, None, None)
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="MVP Dispatch Optimization",
    description="Gợi ý điều phối tối ưu cho logistics - MVP version",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_travel_time_service() -> TravelTimeService:
    """Dependency injection for travel time service"""
    if travel_time_service is None:
        raise HTTPException(status_code=503, detail="Travel time service not available")
    return travel_time_service


def get_port_api_service() -> PortApiService:
    """Dependency injection for port API service"""
    if port_api_service is None:
        raise HTTPException(status_code=503, detail="Port API service not available")
    return port_api_service


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MVP Dispatch Optimization",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {}
    }
    
    # Check travel time service
    try:
        if travel_time_service and travel_time_service.redis_client:
            await travel_time_service.redis_client.ping()
            health_status["services"]["travel_time"] = "healthy"
        else:
            health_status["services"]["travel_time"] = "no_cache"
    except Exception as e:
        health_status["services"]["travel_time"] = f"unhealthy: {str(e)}"
    
    # Check port API service
    try:
        if port_api_service:
            health_status["services"]["port_api"] = "healthy"
        else:
            health_status["services"]["port_api"] = "unavailable"
    except Exception as e:
        health_status["services"]["port_api"] = f"unhealthy: {str(e)}"
    
    return health_status


@app.post("/dispatch/suggest", response_model=DispatchResponse)
async def suggest_dispatch(
    request: DispatchRequest,
    travel_service: TravelTimeService = Depends(get_travel_time_service),
    port_service: PortApiService = Depends(get_port_api_service)
) -> DispatchResponse:
    """
    Gợi ý điều phối tối ưu
    
    Nhận vào orders, fleet, cost config và trả về routes tối ưu
    """
    
    start_time = time.time()
    
    try:
        logger.info(f"Received dispatch request with {len(request.orders)} orders and {len(request.fleet)} trucks")
        
        # Validate basic constraints
        if not request.orders:
            raise HTTPException(status_code=400, detail="No orders provided")
        
        if not request.fleet:
            raise HTTPException(status_code=400, detail="No trucks provided")
        
        # Initialize solver
        solver = LocalSearchSolver(request.costs, request.weights)
        solver.set_services(travel_service, port_service)
        
        # Solve optimization problem
        solver_result = await solver.solve(
            orders=request.orders,
            trucks=request.fleet,
            max_iterations=request.max_iterations or settings.solver_max_iterations,
            time_limit_seconds=request.time_limit_seconds or settings.solver_time_limit_seconds
        )
        
        # Create response
        response = await solver.create_dispatch_response(solver_result, len(request.orders))
        
        # Add processing time to metadata
        response.metadata = response.metadata or {}
        response.metadata["processing_time_seconds"] = time.time() - start_time
        response.metadata["request_summary"] = {
            "orders_count": len(request.orders),
            "trucks_count": len(request.fleet),
            "cost_config": request.costs.dict(),
            "weights": request.weights.dict()
        }
        
        logger.info(f"Dispatch optimization completed in {response.solve_time_seconds:.2f}s, serving {response.kpi.served_orders}/{response.kpi.total_orders} orders")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dispatch optimization error: {e}")
        
        # Return error response
        return DispatchResponse(
            success=False,
            message=f"Optimization failed: {str(e)}",
            routes=[],
            unserved_orders=[order.order_id for order in request.orders],
            kpi={"total_orders": len(request.orders), "served_orders": 0, "unserved_orders": len(request.orders)},
            solve_time_seconds=time.time() - start_time,
            iterations=0,
            algorithm="failed",
            metadata={"error": str(e)}
        )


@app.get("/solver/status")
async def solver_status():
    """Get solver status and capabilities"""
    
    return {
        "available_algorithms": ["greedy_insertion", "greedy_with_local_search"],
        "default_algorithm": "greedy_with_local_search",
        "max_iterations": settings.solver_max_iterations,
        "time_limit_seconds": settings.solver_time_limit_seconds,
        "supported_container_sizes": ["20", "40"],
        "features": {
            "travel_time_optimization": True,
            "port_queue_forecasting": True,
            "cost_optimization": True,
            "time_window_constraints": True,
            "overtime_calculation": True,
            "local_search_improvement": True
        }
    }


@app.post("/test/travel-time")
async def test_travel_time(
    origin: str,
    destination: str,
    travel_service: TravelTimeService = Depends(get_travel_time_service)
):
    """Test travel time calculation"""
    
    try:
        from datetime import datetime
        result = await travel_service.get_travel_time(origin, destination, datetime.now())
        
        return {
            "origin": origin,
            "destination": destination,
            "travel_time_minutes": result.travel_time_minutes,
            "distance_km": result.distance_km,
            "highway_distance_km": result.highway_distance_km,
            "route_summary": result.route_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Travel time calculation failed: {str(e)}")


@app.post("/test/port-dwell")
async def test_port_dwell(
    location: str,
    port_service: PortApiService = Depends(get_port_api_service)
):
    """Test port dwell time prediction"""
    
    try:
        from datetime import datetime
        dwell_info = await port_service.get_gate_dwell_info(location, datetime.now())
        
        return {
            "location": location,
            "port_name": dwell_info.port_name,
            "gate_id": dwell_info.gate_id,
            "expected_dwell_minutes": dwell_info.expected_dwell_minutes,
            "confidence": dwell_info.confidence,
            "source": dwell_info.source,
            "queue_length": dwell_info.queue_length
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Port dwell calculation failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
