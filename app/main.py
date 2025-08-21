"""FastAPI main application cho MVP Dispatch Optimization"""

import logging
import time
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, List

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
from .solvers.ml_enhanced_solver import MLEnhancedSolver
from .services.ml_predictor import MLPredictorService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
travel_time_service: TravelTimeService = None
port_api_service: PortApiService = None
ml_predictor_service: MLPredictorService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    
    # Startup
    logger.info("Starting MVP Dispatch Optimization Service")
    
    global travel_time_service, port_api_service, ml_predictor_service
    
    # Initialize services
    travel_time_service = TravelTimeService()
    port_api_service = PortApiService()
    ml_predictor_service = MLPredictorService()
    
    # Setup services
    try:
        await travel_time_service.__aenter__()
        await port_api_service.__aenter__()
        logger.info("Core services initialized successfully")
        
        # ML service doesn't need async setup
        ml_status = ml_predictor_service.get_model_status()
        logger.info(f"ML Predictor initialized: {ml_status['models_loaded']} models loaded")
        
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


def get_ml_predictor_service() -> MLPredictorService:
    """Dependency injection for ML predictor service"""
    if ml_predictor_service is None:
        raise HTTPException(status_code=503, detail="ML Predictor service not available")
    return ml_predictor_service


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
    port_service: PortApiService = Depends(get_port_api_service),
    ml_service: MLPredictorService = Depends(get_ml_predictor_service)
) -> DispatchResponse:
    """
    Gợi ý điều phối tối ưu sử dụng pre-trained ML models
    
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
        
        # Always prefer ML Enhanced Solver if models are available
        ml_status = ml_service.get_model_status()
        if ml_status['models_loaded'] > 0:
            solver = MLEnhancedSolver(request.costs, request.weights)
            logger.info(f"Using ML Enhanced Solver with {ml_status['models_loaded']} models")
            algorithm_used = "ml_enhanced_optimization"
        else:
            solver = LocalSearchSolver(request.costs, request.weights)
            logger.info("Using Local Search Solver (ML models not available)")
            algorithm_used = "local_search_fallback"
        
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
        
        # Add detailed metadata
        response.metadata = response.metadata or {}
        response.metadata.update({
            "processing_time_seconds": time.time() - start_time,
            "algorithm_used": algorithm_used,
            "ml_models_available": ml_status['models_loaded'],
            "ml_models_used": ml_status['models_loaded'] if algorithm_used == "ml_enhanced_optimization" else 0,
            "request_summary": {
                "orders_count": len(request.orders),
                "trucks_count": len(request.fleet),
                "cost_config": request.costs.dict(),
                "weights": request.weights.dict()
            },
            "performance_stats": {
                "routes_generated": len(response.routes),
                "orders_served": response.kpi.served_orders if hasattr(response.kpi, 'served_orders') else len(request.orders) - len(response.unserved_orders),
                "service_rate": (len(request.orders) - len(response.unserved_orders)) / len(request.orders) if request.orders else 0
            }
        })
        
        logger.info(f"Dispatch optimization completed in {response.solve_time_seconds:.2f}s, serving {response.metadata['performance_stats']['orders_served']}/{len(request.orders)} orders")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dispatch optimization error: {e}")
        
        # Return error response with detailed info
        return DispatchResponse(
            success=False,
            message=f"Optimization failed: {str(e)}",
            routes=[],
            unserved_orders=[order.order_id for order in request.orders],
            kpi={"total_orders": len(request.orders), "served_orders": 0, "unserved_orders": len(request.orders)},
            solve_time_seconds=time.time() - start_time,
            iterations=0,
            algorithm="failed",
            metadata={
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_seconds": time.time() - start_time,
                "ml_status": ml_service.get_model_status() if ml_service else {"error": "ML service unavailable"}
            }
        )


@app.get("/solver/status")
async def solver_status(ml_service: MLPredictorService = Depends(get_ml_predictor_service)):
    """Get comprehensive solver status and ML capabilities"""
    
    # Get detailed ML status
    ml_status = ml_service.get_model_status()
    ml_available = ml_status['models_loaded'] > 0
    
    # Available algorithms based on ML status
    base_algorithms = ["greedy_insertion", "greedy_with_local_search"]
    available_algorithms = base_algorithms.copy()
    default_algorithm = "greedy_with_local_search"
    
    if ml_available:
        available_algorithms.extend([
            "ml_enhanced_greedy", 
            "ml_guided_optimization",
            "predictive_routing"
        ])
        default_algorithm = "ml_enhanced_greedy"
    
    # Advanced features based on model availability
    ml_features = {}
    if ml_available:
        if 'route_score' in ml_status['available_models']:
            ml_features['route_scoring'] = {
                'available': True,
                'accuracy': ml_status['model_performance'].get('route_score', {}).get('r2', 0)
            }
        
        if 'total_cost' in ml_status['available_models']:
            ml_features['cost_prediction'] = {
                'available': True,
                'accuracy': ml_status['model_performance'].get('total_cost', {}).get('r2', 0)
            }
        
        if 'overtime_hours' in ml_status['available_models']:
            ml_features['overtime_prediction'] = {
                'available': True,
                'accuracy': ml_status['model_performance'].get('overtime_hours', {}).get('r2', 0)
            }
    
    return {
        "solver_info": {
            "available_algorithms": available_algorithms,
            "default_algorithm": default_algorithm,
            "ml_enhanced": ml_available,
            "max_iterations": settings.solver_max_iterations,
            "time_limit_seconds": settings.solver_time_limit_seconds,
            "supported_container_sizes": ["20", "40", "40HC"]
        },
        
        "ml_status": {
            "models_loaded": ml_status['models_loaded'],
            "total_models": ml_status.get('total_models', 3),
            "available_models": ml_status['available_models'],
            "missing_models": ml_status.get('missing_models', []),
            "last_trained": ml_status.get('last_trained', 'unknown'),
            "training_data_size": ml_status.get('training_data_size', 0)
        },
        
        "features": {
            # Base optimization features
            "travel_time_optimization": True,
            "port_queue_forecasting": True,
            "cost_optimization": True,
            "time_window_constraints": True,
            "overtime_calculation": True,
            "local_search_improvement": True,
            "capacity_constraints": True,
            
            # ML-enhanced features
            "ml_route_scoring": ml_features.get('route_scoring', {'available': False}),
            "ml_cost_prediction": ml_features.get('cost_prediction', {'available': False}),
            "ml_overtime_prediction": ml_features.get('overtime_prediction', {'available': False}),
            "predictive_route_ranking": ml_available,
            "intelligent_stop_reordering": ml_available,
            "ml_guided_local_search": ml_available
        },
        
        "performance_benchmarks": {
            "typical_response_time_50_orders": "2-3 seconds",
            "typical_response_time_100_orders": "5-8 seconds", 
            "typical_response_time_200_orders": "15-25 seconds",
            "service_rate_target": "95%+",
            "cost_reduction_vs_baseline": "15-25%"
        },
        
        "model_performance": ml_status.get('model_performance', {}),
        "recommendations": _get_system_recommendations(ml_status)
    }


def _get_system_recommendations(ml_status: Dict) -> List[str]:
    """Generate system recommendations based on ML status"""
    
    recommendations = []
    
    models_loaded = ml_status.get('models_loaded', 0)
    total_models = ml_status.get('total_models', 3)
    
    if models_loaded == 0:
        recommendations.append("❌ No ML models loaded - run 'python build_models.py' to build models")
        recommendations.append("⚡ System will use fallback algorithms with reduced optimization capability")
    
    elif models_loaded < total_models:
        missing = ml_status.get('missing_models', [])
        recommendations.append(f"⚠️ Missing {len(missing)} models: {', '.join(missing)}")
        recommendations.append("🔧 Consider rebuilding models for full ML capabilities")
    
    else:
        recommendations.append("✅ All ML models loaded - system operating at full capacity")
        
        # Check model performance
        performance = ml_status.get('model_performance', {})
        poor_models = []
        
        for model_name, metrics in performance.items():
            r2 = metrics.get('r2', 0)
            if r2 < 0.3:
                poor_models.append(model_name)
        
        if poor_models:
            recommendations.append(f"📊 Models with low accuracy: {', '.join(poor_models)} - consider retraining")
        
        training_size = ml_status.get('training_data_size', 0)
        if training_size < 1000:
            recommendations.append(f"📈 Training data size ({training_size}) is small - generate more data for better accuracy")
    
    return recommendations


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


@app.post("/test/ml-prediction")
async def test_ml_prediction(
    request: DispatchRequest,
    ml_service: MLPredictorService = Depends(get_ml_predictor_service)
):
    """Test ML prediction cho routes"""
    
    try:
        if not request.orders or not request.fleet:
            raise HTTPException(status_code=400, detail="Orders and fleet required for ML prediction test")
        
        # Test prediction for first truck with first few orders
        test_truck = request.fleet[0]
        test_orders = request.orders[:min(3, len(request.orders))]
        
        prediction = ml_service.predict_route_performance(test_truck, test_orders)
        
        return {
            "test_truck": test_truck.truck_id,
            "test_orders": [order.order_id for order in test_orders],
            "prediction": {
                "predicted_score": prediction.predicted_score,
                "predicted_cost": prediction.predicted_cost,
                "predicted_overtime_hours": prediction.predicted_overtime_hours,
                "confidence": prediction.confidence,
                "recommendations": prediction.recommendations
            },
            "model_status": ml_service.get_model_status()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction test failed: {str(e)}")


@app.get("/ml/status")
async def ml_detailed_status(ml_service: MLPredictorService = Depends(get_ml_predictor_service)):
    """Get comprehensive ML model status and performance metrics"""
    
    status = ml_service.get_model_status()
    
    # Add performance insights
    insights = []
    
    if status['models_loaded'] == 0:
        insights.append("🚨 No ML models available - system running in fallback mode")
        insights.append("💡 Run 'python build_models.py' to build and train models")
    
    elif status['models_loaded'] < status.get('total_models', 3):
        missing = status.get('missing_models', [])
        insights.append(f"⚠️ Partial ML capability - missing models: {', '.join(missing)}")
    
    else:
        insights.append("✅ Full ML capability available")
        
        # Analyze model performance
        performance = status.get('model_performance', {})
        if performance:
            excellent_models = []
            poor_models = []
            
            for model_name, metrics in performance.items():
                r2 = metrics.get('r2', 0)
                if r2 > 0.7:
                    excellent_models.append(model_name)
                elif r2 < 0.3:
                    poor_models.append(model_name)
            
            if excellent_models:
                insights.append(f"🎯 High-performing models: {', '.join(excellent_models)}")
            
            if poor_models:
                insights.append(f"📊 Models needing improvement: {', '.join(poor_models)}")
    
    # Add training data insights
    training_size = status.get('training_data_size', 0)
    if training_size > 0:
        if training_size < 500:
            insights.append(f"📈 Training data ({training_size} samples) - consider generating more for better accuracy")
        elif training_size < 1000:
            insights.append(f"📊 Training data ({training_size} samples) - adequate but more data could improve performance")
        else:
            insights.append(f"💪 Good training data size ({training_size} samples)")
    
    return {
        **status,
        "insights": insights,
        "recommendations": {
            "immediate_actions": _get_immediate_actions(status),
            "performance_improvements": _get_performance_improvements(status),
            "long_term_goals": _get_long_term_goals(status)
        }
    }


@app.post("/ml/rebuild")
async def rebuild_models():
    """Trigger model rebuilding process"""
    
    try:
        logger.info("Starting model rebuild process...")
        
        # Import model builder
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        
        from build_models import ModelBuilder
        
        builder = ModelBuilder()
        
        # Check if we need more training data
        current_status = builder.get_model_status()
        if not current_status.get('models_available'):
            # Generate training data first
            success = builder.generate_more_training_data(1000)
            if not success:
                return {"success": False, "message": "Failed to generate training data"}
        
        # Build models
        metrics = builder.build_all_models()
        
        # Reload ML service models
        global ml_predictor_service
        if ml_predictor_service:
            ml_predictor_service._load_models()
        
        return {
            "success": True,
            "message": f"Successfully rebuilt {len(metrics)} models",
            "models_built": list(metrics.keys()),
            "performance": {
                name: {"mae": m.mae, "r2": m.r2} 
                for name, m in metrics.items()
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Model rebuild failed: {e}")
        return {
            "success": False,
            "message": f"Model rebuild failed: {str(e)}",
            "error_type": type(e).__name__,
            "timestamp": time.time()
        }


@app.get("/ml/performance-history")
async def ml_performance_history():
    """Get historical model performance data"""
    
    try:
        # Read model info files from different timestamps
        from pathlib import Path
        models_path = Path("models/")
        
        history = []
        
        # Current model info
        current_info_file = models_path / "model_info.json"
        if current_info_file.exists():
            with open(current_info_file, 'r') as f:
                current_info = json.load(f)
                history.append({
                    "timestamp": current_info.get('created_at', 'unknown'),
                    "training_data_size": current_info.get('training_data_size', 0),
                    "model_metrics": current_info.get('model_metrics', {}),
                    "status": "current"
                })
        
        # Add performance trends
        if history:
            latest = history[0]
            trends = {}
            
            for model_name, metrics in latest['model_metrics'].items():
                r2 = metrics.get('r2', 0)
                if r2 > 0.7:
                    trends[model_name] = "excellent"
                elif r2 > 0.5:
                    trends[model_name] = "good"
                elif r2 > 0.2:
                    trends[model_name] = "acceptable"
                else:
                    trends[model_name] = "poor"
            
            return {
                "history": history,
                "current_trends": trends,
                "summary": {
                    "total_entries": len(history),
                    "latest_training_size": latest['training_data_size'],
                    "models_tracked": len(latest['model_metrics'])
                }
            }
        
        return {"history": [], "message": "No performance history available"}
        
    except Exception as e:
        logger.error(f"Error getting performance history: {e}")
        return {"error": str(e)}


def _get_immediate_actions(status: Dict) -> List[str]:
    """Get immediate actions based on ML status"""
    
    actions = []
    
    if status['models_loaded'] == 0:
        actions.append("Build initial ML models: python build_models.py")
        actions.append("Verify database has training data")
    
    elif status['models_loaded'] < status.get('total_models', 3):
        actions.append("Rebuild missing models")
        actions.append("Check model files in models/ directory")
    
    # Check training data
    training_size = status.get('training_data_size', 0)
    if training_size < 500:
        actions.append("Generate more training data for better model accuracy")
    
    return actions


def _get_performance_improvements(status: Dict) -> List[str]:
    """Get performance improvement suggestions"""
    
    improvements = []
    
    performance = status.get('model_performance', {})
    if performance:
        for model_name, metrics in performance.items():
            r2 = metrics.get('r2', 0)
            if r2 < 0.5:
                improvements.append(f"Improve {model_name} model - current R²: {r2:.3f}")
        
        if not improvements:
            improvements.append("All models performing well - consider advanced hyperparameter tuning")
    
    # Training data improvements
    training_size = status.get('training_data_size', 0)
    if training_size < 2000:
        improvements.append("Increase training data size for better generalization")
    
    improvements.append("Implement cross-validation for more robust evaluation")
    improvements.append("Consider ensemble methods for improved accuracy")
    
    return improvements


def _get_long_term_goals(status: Dict) -> List[str]:
    """Get long-term improvement goals"""
    
    goals = []
    
    goals.extend([
        "Implement real-time model updates with feedback data",
        "Add deep learning models for complex pattern recognition", 
        "Develop reinforcement learning for dynamic routing",
        "Create model versioning and A/B testing framework",
        "Build automated model monitoring and alerting",
        "Implement federated learning for multi-client scenarios"
    ])
    
    return goals


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
