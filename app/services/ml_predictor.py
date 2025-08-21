"""
Machine Learning Predictor Service
Sử dụng models đã train để predict route performance
"""

import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import joblib
from dataclasses import dataclass

from ..schemas.orders import Order
from ..schemas.fleet import Truck
from ..schemas.dispatch import Route

logger = logging.getLogger(__name__)

@dataclass
class RoutePrediction:
    """Kết quả prediction cho route"""
    predicted_score: float
    predicted_cost: float
    predicted_overtime_hours: float
    confidence: float
    recommendations: List[str]

@dataclass
class ModelInfo:
    """Thông tin về model"""
    feature_columns: List[str]
    metrics: Dict[str, Dict[str, float]]
    training_data_size: int
    created_at: str

class MLPredictorService:
    """Service để predict route performance sử dụng ML models"""
    
    def __init__(self, db_path: str = "logistics_data.db", models_path: str = "models/"):
        self.db_path = db_path
        self.models_path = Path(models_path)
        self.models = {}
        self.model_info = None
        self._load_models()
    
    def _load_models(self):
        """Load các trained models"""
        
        try:
            # Load model info
            model_info_file = self.models_path / "model_info.json"
            if model_info_file.exists():
                with open(model_info_file, 'r') as f:
                    info_data = json.load(f)
                    self.model_info = ModelInfo(
                        feature_columns=info_data['feature_columns'],
                        metrics=info_data['model_metrics'],
                        training_data_size=info_data['training_data_size'],
                        created_at=info_data['created_at']
                    )
                logger.info(f"Loaded model info: {self.model_info.training_data_size} training samples")
            
            # Load individual models
            model_files = {
                'route_score': 'route_score_model.joblib',
                'total_cost': 'total_cost_model.joblib', 
                'overtime_hours': 'overtime_hours_model.joblib'
            }
            
            for model_name, filename in model_files.items():
                model_file = self.models_path / filename
                if model_file.exists():
                    self.models[model_name] = joblib.load(model_file)
                    logger.info(f"Loaded {model_name} model from {model_file}")
                else:
                    logger.warning(f"Model file not found: {model_file}")
            
            if not self.models:
                logger.warning("No ML models loaded - predictions will use fallback methods")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models = {}
            self.model_info = None
    
    def get_travel_time_from_db(self, origin: str, destination: str) -> Tuple[float, float]:
        """Lấy travel time từ database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT distance_km, travel_time_minutes, traffic_factor
                FROM travel_times 
                WHERE origin_location = ? AND destination_location = ?
            """, (origin, destination))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                distance_km, travel_time_min, traffic_factor = result
                return distance_km, travel_time_min * traffic_factor
            else:
                # Fallback: estimate based on typical values
                return 30.0, 45.0  # 30km, 45 minutes
                
        except Exception as e:
            logger.error(f"Error getting travel time from DB: {e}")
            return 30.0, 45.0
    
    def get_port_dwell_from_db(self, port_location: str, arrival_time: datetime) -> int:
        """Lấy predicted port dwell time từ database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            hour_of_day = arrival_time.hour
            day_of_week = arrival_time.weekday()
            
            cursor.execute("""
                SELECT AVG(expected_dwell_minutes)
                FROM port_dwell_times 
                WHERE port_location = ? AND hour_of_day = ? AND day_of_week = ?
            """, (port_location, hour_of_day, day_of_week))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return int(result[0])
            else:
                return 30  # Default dwell time
                
        except Exception as e:
            logger.error(f"Error getting port dwell from DB: {e}")
            return 30
    
    def extract_route_features(self, truck: Truck, orders: List[Order]) -> Dict[str, float]:
        """Extract features từ route để feed vào models"""
        
        if not orders:
            return {}
        
        # Calculate basic route metrics
        total_distance = 0.0
        total_duration = 0.0
        current_location = truck.start_location
        current_time = truck.shift_start
        
        for order in orders:
            # Distance and time to pickup
            pickup_dist, pickup_time = self.get_travel_time_from_db(current_location, order.pickup)
            total_distance += pickup_dist
            
            # Arrival at pickup
            pickup_arrival = current_time + timedelta(minutes=pickup_time)
            pickup_arrival = max(pickup_arrival, order.tw_start)  # Wait for time window
            
            # Service time at pickup + port dwell if applicable
            service_time = order.service_time_min or 20
            if "PORT" in order.pickup:
                service_time += self.get_port_dwell_from_db(order.pickup, pickup_arrival)
            
            pickup_departure = pickup_arrival + timedelta(minutes=service_time)
            
            # Distance and time to dropoff
            dropoff_dist, dropoff_time = self.get_travel_time_from_db(order.pickup, order.dropoff)
            total_distance += dropoff_dist
            
            dropoff_arrival = pickup_departure + timedelta(minutes=dropoff_time)
            dropoff_departure = dropoff_arrival + timedelta(minutes=service_time)
            
            # Update current position and time
            current_location = order.dropoff
            current_time = dropoff_departure
        
        # Calculate total duration
        total_duration = (current_time - truck.shift_start).total_seconds() / 3600  # hours
        
        # Calculate derived features
        shift_duration = truck.shift_end.hour - truck.shift_start.hour
        overtime_hours = max(0, total_duration - shift_duration)
        utilization_rate = len(orders) / truck.max_orders_per_day
        distance_per_order = total_distance / len(orders) if orders else 0
        
        # Time-based features
        is_weekend = truck.shift_start.weekday() >= 5  # Saturday=5, Sunday=6
        is_peak_hour = truck.shift_start.hour in range(7, 10)
        
        features = {
            'total_distance_km': total_distance,
            'orders_served': len(orders),
            'overtime_hours': overtime_hours,
            'utilization_rate': utilization_rate,
            'distance_per_order': distance_per_order,
            'is_weekend': 1 if is_weekend else 0,
            'is_peak_hour': 1 if is_peak_hour else 0,
            'shift_start_hour': truck.shift_start.hour,
            'shift_end_hour': truck.shift_end.hour,
            'overtime_threshold_min': truck.overtime_threshold_min
        }
        
        return features
    
    def predict_route_performance(self, truck: Truck, orders: List[Order]) -> RoutePrediction:
        """Predict performance cho một route"""
        
        if not orders:
            return RoutePrediction(
                predicted_score=0.0,
                predicted_cost=0.0,
                predicted_overtime_hours=0.0,
                confidence=0.0,
                recommendations=["No orders assigned to this route"]
            )
        
        # Extract features
        features = self.extract_route_features(truck, orders)
        
        if not features or not self.model_info:
            # Fallback prediction without ML
            return self._fallback_prediction(features, orders)
        
        try:
            # Prepare feature vector
            feature_vector = []
            for col in self.model_info.feature_columns:
                feature_vector.append(features.get(col, 0.0))
            
            X = np.array([feature_vector])
            
            # Make predictions
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                pred = model.predict(X)[0]
                predictions[model_name] = pred
                
                # Calculate confidence based on model metrics
                if model_name in self.model_info.metrics:
                    r2 = self.model_info.metrics[model_name].get('r2', 0.5)
                    confidences[model_name] = max(0.1, min(0.95, r2))
                else:
                    confidences[model_name] = 0.5
            
            # Generate recommendations
            recommendations = self._generate_recommendations(features, predictions)
            
            # Overall confidence
            avg_confidence = np.mean(list(confidences.values())) if confidences else 0.5
            
            return RoutePrediction(
                predicted_score=predictions.get('route_score', 0.5),
                predicted_cost=predictions.get('total_cost', 1000.0),
                predicted_overtime_hours=predictions.get('overtime_hours', 0.0),
                confidence=avg_confidence,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return self._fallback_prediction(features, orders)
    
    def _fallback_prediction(self, features: Dict[str, float], orders: List[Order]) -> RoutePrediction:
        """Fallback prediction khi không có ML models"""
        
        if not features:
            features = {'total_distance_km': 50.0, 'orders_served': len(orders)}
        
        # Simple heuristic predictions
        distance = features.get('total_distance_km', 50.0)
        num_orders = features.get('orders_served', len(orders))
        overtime = features.get('overtime_hours', 0.0)
        
        # Predict cost based on distance
        predicted_cost = distance * 2.5 + num_orders * 50 + overtime * 1500
        
        # Predict score based on efficiency metrics
        predicted_score = 0.8
        if overtime > 0:
            predicted_score -= 0.2
        if distance / num_orders > 50:  # Inefficient routing
            predicted_score -= 0.1
        
        predicted_score = max(0.1, min(0.95, predicted_score))
        
        recommendations = [
            "Using heuristic prediction (ML models not available)",
            f"Route covers {distance:.1f}km with {num_orders} orders"
        ]
        
        if overtime > 0:
            recommendations.append(f"Expected {overtime:.1f}h overtime")
        
        return RoutePrediction(
            predicted_score=predicted_score,
            predicted_cost=predicted_cost,
            predicted_overtime_hours=overtime,
            confidence=0.3,  # Lower confidence for heuristic
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, features: Dict[str, float], predictions: Dict[str, float]) -> List[str]:
        """Generate recommendations based on predictions"""
        
        recommendations = []
        
        # Score-based recommendations
        score = predictions.get('route_score', 0.5)
        if score > 0.8:
            recommendations.append("Excellent route - well optimized")
        elif score > 0.6:
            recommendations.append("Good route with minor improvements possible")
        else:
            recommendations.append("Route needs optimization")
        
        # Overtime recommendations
        overtime = predictions.get('overtime_hours', 0.0)
        if overtime > 2:
            recommendations.append("High overtime expected - consider redistributing orders")
        elif overtime > 0:
            recommendations.append("Some overtime expected")
        
        # Distance efficiency
        distance_per_order = features.get('distance_per_order', 0)
        if distance_per_order > 40:
            recommendations.append("High distance per order - route may be inefficient")
        
        # Utilization
        utilization = features.get('utilization_rate', 0)
        if utilization < 0.5:
            recommendations.append("Low truck utilization - consider adding more orders")
        elif utilization > 0.9:
            recommendations.append("High truck utilization - monitor for overload")
        
        # Peak hour warning
        if features.get('is_peak_hour', 0):
            recommendations.append("Peak hour start - expect higher travel times")
        
        return recommendations
    
    def rank_routes(self, route_predictions: List[Tuple[Route, RoutePrediction]]) -> List[Tuple[Route, RoutePrediction]]:
        """Rank routes theo predicted performance"""
        
        def route_ranking_score(route_pred_tuple):
            route, prediction = route_pred_tuple
            
            # Combine score và cost với confidence weighting
            score_component = prediction.predicted_score * 0.4
            cost_component = (1.0 / (1.0 + prediction.predicted_cost / 1000)) * 0.3  # Normalize cost
            overtime_component = max(0, 1.0 - prediction.predicted_overtime_hours / 4.0) * 0.2  # Penalize overtime
            confidence_component = prediction.confidence * 0.1
            
            return score_component + cost_component + overtime_component + confidence_component
        
        return sorted(route_predictions, key=route_ranking_score, reverse=True)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status của ML models"""
        
        status = {
            "models_loaded": len(self.models),
            "available_models": list(self.models.keys()),
            "model_info": None,
            "database_connected": False
        }
        
        if self.model_info:
            status["model_info"] = {
                "training_data_size": self.model_info.training_data_size,
                "created_at": self.model_info.created_at,
                "metrics": self.model_info.metrics
            }
        
        # Test database connection
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM locations")
            conn.close()
            status["database_connected"] = True
        except Exception:
            status["database_connected"] = False
        
        return status
