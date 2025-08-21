"""
ML Model Builder and Manager
Tập trung vào việc xây dựng, đánh giá và quản lý ML models
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Metrics cho model evaluation"""
    mae: float
    r2: float
    mse: float
    rmse: float
    cross_val_mean: float
    cross_val_std: float

@dataclass
class ModelConfig:
    """Configuration cho ML models"""
    name: str
    target_column: str
    feature_columns: List[str]
    model_params: Dict[str, Any]
    preprocessing: Dict[str, Any]

class ModelBuilder:
    """Class để build và train ML models"""
    
    def __init__(self, db_path: str = "logistics_data.db", models_path: str = "models/"):
        self.db_path = db_path
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'route_score': ModelConfig(
                name='route_score',
                target_column='route_score',
                feature_columns=[
                    'total_distance_km', 'orders_served', 'overtime_hours',
                    'utilization_rate', 'distance_per_order', 'is_weekend',
                    'is_peak_hour', 'shift_start_hour', 'shift_end_hour',
                    'overtime_threshold_min'
                ],
                model_params={
                    'n_estimators': 100,
                    'max_depth': 12,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                },
                preprocessing={'scale_features': False, 'handle_outliers': True}
            ),
            
            'total_cost': ModelConfig(
                name='total_cost',
                target_column='total_cost',
                feature_columns=[
                    'total_distance_km', 'orders_served', 'overtime_hours',
                    'utilization_rate', 'distance_per_order', 'is_weekend',
                    'is_peak_hour', 'shift_start_hour', 'shift_end_hour',
                    'overtime_threshold_min'
                ],
                model_params={
                    'n_estimators': 150,
                    'max_depth': 15,
                    'min_samples_split': 3,
                    'min_samples_leaf': 1,
                    'random_state': 42
                },
                preprocessing={'scale_features': False, 'handle_outliers': True}
            ),
            
            'overtime_hours': ModelConfig(
                name='overtime_hours',
                target_column='overtime_hours',
                feature_columns=[
                    'total_distance_km', 'orders_served', 'shift_start_hour',
                    'shift_end_hour', 'utilization_rate', 'is_weekend',
                    'is_peak_hour', 'overtime_threshold_min'
                ],
                model_params={
                    'n_estimators': 80,
                    'max_depth': 10,
                    'min_samples_split': 4,
                    'min_samples_leaf': 2,
                    'random_state': 42
                },
                preprocessing={'scale_features': False, 'handle_outliers': False}
            )
        }
    
    def load_training_data(self) -> pd.DataFrame:
        """Load training data từ database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load từ historical_routes table
            query = """
            SELECT * FROM historical_routes
            ORDER BY route_date DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(df)} training samples from database")
            
            # Basic data validation
            if df.empty:
                raise ValueError("No training data found in database")
            
            # Check for required columns
            required_cols = set()
            for config in self.model_configs.values():
                required_cols.update(config.feature_columns)
                required_cols.add(config.target_column)
            
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing columns in training data: {missing_cols}")
                # Try to create missing columns with defaults
                for col in missing_cols:
                    if col in ['is_weekend', 'is_peak_hour']:
                        df[col] = 0  # Default to False
                    elif 'hour' in col:
                        df[col] = 8  # Default hour
                    elif col == 'overtime_threshold_min':
                        df[col] = 480  # 8 hours default
                    else:
                        df[col] = 0  # Default numeric value
                        
                logger.info(f"Created missing columns with default values: {missing_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame, config: ModelConfig) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data cho specific model"""
        
        # Extract features and target
        X = df[config.feature_columns].copy()
        y = df[config.target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # Preprocessing based on config
        if config.preprocessing.get('handle_outliers', False):
            # Remove outliers using IQR method
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (y >= lower_bound) & (y <= upper_bound)
            X = X[mask]
            y = y[mask]
            
            logger.info(f"Removed {(~mask).sum()} outliers for {config.name}")
        
        if config.preprocessing.get('scale_features', False):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Save scaler
            scaler_path = self.models_path / f"{config.name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved feature scaler for {config.name}")
        
        logger.info(f"Preprocessed data for {config.name}: {len(X)} samples, {len(X.columns)} features")
        
        return X, y
    
    def train_model(self, config: ModelConfig, X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestRegressor, ModelMetrics]:
        """Train single model"""
        
        logger.info(f"Training {config.name} model...")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train model
        model = RandomForestRegressor(**config.model_params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = ((y_test - y_pred) ** 2).mean()
        rmse = np.sqrt(mse)
        
        # Cross validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mean = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        metrics = ModelMetrics(
            mae=mae,
            r2=r2,
            mse=mse,
            rmse=rmse,
            cross_val_mean=cv_mean,
            cross_val_std=cv_std
        )
        
        logger.info(f"{config.name} - MAE: {mae:.3f}, R²: {r2:.3f}, CV: {cv_mean:.3f}±{cv_std:.3f}")
        
        return model, metrics
    
    def save_model(self, model: RandomForestRegressor, config: ModelConfig, metrics: ModelMetrics):
        """Save trained model"""
        
        model_path = self.models_path / f"{config.name}_model.joblib"
        joblib.dump(model, model_path)
        
        logger.info(f"Saved {config.name} model to {model_path}")
        
        # Save feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': config.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_path = self.models_path / f"{config.name}_feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            
            logger.info(f"Top 3 features for {config.name}: {importance_df.head(3)['feature'].tolist()}")
    
    def build_all_models(self) -> Dict[str, ModelMetrics]:
        """Build tất cả models"""
        
        logger.info("Starting ML model building process...")
        
        # Load data
        df = self.load_training_data()
        
        all_metrics = {}
        
        # Build each model
        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Building {model_name} model")
                logger.info(f"{'='*50}")
                
                # Preprocess data
                X, y = self.preprocess_data(df, config)
                
                # Train model
                model, metrics = self.train_model(config, X, y)
                
                # Save model
                self.save_model(model, config, metrics)
                
                all_metrics[model_name] = metrics
                
            except Exception as e:
                logger.error(f"Error building {model_name} model: {e}")
                continue
        
        # Save consolidated model info
        self.save_model_info(all_metrics)
        
        logger.info(f"\nModel building completed. Built {len(all_metrics)} models.")
        
        return all_metrics
    
    def save_model_info(self, all_metrics: Dict[str, ModelMetrics]):
        """Save consolidated model information"""
        
        model_info = {
            'feature_columns': self.model_configs['route_score'].feature_columns,
            'model_metrics': {
                name: {'mae': metrics.mae, 'r2': metrics.r2}
                for name, metrics in all_metrics.items()
            },
            'training_data_size': len(self.load_training_data()),
            'created_at': datetime.now().isoformat(),
            'model_configs': {
                name: {
                    'target': config.target_column,
                    'features_count': len(config.feature_columns),
                    'model_params': config.model_params
                }
                for name, config in self.model_configs.items()
            }
        }
        
        info_path = self.models_path / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Saved model info to {info_path}")
    
    def evaluate_model_performance(self) -> Dict[str, Any]:
        """Evaluate current model performance"""
        
        performance_report = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'overall_status': 'unknown'
        }
        
        try:
            # Load current models
            model_files = {
                'route_score': 'route_score_model.joblib',
                'total_cost': 'total_cost_model.joblib',
                'overtime_hours': 'overtime_hours_model.joblib'
            }
            
            loaded_models = {}
            for name, filename in model_files.items():
                model_path = self.models_path / filename
                if model_path.exists():
                    loaded_models[name] = joblib.load(model_path)
            
            # Load test data
            df = self.load_training_data()
            test_data = df.sample(min(100, len(df)), random_state=42)
            
            # Evaluate each model
            all_scores = []
            for model_name, model in loaded_models.items():
                config = self.model_configs[model_name]
                X_test = test_data[config.feature_columns]
                y_test = test_data[config.target_column]
                
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                performance_report['models'][model_name] = {
                    'mae': mae,
                    'r2': r2,
                    'status': 'good' if r2 > 0.5 else 'poor' if r2 > 0 else 'very_poor'
                }
                
                all_scores.append(r2)
            
            # Overall status
            avg_r2 = np.mean(all_scores) if all_scores else 0
            if avg_r2 > 0.7:
                performance_report['overall_status'] = 'excellent'
            elif avg_r2 > 0.5:
                performance_report['overall_status'] = 'good'
            elif avg_r2 > 0.2:
                performance_report['overall_status'] = 'acceptable'
            else:
                performance_report['overall_status'] = 'poor'
            
            performance_report['average_r2'] = avg_r2
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            performance_report['error'] = str(e)
        
        return performance_report
    
    def generate_more_training_data(self, num_samples: int = 1000) -> bool:
        """Generate additional synthetic training data"""
        
        try:
            logger.info(f"Generating {num_samples} additional training samples...")
            
            # Generate synthetic data directly
            additional_data = self._generate_synthetic_training_data(num_samples)
            
            # Insert into database
            conn = sqlite3.connect(self.db_path)
            additional_data.to_sql('historical_routes', conn, if_exists='append', index=False)
            conn.close()
            
            logger.info(f"Added {len(additional_data)} new training samples to database")
            return True
            
        except Exception as e:
            logger.error(f"Error generating training data: {e}")
            return False
    
    def _generate_synthetic_training_data(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic training data"""
        
        import random
        from datetime import datetime, timedelta
        
        data = []
        
        for i in range(num_samples):
            # Random route parameters
            total_distance_km = random.uniform(50, 500)
            orders_served = random.randint(3, 15)
            utilization_rate = random.uniform(0.3, 1.0)
            distance_per_order = total_distance_km / orders_served
            
            # Time parameters
            is_weekend = random.choice([0, 1])
            is_peak_hour = random.choice([0, 1])
            shift_start_hour = random.randint(6, 10)
            shift_end_hour = shift_start_hour + random.randint(8, 12)
            overtime_threshold_min = 480  # 8 hours
            
            # Calculate derived metrics
            base_time_hours = total_distance_km / 40  # Average 40 km/h
            service_time_hours = orders_served * 0.5  # 30min per order
            total_time_hours = base_time_hours + service_time_hours
            
            # Overtime calculation
            overtime_hours = max(0, total_time_hours - (overtime_threshold_min / 60))
            
            # Route score calculation (business logic)
            efficiency_score = min(1.0, utilization_rate)
            distance_score = max(0, 1 - (distance_per_order - 20) / 50)  # Penalty for long distances per order
            time_score = max(0, 1 - overtime_hours / 4)  # Penalty for overtime
            
            route_score = (efficiency_score * 0.4 + distance_score * 0.3 + time_score * 0.3)
            route_score = max(0, min(1, route_score + random.uniform(-0.1, 0.1)))  # Add noise
            
            # Cost calculation
            distance_cost = total_distance_km * 1.5  # $1.5 per km
            time_cost = total_time_hours * 25  # $25 per hour
            overtime_cost = overtime_hours * 35  # $35 per overtime hour
            total_cost = distance_cost + time_cost + overtime_cost
            
            # Add some business rule variations
            if is_weekend:
                total_cost *= 1.2  # Weekend premium
                route_score *= 0.9  # Weekend complexity
            
            if is_peak_hour:
                total_cost *= 1.1  # Peak hour cost
                route_score *= 0.95  # Peak hour difficulty
            
            # Create data point
            sample = {
                'route_id': f'ROUTE_{i+1000:05d}',
                'route_date': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                'total_distance_km': round(total_distance_km, 2),
                'orders_served': orders_served,
                'overtime_hours': round(overtime_hours, 2),
                'utilization_rate': round(utilization_rate, 3),
                'distance_per_order': round(distance_per_order, 2),
                'is_weekend': is_weekend,
                'is_peak_hour': is_peak_hour,
                'shift_start_hour': shift_start_hour,
                'shift_end_hour': shift_end_hour,
                'overtime_threshold_min': overtime_threshold_min,
                'route_score': round(route_score, 3),
                'total_cost': round(total_cost, 2),
                'truck_id': f'TRUCK_{random.randint(1, 5):03d}',
                'driver_id': f'DRIVER_{random.randint(1, 10):03d}',
                'total_time_hours': round(total_time_hours, 2),
                'fuel_consumption_liters': round(total_distance_km * 0.25, 2),
                'co2_emissions_kg': round(total_distance_km * 2.3, 2)
            }
            
            data.append(sample)
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic training samples")
        
        return df
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive model status"""
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'models_available': [],
            'models_missing': [],
            'total_models': len(self.model_configs),
            'model_info_available': False,
            'performance': None
        }
        
        # Check model files
        for model_name in self.model_configs.keys():
            model_file = self.models_path / f"{model_name}_model.joblib"
            if model_file.exists():
                status['models_available'].append(model_name)
            else:
                status['models_missing'].append(model_name)
        
        # Check model info
        info_file = self.models_path / "model_info.json"
        status['model_info_available'] = info_file.exists()
        
        # Get performance if models exist
        if status['models_available']:
            try:
                status['performance'] = self.evaluate_model_performance()
            except Exception as e:
                status['performance_error'] = str(e)
        
        return status


def main():
    """Main function để build models"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🤖 ML Model Builder - Logistics Optimization")
    print("=" * 60)
    
    builder = ModelBuilder()
    
    # Check current status
    print("\n📊 Current Model Status:")
    status = builder.get_model_status()
    print(f"Available models: {status['models_available']}")
    print(f"Missing models: {status['models_missing']}")
    
    # Generate more data if needed
    print("\n📈 Generating additional training data...")
    builder.generate_more_training_data(500)
    
    # Build all models
    print("\n🔨 Building ML Models...")
    metrics = builder.build_all_models()
    
    # Performance summary
    print("\n📋 Model Performance Summary:")
    for model_name, model_metrics in metrics.items():
        print(f"{model_name}:")
        print(f"  MAE: {model_metrics.mae:.3f}")
        print(f"  R²: {model_metrics.r2:.3f}")
        print(f"  CV: {model_metrics.cross_val_mean:.3f}±{model_metrics.cross_val_std:.3f}")
    
    # Final evaluation
    print("\n🎯 Final Model Evaluation:")
    performance = builder.evaluate_model_performance()
    print(f"Overall Status: {performance['overall_status']}")
    print(f"Average R²: {performance.get('average_r2', 0):.3f}")
    
    print("\n✅ Model building completed!")


if __name__ == "__main__":
    main()
