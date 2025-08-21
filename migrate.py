"""
Migration script cho logistics optimization system
Tạo SQLite database, seed sample data và build model
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = "logistics_data.db"
MODEL_PATH = "models/"

class LogisticsMigration:
    """Class chính để handle migration và model building"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.model_path = Path(MODEL_PATH)
        self.model_path.mkdir(exist_ok=True)
        
    def create_database_schema(self):
        """Tạo schema cho SQLite database"""
        
        logger.info("Creating database schema...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bảng locations - lưu thông tin các địa điểm
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_code TEXT UNIQUE NOT NULL,
            location_name TEXT NOT NULL,
            location_type TEXT NOT NULL, -- depot, port, warehouse
            latitude REAL,
            longitude REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Bảng trucks - thông tin xe tải
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trucks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            truck_id TEXT UNIQUE NOT NULL,
            start_location TEXT NOT NULL,
            shift_start_hour INTEGER NOT NULL,
            shift_end_hour INTEGER NOT NULL,
            overtime_threshold_min INTEGER DEFAULT 600,
            overtime_rate_per_hour REAL DEFAULT 1500,
            allowed_sizes TEXT NOT NULL, -- JSON array ["20", "40"]
            max_orders_per_day INTEGER DEFAULT 5,
            driver_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (start_location) REFERENCES locations(location_code)
        )
        """)
        
        # Bảng orders - đơn hàng
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT UNIQUE NOT NULL,
            pickup_location TEXT NOT NULL,
            dropoff_location TEXT NOT NULL,
            container_size TEXT NOT NULL,
            tw_start TIMESTAMP NOT NULL,
            tw_end TIMESTAMP NOT NULL,
            service_time_min INTEGER DEFAULT 20,
            priority INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pickup_location) REFERENCES locations(location_code),
            FOREIGN KEY (dropoff_location) REFERENCES locations(location_code)
        )
        """)
        
        # Bảng travel_times - ma trận thời gian di chuyển
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS travel_times (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin_location TEXT NOT NULL,
            destination_location TEXT NOT NULL,
            distance_km REAL NOT NULL,
            travel_time_minutes REAL NOT NULL,
            highway_distance_km REAL DEFAULT 0,
            traffic_factor REAL DEFAULT 1.0, -- multiplier based on time of day
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (origin_location) REFERENCES locations(location_code),
            FOREIGN KEY (destination_location) REFERENCES locations(location_code),
            UNIQUE(origin_location, destination_location)
        )
        """)
        
        # Bảng port_dwell_times - thời gian chờ tại cảng
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS port_dwell_times (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            port_location TEXT NOT NULL,
            gate_id TEXT,
            hour_of_day INTEGER NOT NULL,
            day_of_week INTEGER NOT NULL, -- 0=Monday, 6=Sunday
            expected_dwell_minutes INTEGER NOT NULL,
            queue_length INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0.8,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (port_location) REFERENCES locations(location_code)
        )
        """)
        
        # Bảng historical_routes - lịch sử routes đã thực hiện
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_routes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            route_id TEXT UNIQUE NOT NULL,
            truck_id TEXT NOT NULL,
            route_date DATE NOT NULL,
            total_distance_km REAL NOT NULL,
            total_duration_hours REAL NOT NULL,
            total_cost REAL NOT NULL,
            fuel_cost REAL DEFAULT 0,
            toll_cost REAL DEFAULT 0,
            overtime_cost REAL DEFAULT 0,
            late_penalty REAL DEFAULT 0,
            orders_served INTEGER NOT NULL,
            orders_late INTEGER DEFAULT 0,
            route_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (truck_id) REFERENCES trucks(truck_id)
        )
        """)
        
        # Bảng route_stops - chi tiết stops trong routes
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS route_stops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            route_id TEXT NOT NULL,
            order_id TEXT NOT NULL,
            stop_sequence INTEGER NOT NULL,
            location TEXT NOT NULL,
            stop_type TEXT NOT NULL, -- pickup, dropoff
            scheduled_arrival TIMESTAMP NOT NULL,
            actual_arrival TIMESTAMP,
            service_time_min INTEGER NOT NULL,
            is_late BOOLEAN DEFAULT FALSE,
            late_minutes INTEGER DEFAULT 0,
            gate_dwell_min INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (route_id) REFERENCES historical_routes(route_id),
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (location) REFERENCES locations(location_code)
        )
        """)
        
        # Bảng cost_factors - các yếu tố chi phí
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cost_factors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_name TEXT UNIQUE NOT NULL,
            base_value REAL NOT NULL,
            time_of_day_multiplier TEXT, -- JSON object with hour -> multiplier
            day_of_week_multiplier TEXT, -- JSON object with day -> multiplier
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Tạo indexes cho performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_tw_start ON orders(tw_start)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_pickup ON orders(pickup_location)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_dropoff ON orders(dropoff_location)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_travel_times_origin ON travel_times(origin_location)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_routes_date ON historical_routes(route_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_route_stops_route ON route_stops(route_id)")
        
        conn.commit()
        conn.close()
        
        logger.info("Database schema created successfully!")
    
    def seed_sample_data(self):
        """Seed sample data vào database"""
        
        logger.info("Seeding sample data...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Seed locations
        locations_data = [
            ("DEPOT_1", "Depot Central", "depot", 10.7769, 106.7009),
            ("DEPOT_2", "Depot North", "depot", 10.8231, 106.6297),
            ("PORT_A_GATE_1", "Cat Lai Port Gate 1", "port", 10.7615, 106.7539),
            ("PORT_A_GATE_2", "Cat Lai Port Gate 2", "port", 10.7625, 106.7549),
            ("PORT_B_GATE_1", "Hiep Phuoc Port Gate 1", "port", 10.6891, 106.7234),
            ("PORT_B_GATE_2", "Hiep Phuoc Port Gate 2", "port", 10.6901, 106.7244),
            ("WAREHOUSE_X", "Warehouse District 7", "warehouse", 10.7378, 106.7019),
            ("WAREHOUSE_Y", "Warehouse Thu Duc", "warehouse", 10.8709, 106.7730),
            ("WAREHOUSE_Z", "Warehouse Binh Duong", "warehouse", 10.9804, 106.6519),
            ("CUSTOMER_A", "Customer Tan Binh", "customer", 10.8015, 106.6525),
            ("CUSTOMER_B", "Customer Go Vap", "customer", 10.8368, 106.6776),
            ("CUSTOMER_C", "Customer Binh Thanh", "customer", 10.8142, 106.7106),
        ]
        
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR IGNORE INTO locations (location_code, location_name, location_type, latitude, longitude)
            VALUES (?, ?, ?, ?, ?)
        """, locations_data)
        
        # Seed trucks
        trucks_data = [
            ("T001", "DEPOT_1", 7, 19, 600, 1500, '["20", "40"]', 5, "D001"),
            ("T002", "DEPOT_1", 8, 20, 540, 1800, '["20", "40"]', 4, "D002"),
            ("T003", "DEPOT_2", 6, 18, 660, 1400, '["20"]', 6, "D003"),
            ("T004", "DEPOT_2", 7, 19, 600, 1600, '["40"]', 4, "D004"),
            ("T005", "DEPOT_1", 8, 20, 600, 1500, '["20", "40"]', 5, "D005"),
        ]
        
        cursor.executemany("""
            INSERT OR IGNORE INTO trucks 
            (truck_id, start_location, shift_start_hour, shift_end_hour, 
             overtime_threshold_min, overtime_rate_per_hour, allowed_sizes, 
             max_orders_per_day, driver_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, trucks_data)
        
        # Seed travel times (ma trận khoảng cách)
        locations = [row[0] for row in locations_data]
        travel_times_data = []
        
        for i, origin in enumerate(locations):
            for j, dest in enumerate(locations):
                if i != j:
                    # Simulate realistic travel times based on location types
                    base_distance = random.uniform(15, 80)  # km
                    base_time = base_distance * random.uniform(1.5, 3.0)  # minutes
                    highway_ratio = random.uniform(0.3, 0.8)
                    highway_distance = base_distance * highway_ratio
                    
                    # Port locations typically have longer access times
                    if "PORT" in origin or "PORT" in dest:
                        base_time *= random.uniform(1.2, 1.5)
                    
                    travel_times_data.append((
                        origin, dest, base_distance, base_time, 
                        highway_distance, random.uniform(0.8, 1.3)
                    ))
        
        cursor.executemany("""
            INSERT OR IGNORE INTO travel_times 
            (origin_location, destination_location, distance_km, 
             travel_time_minutes, highway_distance_km, traffic_factor)
            VALUES (?, ?, ?, ?, ?, ?)
        """, travel_times_data)
        
        # Seed port dwell times
        port_locations = [loc for loc in locations if "PORT" in loc]
        port_dwell_data = []
        
        for port in port_locations:
            for hour in range(24):
                for day in range(7):
                    # Simulate realistic port dwell patterns
                    base_dwell = 30  # minutes
                    
                    # Peak hours (7-9 AM, 1-3 PM) have longer dwell times
                    if hour in [7, 8, 13, 14]:
                        dwell_multiplier = random.uniform(1.5, 2.0)
                    elif hour in [6, 9, 12, 15]:
                        dwell_multiplier = random.uniform(1.2, 1.5)
                    else:
                        dwell_multiplier = random.uniform(0.8, 1.2)
                    
                    # Weekdays are busier than weekends
                    if day < 5:  # Monday to Friday
                        day_multiplier = random.uniform(1.0, 1.3)
                    else:  # Weekend
                        day_multiplier = random.uniform(0.7, 1.0)
                    
                    expected_dwell = int(base_dwell * dwell_multiplier * day_multiplier)
                    queue_length = max(0, int(expected_dwell / 10) + random.randint(-2, 5))
                    
                    port_dwell_data.append((
                        port, f"GATE_{hour%2 + 1}", hour, day, 
                        expected_dwell, queue_length, random.uniform(0.7, 0.95)
                    ))
        
        cursor.executemany("""
            INSERT OR IGNORE INTO port_dwell_times 
            (port_location, gate_id, hour_of_day, day_of_week, 
             expected_dwell_minutes, queue_length, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, port_dwell_data)
        
        # Seed cost factors
        cost_factors_data = [
            ("fuel_cost_per_km", 0.25, '{"7": 1.1, "8": 1.2, "17": 1.2, "18": 1.1}', '{}'),
            ("toll_per_km_highway", 0.15, '{"7": 1.2, "8": 1.3, "17": 1.3, "18": 1.2}', '{}'),
            ("toll_per_km_urban", 0.05, '{"7": 1.1, "8": 1.2, "17": 1.2, "18": 1.1}', '{}'),
            ("late_penalty_per_min", 2.0, '{}', '{}'),
            ("overtime_base_rate", 1500.0, '{}', '{"5": 1.2, "6": 1.5}'),  # Weekend overtime
        ]
        
        cursor.executemany("""
            INSERT OR IGNORE INTO cost_factors 
            (factor_name, base_value, time_of_day_multiplier, day_of_week_multiplier)
            VALUES (?, ?, ?, ?)
        """, cost_factors_data)
        
        conn.commit()
        conn.close()
        
        logger.info("Sample data seeded successfully!")
    
    def generate_historical_data(self, num_routes: int = 500):
        """Generate historical route data để train model"""
        
        logger.info(f"Generating {num_routes} historical routes...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get available trucks and locations
        cursor.execute("SELECT truck_id FROM trucks")
        truck_ids = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("SELECT location_code FROM locations WHERE location_type != 'depot'")
        non_depot_locations = [row[0] for row in cursor.fetchall()]
        
        # Generate historical routes
        historical_routes = []
        route_stops = []
        
        for i in range(num_routes):
            route_id = f"HIST_{i+1:04d}"
            truck_id = random.choice(truck_ids)
            
            # Random date trong 6 tháng qua
            route_date = datetime.now() - timedelta(days=random.randint(1, 180))
            
            # Simulate route characteristics
            num_orders = random.randint(1, 5)
            total_distance = random.uniform(50, 250)
            total_duration = total_distance / random.uniform(25, 45)  # hours
            
            # Calculate costs based on distance and duration
            fuel_cost = total_distance * 0.25 * random.uniform(0.9, 1.1)
            toll_cost = total_distance * 0.1 * random.uniform(0.8, 1.2)
            
            # Overtime cost
            overtime_hours = max(0, total_duration - 10)  # Assuming 10h normal shift
            overtime_cost = overtime_hours * 1500 * random.uniform(0.9, 1.1)
            
            # Late penalty
            late_orders = random.randint(0, min(2, num_orders))
            late_penalty = late_orders * random.uniform(50, 200)
            
            total_cost = fuel_cost + toll_cost + overtime_cost + late_penalty
            
            # Route score (higher is better)
            route_score = random.uniform(0.3, 0.95)
            if late_orders == 0:
                route_score = max(route_score, 0.7)
            if overtime_hours == 0:
                route_score = max(route_score, 0.6)
            
            historical_routes.append((
                route_id, truck_id, route_date.date(), total_distance, 
                total_duration, total_cost, fuel_cost, toll_cost, 
                overtime_cost, late_penalty, num_orders, late_orders, route_score
            ))
            
            # Generate stops for this route
            selected_locations = random.sample(non_depot_locations, min(num_orders * 2, len(non_depot_locations)))
            
            for j in range(num_orders):
                order_id = f"O_{route_id}_{j+1}"
                
                # Pickup stop
                pickup_loc = selected_locations[j*2] if j*2 < len(selected_locations) else selected_locations[0]
                pickup_time = route_date + timedelta(hours=random.uniform(8, 16))
                
                route_stops.append((
                    route_id, order_id, j*2+1, pickup_loc, "pickup",
                    pickup_time, pickup_time, random.randint(15, 30),
                    False, 0, random.randint(0, 45)
                ))
                
                # Dropoff stop
                dropoff_loc = selected_locations[j*2+1] if j*2+1 < len(selected_locations) else selected_locations[1]
                dropoff_time = pickup_time + timedelta(hours=random.uniform(1, 3))
                is_late = random.random() < (late_orders / num_orders)
                late_min = random.randint(5, 60) if is_late else 0
                
                route_stops.append((
                    route_id, order_id, j*2+2, dropoff_loc, "dropoff",
                    dropoff_time, dropoff_time, random.randint(10, 25),
                    is_late, late_min, 0
                ))
        
        # Insert historical routes
        cursor.executemany("""
            INSERT OR IGNORE INTO historical_routes 
            (route_id, truck_id, route_date, total_distance_km, total_duration_hours,
             total_cost, fuel_cost, toll_cost, overtime_cost, late_penalty,
             orders_served, orders_late, route_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, historical_routes)
        
        # Insert route stops
        cursor.executemany("""
            INSERT OR IGNORE INTO route_stops 
            (route_id, order_id, stop_sequence, location, stop_type,
             scheduled_arrival, actual_arrival, service_time_min,
             is_late, late_minutes, gate_dwell_min)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, route_stops)
        
        conn.commit()
        conn.close()
        
        logger.info("Historical data generated successfully!")
    
    def build_prediction_models(self):
        """Build machine learning models từ historical data"""
        
        logger.info("Building prediction models...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load historical data
        query = """
        SELECT 
            hr.total_distance_km,
            hr.total_duration_hours,
            hr.total_cost,
            hr.fuel_cost,
            hr.toll_cost,
            hr.overtime_cost,
            hr.late_penalty,
            hr.orders_served,
            hr.orders_late,
            hr.route_score,
            strftime('%w', hr.route_date) as day_of_week,
            strftime('%H', hr.created_at) as hour_of_day,
            t.shift_start_hour,
            t.shift_end_hour,
            t.overtime_threshold_min,
            t.max_orders_per_day
        FROM historical_routes hr
        JOIN trucks t ON hr.truck_id = t.truck_id
        WHERE hr.route_score IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) < 50:
            logger.warning("Not enough historical data to build reliable models")
            return
        
        logger.info(f"Loaded {len(df)} historical routes for model training")
        
        # Feature engineering
        df['overtime_hours'] = df['total_duration_hours'] - (df['shift_end_hour'] - df['shift_start_hour'])
        df['overtime_hours'] = df['overtime_hours'].clip(lower=0)
        df['utilization_rate'] = df['orders_served'] / df['max_orders_per_day']
        df['distance_per_order'] = df['total_distance_km'] / df['orders_served']
        df['is_weekend'] = df['day_of_week'].isin(['0', '6']).astype(int)  # Sunday=0, Saturday=6
        df['is_peak_hour'] = df['hour_of_day'].astype(int).isin(range(7, 10)).astype(int)
        
        # Features for models
        feature_columns = [
            'total_distance_km', 'orders_served', 'overtime_hours',
            'utilization_rate', 'distance_per_order', 'is_weekend', 'is_peak_hour',
            'shift_start_hour', 'shift_end_hour', 'overtime_threshold_min'
        ]
        
        X = df[feature_columns]
        
        # Build multiple models
        models = {}
        
        # 1. Route Score Prediction Model
        y_score = df['route_score']
        X_train, X_test, y_train, y_test = train_test_split(X, y_score, test_size=0.2, random_state=42)
        
        score_model = RandomForestRegressor(n_estimators=100, random_state=42)
        score_model.fit(X_train, y_train)
        
        y_pred = score_model.predict(X_test)
        score_mae = mean_absolute_error(y_test, y_pred)
        score_r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Route Score Model - MAE: {score_mae:.3f}, R²: {score_r2:.3f}")
        models['route_score'] = score_model
        
        # 2. Total Cost Prediction Model
        y_cost = df['total_cost']
        X_train, X_test, y_train, y_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
        
        cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
        cost_model.fit(X_train, y_train)
        
        y_pred = cost_model.predict(X_test)
        cost_mae = mean_absolute_error(y_test, y_pred)
        cost_r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Total Cost Model - MAE: {cost_mae:.2f}, R²: {cost_r2:.3f}")
        models['total_cost'] = cost_model
        
        # 3. Overtime Prediction Model
        y_overtime = df['overtime_hours']
        X_train, X_test, y_train, y_test = train_test_split(X, y_overtime, test_size=0.2, random_state=42)
        
        overtime_model = RandomForestRegressor(n_estimators=100, random_state=42)
        overtime_model.fit(X_train, y_train)
        
        y_pred = overtime_model.predict(X_test)
        overtime_mae = mean_absolute_error(y_test, y_pred)
        overtime_r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Overtime Model - MAE: {overtime_mae:.3f} hours, R²: {overtime_r2:.3f}")
        models['overtime_hours'] = overtime_model
        
        # Save models
        for model_name, model in models.items():
            model_file = self.model_path / f"{model_name}_model.joblib"
            joblib.dump(model, model_file)
            logger.info(f"Saved {model_name} model to {model_file}")
        
        # Save feature columns for later use
        feature_info = {
            'feature_columns': feature_columns,
            'model_metrics': {
                'route_score': {'mae': score_mae, 'r2': score_r2},
                'total_cost': {'mae': cost_mae, 'r2': cost_r2},
                'overtime_hours': {'mae': overtime_mae, 'r2': overtime_r2}
            },
            'training_data_size': len(df),
            'created_at': datetime.now().isoformat()
        }
        
        feature_info_file = self.model_path / "model_info.json"
        with open(feature_info_file, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        logger.info("Models built and saved successfully!")
        
        return models
    
    def run_full_migration(self):
        """Chạy toàn bộ migration process"""
        
        logger.info("Starting full migration process...")
        
        # Step 1: Create database schema
        self.create_database_schema()
        
        # Step 2: Seed sample data
        self.seed_sample_data()
        
        # Step 3: Generate historical data
        self.generate_historical_data(num_routes=500)
        
        # Step 4: Build prediction models
        self.build_prediction_models()
        
        logger.info("Full migration completed successfully!")
        
        # Print summary
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM locations")
        locations_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trucks")
        trucks_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM travel_times")
        travel_times_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM historical_routes")
        routes_count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                     MIGRATION SUMMARY                       ║
╠══════════════════════════════════════════════════════════════╣
║ Database: {self.db_path:<48} ║
║ Locations: {locations_count:<47} ║
║ Trucks: {trucks_count:<50} ║
║ Travel Times: {travel_times_count:<44} ║
║ Historical Routes: {routes_count:<41} ║
║ Models: route_score, total_cost, overtime_hours             ║
║ Models Path: {str(self.model_path):<45} ║
╚══════════════════════════════════════════════════════════════╝
        """)


def main():
    """Main function để chạy migration"""
    
    print("🚛 Logistics Optimization - Database Migration")
    print("=" * 60)
    
    migration = LogisticsMigration()
    
    try:
        migration.run_full_migration()
        
        print("\n✅ Migration completed successfully!")
        print("📊 Models are ready for prediction")
        print("🔗 Update your API to use the new models")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        print(f"\n❌ Migration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
