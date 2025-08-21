#!/usr/bin/env python3
"""Demo script cho MVP Dispatch Optimization"""

import asyncio
import json
import time
from datetime import datetime, timedelta

from app.schemas.dispatch import DispatchRequest, WeightConfig  
from app.schemas.orders import Order
from app.schemas.fleet import Truck
from app.schemas.costs import CostConfig
from app.services.travel_time import TravelTimeService
from app.services.port_api import PortApiService
from app.solvers.local_search import LocalSearchSolver


async def demo_complete_system():
    """Demo toàn bộ hệ thống optimization"""
    
    print("=== MVP Dispatch Optimization Demo ===\n")
    
    # Tạo test data
    print("1. Tạo test data...")
    
    # Orders
    base_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    orders = [
        Order(
            order_id="O001",
            pickup="PORT_A_GATE_1",
            dropoff="WAREHOUSE_X",
            container_size="40",
            tw_start=base_time,
            tw_end=base_time + timedelta(hours=4),
            service_time_min=20,
            priority=2
        ),
        Order(
            order_id="O002", 
            pickup="PORT_A_GATE_2",
            dropoff="WAREHOUSE_Y",
            container_size="20",
            tw_start=base_time + timedelta(hours=1),
            tw_end=base_time + timedelta(hours=6),
            service_time_min=15,
            priority=1
        ),
        Order(
            order_id="O003",
            pickup="PORT_B_GATE_1", 
            dropoff="WAREHOUSE_X",
            container_size="40",
            tw_start=base_time + timedelta(hours=2),
            tw_end=base_time + timedelta(hours=8),
            service_time_min=25,
            priority=3
        ),
        Order(
            order_id="O004",
            pickup="WAREHOUSE_Y",
            dropoff="PORT_A_GATE_1",
            container_size="20", 
            tw_start=base_time + timedelta(hours=3),
            tw_end=base_time + timedelta(hours=7),
            service_time_min=30,
            priority=2
        )
    ]
    
    # Trucks
    trucks = [
        Truck(
            truck_id="T001",
            start_location="DEPOT_1",
            shift_start=base_time - timedelta(hours=1),
            shift_end=base_time + timedelta(hours=12),
            overtime_threshold_min=600,
            overtime_rate_per_hour=1500,
            allowed_sizes=["20", "40"],
            max_orders_per_day=5
        ),
        Truck(
            truck_id="T002",
            start_location="DEPOT_1", 
            shift_start=base_time,
            shift_end=base_time + timedelta(hours=12),
            overtime_threshold_min=540,
            overtime_rate_per_hour=1800,
            allowed_sizes=["20", "40"],
            max_orders_per_day=4
        )
    ]
    
    # Cost config
    cost_config = CostConfig(
        fuel_cost_per_km=0.25,
        avg_consumption_km_per_liter=4.0,
        toll_per_km_highway=0.15,
        toll_per_km_urban=0.05,
        toll_multiplier_peak=1.2,
        late_penalty_per_min=2.0,
        overtime_base_rate=1500.0
    )
    
    # Weights
    weights = WeightConfig(
        lambda_late=1.0,
        lambda_ot=1.0,
        lambda_tw=10.0,
        lambda_priority=0.5
    )
    
    print(f"   Created {len(orders)} orders and {len(trucks)} trucks")
    
    # Initialize services
    print("\n2. Khởi tạo services...")
    
    travel_service = TravelTimeService()
    port_service = PortApiService()
    
    try:
        await travel_service.__aenter__()
        await port_service.__aenter__()
        print("   Services initialized successfully")
        
        # Test individual services
        print("\n3. Test individual services...")
        
        # Test travel time
        travel_result = await travel_service.get_travel_time(
            "PORT_A_GATE_1", "WAREHOUSE_X", datetime.now()
        )
        print(f"   Travel time PORT_A_GATE_1 -> WAREHOUSE_X: {travel_result.travel_time_minutes:.1f} min, {travel_result.distance_km:.1f} km")
        
        # Test port dwell
        dwell_info = await port_service.get_gate_dwell_info("PORT_A_GATE_1")
        print(f"   Port dwell PORT_A_GATE_1: {dwell_info.expected_dwell_minutes} min (confidence: {dwell_info.confidence:.2f})")
        
        # Run optimization
        print("\n4. Chạy optimization...")
        
        solver = LocalSearchSolver(cost_config, weights)
        solver.set_services(travel_service, port_service)
        
        start_time = time.time()
        result = await solver.solve(orders, trucks, max_iterations=50, time_limit_seconds=30)
        solve_time = time.time() - start_time
        
        print(f"   Optimization completed in {solve_time:.2f}s")
        print(f"   Algorithm: {result.algorithm}")
        print(f"   Iterations: {result.iterations}")
        print(f"   Routes generated: {len(result.routes)}")
        print(f"   Unserved orders: {len(result.unserved_orders)}")
        
        # Show results
        print("\n5. Kết quả optimization:")
        
        total_cost = sum(route.cost_breakdown.total_cost for route in result.routes)
        total_distance = sum(route.total_distance_km for route in result.routes)
        total_late = sum(route.cost_breakdown.late_minutes for route in result.routes)
        total_overtime = sum(route.cost_breakdown.overtime_hours for route in result.routes)
        
        print(f"   Tổng chi phí: ¥{total_cost:.2f}")
        print(f"   Tổng khoảng cách: {total_distance:.1f} km")
        print(f"   Số phút trễ: {total_late}")
        print(f"   Overtime: {total_overtime:.1f} hours")
        
        # Route details
        for i, route in enumerate(result.routes):
            print(f"\n   Route {i+1} (Truck {route.truck_id}):")
            print(f"     Orders: {route.order_ids}")
            print(f"     Distance: {route.total_distance_km:.1f} km")
            print(f"     Duration: {route.total_duration_hours:.1f} hours")
            print(f"     Cost breakdown:")
            print(f"       Fuel: ¥{route.cost_breakdown.fuel_cost:.2f}")
            print(f"       Toll: ¥{route.cost_breakdown.toll_cost:.2f}")
            print(f"       Overtime: ¥{route.cost_breakdown.overtime_cost:.2f}")
            print(f"       Penalty: ¥{route.cost_breakdown.penalty_cost:.2f}")
            print(f"       Total: ¥{route.cost_breakdown.total_cost:.2f}")
            print(f"     Explain: {route.explain}")
            
            # Show stops
            for stop in route.stops:
                eta_str = stop.eta.strftime("%H:%M")
                etd_str = stop.etd.strftime("%H:%M")
                status = "LATE" if stop.is_late else "OK"
                print(f"       {stop.stop_type} at {stop.location}: {eta_str}-{etd_str} [{status}]")
                if stop.note:
                    print(f"         Note: {stop.note}")
        
        # Unserved orders
        if result.unserved_orders:
            print(f"\n   Unserved orders: {result.unserved_orders}")
        
        # Performance metrics  
        print(f"\n6. Performance metrics:")
        served_ratio = (len(orders) - len(result.unserved_orders)) / len(orders)
        print(f"   Service ratio: {served_ratio:.1%}")
        print(f"   Cost per order: ¥{total_cost / max(1, len(orders) - len(result.unserved_orders)):.2f}")
        print(f"   Distance efficiency: {total_distance / max(1, len(orders) - len(result.unserved_orders)):.1f} km/order")
        
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            await travel_service.__aexit__(None, None, None)
            await port_service.__aexit__(None, None, None)
        except:
            pass
    
    print("\n=== Demo completed ===")


async def demo_api_json_format():
    """Demo tạo JSON request format cho API"""
    
    print("\n=== Tạo API Request Format ===")
    
    base_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    
    request_data = {
        "orders": [
            {
                "order_id": "O001",
                "pickup": "PORT_A_GATE_1",
                "dropoff": "WAREHOUSE_X",
                "container_size": "40",
                "tw_start": (base_time).isoformat(),
                "tw_end": (base_time + timedelta(hours=4)).isoformat(),
                "service_time_min": 20,
                "priority": 2
            }
        ],
        "fleet": [
            {
                "truck_id": "T001",
                "start_location": "DEPOT_1",
                "shift_start": (base_time - timedelta(hours=1)).isoformat(),
                "shift_end": (base_time + timedelta(hours=12)).isoformat(),
                "overtime_threshold_min": 600,
                "overtime_rate_per_hour": 1500,
                "allowed_sizes": ["20", "40"],
                "max_orders_per_day": 5
            }
        ],
        "costs": {
            "fuel_cost_per_km": 0.25,
            "avg_consumption_km_per_liter": 4.0,
            "toll_per_km_highway": 0.15,
            "toll_per_km_urban": 0.05,
            "toll_multiplier_peak": 1.2,
            "late_penalty_per_min": 2.0,
            "overtime_base_rate": 1500.0
        },
        "weights": {
            "lambda_late": 1.0,
            "lambda_ot": 1.0,
            "lambda_tw": 10.0,
            "lambda_priority": 0.5
        },
        "max_iterations": 100,
        "time_limit_seconds": 30
    }
    
    print("Sample API request JSON:")
    print(json.dumps(request_data, indent=2, ensure_ascii=False))


async def main():
    """Main demo function"""
    choice = input("Choose demo:\n1. Complete system demo\n2. API JSON format\n3. Both\nChoice (1/2/3): ").strip()
    
    if choice in ["1", "3"]:
        await demo_complete_system()
    
    if choice in ["2", "3"]:
        await demo_api_json_format()


if __name__ == "__main__":
    asyncio.run(main())
