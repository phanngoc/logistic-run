#!/usr/bin/env python3
"""Test script cho MVP API"""

import asyncio
import json
import httpx
from datetime import datetime


async def test_api():
    """Test basic API functionality"""
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        
        print("=== Testing MVP Dispatch Optimization API ===\n")
        
        # Test health check
        print("1. Testing health check...")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test solver status
        print("2. Testing solver status...")
        try:
            response = await client.get(f"{base_url}/solver/status")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test travel time
        print("3. Testing travel time...")
        try:
            response = await client.post(f"{base_url}/test/travel-time", params={
                "origin": "PORT_A_GATE_1",
                "destination": "WAREHOUSE_X"
            })
            print(f"   Status: {response.status_code}")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test port dwell
        print("4. Testing port dwell...")
        try:
            response = await client.post(f"{base_url}/test/port-dwell", params={
                "location": "PORT_A_GATE_1"
            })
            print(f"   Status: {response.status_code}")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test dispatch optimization
        print("5. Testing dispatch optimization...")
        try:
            # Load example request
            with open("example_request.json", "r") as f:
                request_data = json.load(f)
            
            print(f"   Sending request with {len(request_data['orders'])} orders and {len(request_data['fleet'])} trucks")
            
            start_time = datetime.now()
            response = await client.post(
                f"{base_url}/dispatch/suggest",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            end_time = datetime.now()
            
            print(f"   Status: {response.status_code}")
            print(f"   Request time: {(end_time - start_time).total_seconds():.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Success: {result['success']}")
                print(f"   Algorithm: {result['algorithm']}")
                print(f"   Solve time: {result['solve_time_seconds']:.2f}s")
                print(f"   Iterations: {result['iterations']}")
                print(f"   Routes: {len(result['routes'])}")
                print(f"   Served orders: {result['kpi']['served_orders']}/{result['kpi']['total_orders']}")
                print(f"   Unserved orders: {len(result['unserved_orders'])}")
                print(f"   Total cost: ¥{result['kpi']['total_cost']:.2f}")
                print(f"   Total distance: {result['kpi']['total_distance_km']:.1f} km")
                print(f"   Late orders: {result['kpi']['late_orders']}")
                print(f"   Overtime hours: {result['kpi']['overtime_hours']:.1f}")
                
                # Show route details
                for i, route in enumerate(result['routes']):
                    print(f"\n   Route {i+1} (Truck {route['truck_id']}):")
                    print(f"     Orders: {route['order_ids']}")
                    print(f"     Distance: {route['total_distance_km']:.1f} km")
                    print(f"     Duration: {route['total_duration_hours']:.1f} hours")
                    print(f"     Cost: ¥{route['cost_breakdown']['total_cost']:.2f}")
                    print(f"     Explain: {route['explain']}")
                    
                    for stop in route['stops']:
                        print(f"       {stop['stop_type']} at {stop['location']} - ETA: {stop['eta']}")
                        if stop['note']:
                            print(f"         Note: {stop['note']}")
            else:
                print(f"   Error response: {response.text}")
                
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\n=== Test completed ===")


if __name__ == "__main__":
    asyncio.run(test_api())
