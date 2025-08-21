#!/usr/bin/env python3
"""
ML System Test & Validation Script
Kiểm tra và validate toàn bộ hệ thống ML
"""

import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class MLSystemTester:
    """Class để test và validate ML system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = {}
        
    def print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print(f"{'='*60}")
    
    def print_result(self, test_name: str, success: bool, details: str = ""):
        """Print test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")
        
        self.test_results[test_name] = {
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
    
    def test_api_health(self) -> bool:
        """Test API health"""
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                self.print_result("API Health Check", True, f"Status: {health_data.get('status', 'unknown')}")
                return True
            else:
                self.print_result("API Health Check", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.print_result("API Health Check", False, f"Connection error: {e}")
            return False
    
    def test_ml_status(self) -> dict:
        """Test ML model status"""
        
        try:
            response = requests.get(f"{self.base_url}/ml/status", timeout=10)
            
            if response.status_code == 200:
                ml_status = response.json()
                
                models_loaded = ml_status.get('models_loaded', 0)
                total_models = ml_status.get('total_models', 3)
                
                success = models_loaded > 0
                details = f"Models: {models_loaded}/{total_models} loaded"
                
                if models_loaded == total_models:
                    details += " (Full ML capability)"
                elif models_loaded > 0:
                    details += " (Partial ML capability)"
                else:
                    details += " (No ML capability)"
                
                self.print_result("ML Models Status", success, details)
                return ml_status
                
            else:
                self.print_result("ML Models Status", False, f"HTTP {response.status_code}")
                return {}
                
        except Exception as e:
            self.print_result("ML Models Status", False, f"Error: {e}")
            return {}
    
    def test_solver_capabilities(self) -> dict:
        """Test solver status and capabilities"""
        
        try:
            response = requests.get(f"{self.base_url}/solver/status", timeout=10)
            
            if response.status_code == 200:
                solver_status = response.json()
                
                ml_enhanced = solver_status.get('ml_enhanced', False)
                algorithms = solver_status.get('solver_info', {}).get('available_algorithms', [])
                
                success = len(algorithms) > 0
                details = f"Algorithms: {len(algorithms)}, ML Enhanced: {ml_enhanced}"
                
                self.print_result("Solver Capabilities", success, details)
                return solver_status
                
            else:
                self.print_result("Solver Capabilities", False, f"HTTP {response.status_code}")
                return {}
                
        except Exception as e:
            self.print_result("Solver Capabilities", False, f"Error: {e}")
            return {}
    
    def test_dispatch_optimization(self) -> bool:
        """Test dispatch optimization with sample data"""
        
        try:
            # Load example request
            example_file = project_root / "example_request.json"
            if not example_file.exists():
                self.print_result("Dispatch Optimization", False, "example_request.json not found")
                return False
            
            with open(example_file, 'r') as f:
                request_data = json.load(f)
            
            # Make request
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/dispatch/suggest",
                json=request_data,
                timeout=30
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                routes_count = len(result.get('routes', []))
                orders_count = len(request_data.get('orders', []))
                unserved_count = len(result.get('unserved_orders', []))
                service_rate = (orders_count - unserved_count) / orders_count if orders_count > 0 else 0
                
                success = routes_count > 0 and service_rate > 0.5
                details = f"Routes: {routes_count}, Service rate: {service_rate:.1%}, Time: {response_time:.2f}s"
                
                # Check if ML was used
                metadata = result.get('metadata', {})
                algorithm_used = metadata.get('algorithm_used', 'unknown')
                ml_models_used = metadata.get('ml_models_used', 0)
                
                if ml_models_used > 0:
                    details += f", ML models used: {ml_models_used}"
                
                details += f", Algorithm: {algorithm_used}"
                
                self.print_result("Dispatch Optimization", success, details)
                return success
                
            else:
                self.print_result("Dispatch Optimization", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.print_result("Dispatch Optimization", False, f"Error: {e}")
            return False
    
    def test_ml_prediction(self) -> bool:
        """Test ML prediction endpoint"""
        
        try:
            # Load example request for ML test
            example_file = project_root / "example_request.json"
            if not example_file.exists():
                self.print_result("ML Prediction Test", False, "example_request.json not found")
                return False
            
            with open(example_file, 'r') as f:
                request_data = json.load(f)
            
            response = requests.post(
                f"{self.base_url}/test/ml-prediction",
                json=request_data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                
                prediction = result.get('prediction', {})
                predicted_score = prediction.get('predicted_score', 0)
                predicted_cost = prediction.get('predicted_cost', 0)
                confidence = prediction.get('confidence', 0)
                
                success = predicted_score > 0 and predicted_cost > 0 and confidence > 0
                details = f"Score: {predicted_score:.3f}, Cost: ${predicted_cost:.0f}, Confidence: {confidence:.3f}"
                
                self.print_result("ML Prediction Test", success, details)
                return success
                
            else:
                # ML prediction might not be available if no models
                self.print_result("ML Prediction Test", False, f"HTTP {response.status_code} (Expected if no ML models)")
                return False
                
        except Exception as e:
            self.print_result("ML Prediction Test", False, f"Error: {e}")
            return False
    
    def test_model_files(self) -> dict:
        """Test model files existence and validity"""
        
        models_path = project_root / "models"
        expected_files = [
            "model_info.json",
            "route_score_model.joblib", 
            "total_cost_model.joblib",
            "overtime_hours_model.joblib"
        ]
        
        file_status = {}
        
        for filename in expected_files:
            file_path = models_path / filename
            exists = file_path.exists()
            
            if exists:
                try:
                    if filename.endswith('.json'):
                        with open(file_path, 'r') as f:
                            json.load(f)  # Validate JSON
                        file_status[filename] = "valid"
                    else:
                        # Check file size for model files
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        file_status[filename] = f"valid ({size_mb:.1f}MB)"
                        
                except Exception as e:
                    file_status[filename] = f"corrupted: {e}"
            else:
                file_status[filename] = "missing"
        
        # Overall assessment
        valid_files = sum(1 for status in file_status.values() if status.startswith("valid"))
        total_files = len(expected_files)
        
        success = valid_files == total_files
        details = f"Valid files: {valid_files}/{total_files}"
        
        self.print_result("Model Files Check", success, details)
        
        # Print individual file status
        for filename, status in file_status.items():
            status_icon = "✅" if status.startswith("valid") else "❌" if status == "missing" else "⚠️"
            print(f"   {status_icon} {filename}: {status}")
        
        return file_status
    
    def run_performance_benchmark(self) -> dict:
        """Run performance benchmark"""
        
        self.print_header("Performance Benchmark")
        
        # Test với different request sizes
        test_cases = [
            {"orders": 10, "trucks": 3, "expected_time": 2.0},
            {"orders": 20, "trucks": 5, "expected_time": 4.0},
            {"orders": 50, "trucks": 10, "expected_time": 8.0}
        ]
        
        benchmark_results = {}
        
        for i, case in enumerate(test_cases):
            print(f"\n📊 Benchmark Case {i+1}: {case['orders']} orders, {case['trucks']} trucks")
            
            # Generate test data (simplified)
            test_request = self._generate_test_request(case['orders'], case['trucks'])
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/dispatch/suggest",
                    json=test_request,
                    timeout=30
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    routes_count = len(result.get('routes', []))
                    service_rate = self._calculate_service_rate(result, case['orders'])
                    
                    within_expected = response_time <= case['expected_time']
                    
                    benchmark_results[f"case_{i+1}"] = {
                        "response_time": response_time,
                        "expected_time": case['expected_time'],
                        "within_expected": within_expected,
                        "routes_generated": routes_count,
                        "service_rate": service_rate
                    }
                    
                    status = "✅" if within_expected else "⚠️"
                    print(f"   {status} Response time: {response_time:.2f}s (expected ≤ {case['expected_time']}s)")
                    print(f"   📍 Routes: {routes_count}, Service rate: {service_rate:.1%}")
                    
                else:
                    print(f"   ❌ HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return benchmark_results
    
    def _generate_test_request(self, num_orders: int, num_trucks: int) -> dict:
        """Generate test request data"""
        
        # Simplified test data generation
        orders = []
        for i in range(num_orders):
            orders.append({
                "order_id": f"ORDER_{i+1:03d}",
                "pickup": f"PICKUP_LOC_{i % 5 + 1}",
                "dropoff": f"DROPOFF_LOC_{i % 3 + 1}",
                "container_size": "40",
                "tw_start": "2025-08-22T08:00:00",
                "tw_end": "2025-08-22T18:00:00",
                "service_time_min": 30,
                "priority": 1
            })
        
        trucks = []
        for i in range(num_trucks):
            trucks.append({
                "truck_id": f"TRUCK_{i+1:03d}",
                "driver_id": f"DRIVER_{i+1:03d}",
                "capacity": 2,
                "current_location": "BASE_DEPOT",
                "shift_start": "2025-08-22T06:00:00",
                "shift_end": "2025-08-22T20:00:00",
                "overtime_threshold_min": 480
            })
        
        return {
            "orders": orders,
            "fleet": trucks,
            "costs": {
                "cost_per_km": 1.5,
                "cost_per_hour": 25.0,
                "overtime_rate": 35.0,
                "late_penalty_per_min": 2.0
            },
            "weights": {
                "distance_weight": 0.4,
                "time_weight": 0.3,
                "overtime_weight": 0.2,
                "delay_weight": 0.1
            }
        }
    
    def _calculate_service_rate(self, result: dict, total_orders: int) -> float:
        """Calculate service rate from result"""
        
        unserved_count = len(result.get('unserved_orders', []))
        served_count = total_orders - unserved_count
        return served_count / total_orders if total_orders > 0 else 0
    
    def run_full_test_suite(self):
        """Run complete test suite"""
        
        print("🚛 ML System Test Suite - Logistics Optimization")
        print(f"Started at: {datetime.now()}")
        print(f"Target API: {self.base_url}")
        
        # Core API tests
        self.print_header("Core API Tests")
        api_healthy = self.test_api_health()
        
        if not api_healthy:
            print("\n❌ API not accessible. Please start the server first:")
            print("   python run_server.py")
            return False
        
        # ML System tests
        self.print_header("ML System Tests")
        ml_status = self.test_ml_status()
        solver_status = self.test_solver_capabilities()
        
        # Model files check
        self.print_header("Model Files Validation")
        file_status = self.test_model_files()
        
        # Functional tests
        self.print_header("Functional Tests")
        dispatch_success = self.test_dispatch_optimization()
        ml_prediction_success = self.test_ml_prediction()
        
        # Performance benchmark
        benchmark_results = self.run_performance_benchmark()
        
        # Summary
        self.print_header("Test Summary")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        
        print(f"📊 Test Results: {passed_tests}/{total_tests} passed")
        print(f"🕒 Test Duration: {time.time() - self.start_time:.2f} seconds")
        
        # Recommendations
        print(f"\n🎯 Recommendations:")
        
        if ml_status.get('models_loaded', 0) == 0:
            print("   1. 🔧 Build ML models: python build_models.py")
        
        if not dispatch_success:
            print("   2. 🚨 Check dispatch optimization functionality")
        
        if passed_tests < total_tests:
            print("   3. 📋 Review failed tests and fix issues")
        
        if passed_tests == total_tests:
            print("   ✅ All tests passed! System is ready for production.")
        
        # Save test results
        self._save_test_results(benchmark_results)
        
        return passed_tests == total_tests
    
    def _save_test_results(self, benchmark_results: dict):
        """Save test results to file"""
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "test_results": self.test_results,
            "benchmark_results": benchmark_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed_tests": sum(1 for r in self.test_results.values() if r['success']),
                "duration_seconds": time.time() - self.start_time
            }
        }
        
        results_file = project_root / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Test results saved to: {results_file}")


def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="ML System Test Suite")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    
    args = parser.parse_args()
    
    tester = MLSystemTester(args.url)
    tester.start_time = time.time()
    
    if args.quick:
        # Quick health check
        tester.print_header("Quick Health Check")
        api_healthy = tester.test_api_health()
        ml_status = tester.test_ml_status()
        
        if api_healthy and ml_status.get('models_loaded', 0) > 0:
            print("\n✅ System appears healthy with ML capability")
        elif api_healthy:
            print("\n⚠️ System healthy but no ML models loaded")
        else:
            print("\n❌ System not accessible")
    else:
        # Full test suite
        success = tester.run_full_test_suite()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
