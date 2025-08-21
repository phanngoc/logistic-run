#!/usr/bin/env python3
"""
Demo Script - ML-Enhanced Logistics Optimization System
Comprehensive demo of the ML system capabilities
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path

class LogisticsDemo:
    """Demo class for ML-enhanced logistics system"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def print_banner(self, text):
        """Print formatted banner"""
        print(f"\n{'='*80}")
        print(f"🚛 {text}")
        print(f"{'='*80}")
    
    def print_section(self, text):
        """Print section header"""
        print(f"\n{'─'*60}")
        print(f"📋 {text}")
        print(f"{'─'*60}")
    
    def check_system_health(self):
        """Check if system is running and healthy"""
        
        self.print_section("System Health Check")
        
        try:
            # Basic health check
            health_response = requests.get(f"{self.base_url}/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"✅ API Status: {health_data.get('status', 'unknown')}")
                
                services = health_data.get('services', {})
                for service, status in services.items():
                    icon = "✅" if status == "healthy" else "⚠️"
                    print(f"{icon} {service}: {status}")
                
                return True
            else:
                print(f"❌ API Health Check Failed: HTTP {health_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Cannot connect to system: {e}")
            print(f"💡 Make sure server is running: python run_server.py")
            return False
    
    def show_ml_capabilities(self):
        """Display ML system capabilities"""
        
        self.print_section("ML System Capabilities")
        
        try:
            # Get ML status
            ml_response = requests.get(f"{self.base_url}/ml/status", timeout=10)
            if ml_response.status_code == 200:
                ml_data = ml_response.json()
                
                models_loaded = ml_data.get('models_loaded', 0)
                available_models = ml_data.get('available_models', [])
                
                print(f"🤖 ML Models Loaded: {models_loaded}/3")
                print(f"📊 Available Models: {', '.join(available_models)}")
                
                # Model performance
                model_info = ml_data.get('model_info', {})
                if 'metrics' in model_info:
                    print(f"\n📈 Model Performance:")
                    for model_name, metrics in model_info['metrics'].items():
                        mae = metrics.get('mae', 0)
                        r2 = metrics.get('r2', 0)
                        
                        # Performance rating
                        if r2 > 0.7:
                            rating = "🌟 Excellent"
                        elif r2 > 0.5:
                            rating = "👍 Good"
                        elif r2 > 0.2:
                            rating = "✅ Acceptable"
                        else:
                            rating = "⚠️ Needs Improvement"
                        
                        print(f"   • {model_name}: MAE={mae:.3f}, R²={r2:.3f} {rating}")
                
                # Training data info
                training_size = model_info.get('training_data_size', 0)
                created_at = model_info.get('created_at', 'unknown')
                print(f"\n📚 Training Data: {training_size} samples")
                print(f"🕒 Last Updated: {created_at}")
                
                # Insights
                insights = ml_data.get('insights', [])
                if insights:
                    print(f"\n💡 Insights:")
                    for insight in insights[:3]:
                        print(f"   {insight}")
                
                return models_loaded > 0
                
            else:
                print(f"❌ ML Status Check Failed: HTTP {ml_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ ML Status Error: {e}")
            return False
    
    def show_solver_features(self):
        """Display solver capabilities"""
        
        self.print_section("Optimization Solver Features")
        
        try:
            solver_response = requests.get(f"{self.base_url}/solver/status", timeout=10)
            if solver_response.status_code == 200:
                solver_data = solver_response.json()
                
                solver_info = solver_data.get('solver_info', {})
                algorithms = solver_info.get('available_algorithms', [])
                default_algo = solver_info.get('default_algorithm', 'unknown')
                ml_enhanced = solver_info.get('ml_enhanced', False)
                
                print(f"🔧 Available Algorithms: {len(algorithms)}")
                for algo in algorithms:
                    icon = "🤖" if "ml" in algo else "⚡"
                    marker = " (default)" if algo == default_algo else ""
                    print(f"   {icon} {algo}{marker}")
                
                print(f"\n🚀 ML Enhanced: {'✅ Yes' if ml_enhanced else '❌ No'}")
                
                # Features
                features = solver_data.get('features', {})
                ml_features = []
                base_features = []
                
                for feature_name, feature_info in features.items():
                    if isinstance(feature_info, dict):
                        available = feature_info.get('available', False)
                        icon = "✅" if available else "❌"
                        
                        if 'ml_' in feature_name:
                            ml_features.append(f"   {icon} {feature_name}")
                        else:
                            base_features.append(f"   {icon} {feature_name}")
                    else:
                        icon = "✅" if feature_info else "❌"
                        base_features.append(f"   {icon} {feature_name}")
                
                if base_features:
                    print(f"\n⚡ Base Features:")
                    for feature in base_features[:5]:  # Show top 5
                        print(feature)
                
                if ml_features:
                    print(f"\n🤖 ML Features:")
                    for feature in ml_features:
                        print(feature)
                
                # Performance benchmarks
                benchmarks = solver_data.get('performance_benchmarks', {})
                if benchmarks:
                    print(f"\n📊 Performance Benchmarks:")
                    for benchmark, value in benchmarks.items():
                        if 'time' in benchmark:
                            print(f"   ⏱️ {benchmark}: {value}")
                        else:
                            print(f"   📈 {benchmark}: {value}")
                
                return True
                
            else:
                print(f"❌ Solver Status Failed: HTTP {solver_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Solver Status Error: {e}")
            return False
    
    def demo_optimization(self):
        """Demo the optimization capability"""
        
        self.print_section("Live Optimization Demo")
        
        # Load example request
        example_file = Path("example_request.json")
        if not example_file.exists():
            print("❌ example_request.json not found")
            return False
        
        try:
            with open(example_file, 'r') as f:
                request_data = json.load(f)
            
            orders_count = len(request_data.get('orders', []))
            trucks_count = len(request_data.get('fleet', []))
            
            print(f"📦 Orders: {orders_count}")
            print(f"🚛 Trucks: {trucks_count}")
            print(f"🎯 Goal: Optimize routes with ML-enhanced algorithms")
            
            print(f"\n⏳ Running optimization...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/dispatch/suggest",
                json=request_data,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Basic results
                routes = result.get('routes', [])
                unserved = result.get('unserved_orders', [])
                kpi = result.get('kpi', {})
                algorithm = result.get('algorithm', 'unknown')
                
                print(f"\n🎉 Optimization Results:")
                print(f"   ⏱️ Response Time: {response_time:.2f} seconds")
                print(f"   🔧 Algorithm Used: {algorithm}")
                print(f"   📍 Routes Generated: {len(routes)}")
                print(f"   ✅ Orders Served: {orders_count - len(unserved)}/{orders_count}")
                print(f"   📊 Service Rate: {((orders_count - len(unserved)) / orders_count * 100):.1f}%")
                
                # Cost analysis
                total_cost = kpi.get('total_cost', 0)
                total_distance = kpi.get('total_distance_km', 0)
                overtime_hours = kpi.get('overtime_hours', 0)
                
                print(f"\n💰 Cost Analysis:")
                print(f"   💵 Total Cost: ${total_cost:.2f}")
                print(f"   🛣️ Total Distance: {total_distance:.1f} km")
                print(f"   ⏰ Overtime Hours: {overtime_hours:.1f} hours")
                
                # ML insights
                metadata = result.get('metadata', {})
                ml_models_used = metadata.get('ml_models_used', 0)
                algorithm_used = metadata.get('algorithm_used', 'unknown')
                
                if ml_models_used > 0:
                    print(f"\n🤖 ML Enhancement:")
                    print(f"   🧠 Models Used: {ml_models_used}")
                    print(f"   🎯 Algorithm: {algorithm_used}")
                    
                    # Show sample route insights
                    if routes:
                        sample_route = routes[0]
                        route_score = sample_route.get('score', 0)
                        explain = sample_route.get('explain', '')
                        
                        print(f"   📊 Sample Route Score: {route_score:.3f}")
                        if 'ML' in explain:
                            ml_part = explain.split('ML:')[1].split('|')[0] if 'ML:' in explain else ''
                            print(f"   💡 ML Insight: {ml_part.strip()}")
                
                # Route details
                print(f"\n🗺️ Route Summary:")
                for i, route in enumerate(routes[:2], 1):  # Show first 2 routes
                    truck_id = route.get('truck_id', 'unknown')
                    stops_count = len(route.get('stops', []))
                    route_distance = route.get('total_distance_km', 0)
                    route_duration = route.get('total_duration_hours', 0)
                    
                    print(f"   Route {i} (Truck {truck_id}):")
                    print(f"     • {stops_count} stops, {route_distance:.1f}km, {route_duration:.1f}h")
                
                return True
                
            else:
                print(f"❌ Optimization Failed: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"Response: {response.text[:200]}...")
                return False
                
        except Exception as e:
            print(f"❌ Demo Error: {e}")
            return False
    
    def show_recommendations(self):
        """Show system recommendations"""
        
        self.print_section("System Recommendations")
        
        try:
            ml_response = requests.get(f"{self.base_url}/ml/status", timeout=10)
            if ml_response.status_code == 200:
                ml_data = ml_response.json()
                
                recommendations = ml_data.get('recommendations', {})
                
                immediate = recommendations.get('immediate_actions', [])
                if immediate:
                    print("🚨 Immediate Actions:")
                    for action in immediate[:3]:
                        print(f"   • {action}")
                
                improvements = recommendations.get('performance_improvements', [])
                if improvements:
                    print("\n📈 Performance Improvements:")
                    for improvement in improvements[:3]:
                        print(f"   • {improvement}")
                
                long_term = recommendations.get('long_term_goals', [])
                if long_term:
                    print("\n🎯 Long-term Goals:")
                    for goal in long_term[:3]:
                        print(f"   • {goal}")
                
                return True
                
        except Exception as e:
            print(f"❌ Recommendations Error: {e}")
            return False
    
    def run_full_demo(self):
        """Run complete system demo"""
        
        self.print_banner("ML-Enhanced Logistics Route Optimization Demo")
        
        print(f"🕒 Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Target System: {self.base_url}")
        
        # Step 1: Health check
        if not self.check_system_health():
            print("\n❌ Demo cannot proceed - system not healthy")
            return False
        
        # Step 2: ML capabilities
        ml_available = self.show_ml_capabilities()
        
        # Step 3: Solver features
        self.show_solver_features()
        
        # Step 4: Live optimization demo
        optimization_success = self.demo_optimization()
        
        # Step 5: Recommendations
        self.show_recommendations()
        
        # Final summary
        self.print_banner("Demo Summary")
        
        print(f"✅ System Health: Operational")
        print(f"🤖 ML Capabilities: {'Available' if ml_available else 'Limited'}")
        print(f"🎯 Optimization: {'Success' if optimization_success else 'Failed'}")
        
        if ml_available and optimization_success:
            print(f"\n🌟 DEMO SUCCESS: Full ML-enhanced system operational!")
            print(f"🚀 Ready for production deployment")
        elif optimization_success:
            print(f"\n✅ DEMO PARTIAL: System working with fallback algorithms")
            print(f"💡 Consider building ML models for enhanced performance")
        else:
            print(f"\n❌ DEMO FAILED: Issues detected")
            print(f"🔧 Check system configuration and try again")
        
        # Next steps
        print(f"\n📋 Next Steps:")
        print(f"   1. 📖 Review documentation: ALGORITHM_DOCUMENTATION.md")
        print(f"   2. 🤖 ML guide: ML_MODEL_GUIDE.md") 
        print(f"   3. 🧪 Run tests: python test_ml_system.py")
        print(f"   4. 📊 API docs: {self.base_url}/docs")
        
        return optimization_success


def main():
    """Main demo function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Logistics System Demo")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--quick", action="store_true", help="Quick demo mode")
    
    args = parser.parse_args()
    
    demo = LogisticsDemo(args.url)
    
    if args.quick:
        # Quick health check and basic optimization
        print("🚛 Quick ML System Demo")
        print("=" * 40)
        
        if demo.check_system_health():
            print("\n⚡ Running quick optimization test...")
            success = demo.demo_optimization()
            
            if success:
                print("\n✅ Quick demo successful!")
            else:
                print("\n❌ Quick demo failed")
        else:
            print("\n❌ System not available")
    else:
        # Full comprehensive demo
        demo.run_full_demo()


if __name__ == "__main__":
    main()
