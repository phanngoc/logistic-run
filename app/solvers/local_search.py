"""Local Search solver với 2-opt và relocate moves"""

import time
import random
from typing import List, Dict, Optional, Tuple
import logging

from .greedy_solver import GreedySolver
from .base_solver import SolverResult
from ..schemas.orders import Order
from ..schemas.fleet import Truck

logger = logging.getLogger(__name__)


class LocalSearchSolver(GreedySolver):
    """Local Search solver kế thừa từ GreedySolver và thêm local search moves"""
    
    def __init__(self, cost_config, weights):
        super().__init__(cost_config, weights)
        self.algorithm_name = "greedy_with_local_search"
    
    async def solve(self, 
                   orders: List[Order], 
                   trucks: List[Truck],
                   max_iterations: Optional[int] = None,
                   time_limit_seconds: Optional[int] = None) -> SolverResult:
        """
        Solve bằng Greedy + Local Search
        
        1. Chạy greedy insertion để có initial solution
        2. Cải thiện bằng local search moves (2-opt, relocate)
        """
        
        # Store orders for later use
        self.all_orders = orders
        
        # Get initial solution from greedy
        initial_result = await super().solve(orders, trucks, max_iterations, time_limit_seconds)
        
        if not initial_result.routes:
            return initial_result
        
        start_time = time.time()
        
        # Convert routes back to solution format for local search
        current_solution = self._routes_to_solution(initial_result.routes)
        current_cost = self._calculate_solution_cost(initial_result.routes)
        
        # Local search parameters
        max_ls_iterations = max_iterations or 100
        no_improvement_limit = 20
        no_improvement_count = 0
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        iteration = initial_result.iterations
        
        logger.info(f"Starting local search from cost: {current_cost:.2f}")
        
        try:
            while (iteration < max_ls_iterations and 
                   no_improvement_count < no_improvement_limit):
                
                # Check time limit
                if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
                    logger.info(f"Time limit reached in local search after {iteration} iterations")
                    break
                
                iteration += 1
                improvement_found = False
                
                # Try different neighborhood moves
                moves = [
                    self._try_relocate_move,
                    self._try_swap_move,
                    self._try_2opt_move
                ]
                
                random.shuffle(moves)  # Randomize move order
                
                for move_func in moves:
                    improved_solution, improved_cost = await move_func(current_solution, current_cost)
                    
                    if improved_cost < current_cost - 0.01:  # Improvement threshold
                        current_solution = improved_solution
                        current_cost = improved_cost
                        improvement_found = True
                        no_improvement_count = 0
                        
                        if improved_cost < best_cost:
                            best_solution = improved_solution.copy()
                            best_cost = improved_cost
                        
                        logger.debug(f"Iteration {iteration}: improved to {improved_cost:.2f}")
                        break
                
                if not improvement_found:
                    no_improvement_count += 1
        
        except Exception as e:
            logger.error(f"Local search error: {e}")
            # Return initial solution if local search fails
            return initial_result
        
        # Convert best solution back to routes
        final_routes = await self._solution_to_routes(best_solution, trucks)
        
        # Create final result
        result = SolverResult()
        result.algorithm = self.algorithm_name
        result.routes = final_routes
        result.unserved_orders = initial_result.unserved_orders  # Keep same unserved orders
        result.solve_time_seconds = time.time() - start_time + initial_result.solve_time_seconds
        result.iterations = iteration
        result.metadata = {
            "initial_cost": current_cost,
            "final_cost": best_cost,
            "improvement": current_cost - best_cost,
            "total_orders": len(orders),
            "served_orders": len(orders) - len(result.unserved_orders),
            "trucks_used": len([r for r in result.routes if r.order_ids])
        }
        
        logger.info(f"Local search completed. Initial: {current_cost:.2f}, Final: {best_cost:.2f}, Improvement: {current_cost - best_cost:.2f}")
        
        return result
    
    def _routes_to_solution(self, routes: List) -> Dict[str, List[Order]]:
        """Convert routes back to solution dict format"""
        solution = {}
        
        for route in routes:
            truck_id = route.truck_id
            
            # Rebuild order list from stops
            orders = []
            processed_orders = set()
            
            for stop in route.stops:
                if stop.order_id and stop.order_id not in processed_orders and stop.stop_type == "pickup":
                    # Find the corresponding order
                    order = next((o for o in self.all_orders if o.order_id == stop.order_id), None)
                    if order:
                        orders.append(order)
                        processed_orders.add(stop.order_id)
            
            solution[truck_id] = orders
        
        return solution
    
    def _calculate_solution_cost(self, routes: List) -> float:
        """Calculate total cost of solution"""
        return sum(route.cost_breakdown.total_cost for route in routes)
    
    async def _try_relocate_move(self, solution: Dict[str, List[Order]], current_cost: float) -> Tuple[Dict[str, List[Order]], float]:
        """Try relocating an order from one truck to another"""
        
        best_solution = solution
        best_cost = current_cost
        
        # Get all trucks with orders
        trucks_with_orders = [(truck_id, orders) for truck_id, orders in solution.items() if orders]
        
        if len(trucks_with_orders) < 2:
            return best_solution, best_cost
        
        # Try relocating orders
        for from_truck_id, from_orders in trucks_with_orders:
            for order_idx, order in enumerate(from_orders):
                for to_truck_id, to_orders in solution.items():
                    if from_truck_id == to_truck_id:
                        continue
                    
                    # Check if target truck can handle this order
                    to_truck = next((t for t in self.trucks if t.truck_id == to_truck_id), None)
                    if not to_truck or not self._can_truck_handle_order(to_truck, order, to_orders):
                        continue
                    
                    # Create new solution with relocated order
                    new_solution = self._copy_solution(solution)
                    relocated_order = new_solution[from_truck_id].pop(order_idx)
                    new_solution[to_truck_id].append(relocated_order)
                    
                    # Calculate new cost
                    new_cost = await self._evaluate_solution_cost(new_solution)
                    
                    if new_cost < best_cost:
                        best_solution = new_solution
                        best_cost = new_cost
        
        return best_solution, best_cost
    
    async def _try_swap_move(self, solution: Dict[str, List[Order]], current_cost: float) -> Tuple[Dict[str, List[Order]], float]:
        """Try swapping orders between trucks"""
        
        best_solution = solution
        best_cost = current_cost
        
        trucks_with_orders = [(truck_id, orders) for truck_id, orders in solution.items() if orders]
        
        if len(trucks_with_orders) < 2:
            return best_solution, best_cost
        
        # Try swapping orders between trucks
        for i, (truck1_id, orders1) in enumerate(trucks_with_orders):
            for j, (truck2_id, orders2) in enumerate(trucks_with_orders[i+1:], i+1):
                
                for order1_idx, order1 in enumerate(orders1):
                    for order2_idx, order2 in enumerate(orders2):
                        
                        # Check compatibility
                        truck1 = next((t for t in self.trucks if t.truck_id == truck1_id), None)
                        truck2 = next((t for t in self.trucks if t.truck_id == truck2_id), None)
                        
                        if (not truck1 or not truck2 or 
                            not self._can_truck_handle_order(truck1, order2, [o for k, o in enumerate(orders1) if k != order1_idx]) or
                            not self._can_truck_handle_order(truck2, order1, [o for k, o in enumerate(orders2) if k != order2_idx])):
                            continue
                        
                        # Create new solution with swapped orders
                        new_solution = self._copy_solution(solution)
                        new_solution[truck1_id][order1_idx] = order2
                        new_solution[truck2_id][order2_idx] = order1
                        
                        # Calculate new cost
                        new_cost = await self._evaluate_solution_cost(new_solution)
                        
                        if new_cost < best_cost:
                            best_solution = new_solution
                            best_cost = new_cost
        
        return best_solution, best_cost
    
    async def _try_2opt_move(self, solution: Dict[str, List[Order]], current_cost: float) -> Tuple[Dict[str, List[Order]], float]:
        """Try 2-opt moves within each truck route"""
        
        best_solution = solution
        best_cost = current_cost
        
        # Apply 2-opt to each truck route
        for truck_id, orders in solution.items():
            if len(orders) < 3:  # Need at least 3 orders for 2-opt
                continue
            
            # Try all possible 2-opt moves
            for i in range(len(orders) - 1):
                for j in range(i + 2, len(orders) + 1):
                    # Create new order sequence with 2-opt swap
                    new_orders = orders[:i] + orders[i:j][::-1] + orders[j:]
                    
                    # Create new solution
                    new_solution = self._copy_solution(solution)
                    new_solution[truck_id] = new_orders
                    
                    # Calculate new cost
                    new_cost = await self._evaluate_solution_cost(new_solution)
                    
                    if new_cost < best_cost:
                        best_solution = new_solution
                        best_cost = new_cost
        
        return best_solution, best_cost
    
    def _can_truck_handle_order(self, truck: Truck, order: Order, existing_orders: List[Order]) -> bool:
        """Check if truck can handle additional order"""
        # Check container size
        if order.container_size not in truck.allowed_sizes:
            return False
        
        # Check capacity
        if len(existing_orders) >= truck.max_orders_per_day:
            return False
        
        return True
    
    def _copy_solution(self, solution: Dict[str, List[Order]]) -> Dict[str, List[Order]]:
        """Deep copy solution"""
        return {truck_id: orders.copy() for truck_id, orders in solution.items()}
    
    async def _evaluate_solution_cost(self, solution: Dict[str, List[Order]]) -> float:
        """Evaluate total cost of solution"""
        total_cost = 0.0
        
        for truck_id, orders in solution.items():
            if not orders:
                continue
            
            truck = next((t for t in self.trucks if t.truck_id == truck_id), None)
            if not truck:
                continue
            
            route_cost = await self._calculate_route_cost(truck, orders)
            total_cost += route_cost.total_cost
        
        return total_cost
