import time
import json
import re
import math
import os
import traceback
from z3 import *

class SMT_Solver:
    """
    An enhanced SMT solver for the Min-Max Capacitated Vehicle Routing Problem (MCP/CVRP).
    
    This version incorporates several key improvements:
    1.  **Native Optimization Engine:** Uses Z3's `Optimize` class for more efficient handling
        of the min-max objective, replacing manual search loops.
    2.  **Valid Inequalities:** Adds explicit linking constraints between item assignment
        and arc usage to improve constraint propagation.
    3.  **Heuristic-Driven:** Includes an "auto" mode that combines pre-sorting and a greedy
        heuristic to provide the solver with the best possible starting point.
    """

    def __init__(self, timeout: int = 300):
        """
        Initializes the solver with a timeout and a suite of experimental configurations.
        """
        self.timeout = timeout
        # --- Configurations for experiments ---
        self.configs = [
            # NEW: The "auto" option enables all enhancements for the best performance.
            ("smt_auto",             {"use_greedy": True,  "use_sorting": True}),
            
            # Original configurations for comparison and ablation studies.
            ("smt_base",             {"use_greedy": False, "use_sorting": False}),
            ("smt_greedy",           {"use_greedy": True,  "use_sorting": False}),
            ("smt_sorted",           {"use_greedy": False, "use_sorting": True}),
            # This one is identical to "smt_auto" but kept for legacy/clarity.
            ("smt_greedy_sorted",    {"use_greedy": True,  "use_sorting": True}),
        ]

    # ─────────────────────────── Helper methods ───────────────────────────
    @staticmethod
    def _exactly_one(bools: list) -> BoolRef:
        """
        Ensures exactly one of the booleans in the list is true.
        Uses a Pseudo-Boolean constraint for efficiency.
        """
        return PbEq([(b, 1) for b in bools], 1)
        
    @staticmethod
    def _lex_leq(a: list, b: list) -> BoolRef:
        """
        Adds a lexicographical ordering constraint (a <= b) to break symmetry.
        """
        less = And(Not(a[0]), b[0])
        equal = a[0] == b[0]
        for i in range(1, len(a)):
            less = Or(less, And(equal, Not(a[i]), b[i]))
            equal = And(equal, a[i] == b[i])
        return Or(equal, less)

    # ─────────────────── Greedy Heuristic Pre-Solver ───────────────────
    def _greedy_heuristic(self, m: int, n: int, caps: list, sizes: list, D: list) -> (list, int):
        """
        A cheapest-insertion greedy heuristic to find a feasible, low-cost initial solution.
        This provides a strong initial upper bound for the optimizer.
        """
        depot = n + 1
        routes = [[] for _ in range(m)]
        loads = [0] * m
        unassigned_items = list(range(1, n + 1))
        
        while unassigned_items:
            best_cost_increase, best_insertion = float('inf'), None
            for item_idx, item_id in enumerate(unassigned_items):
                for k in range(m):
                    if loads[k] + sizes[item_id] <= caps[k]:
                        for pos in range(len(routes[k]) + 1):
                            route = routes[k]
                            prev_stop = route[pos-1] if pos > 0 else depot
                            next_stop = route[pos] if pos < len(route) else depot
                            # Cost increase is calculated as (new_arc1 + new_arc2 - old_arc)
                            cost_increase = (D[prev_stop-1][item_id-1] + D[item_id-1][next_stop-1] - D[prev_stop-1][next_stop-1])
                            if cost_increase < best_cost_increase:
                                best_cost_increase = cost_increase
                                best_insertion = (item_idx, k, pos)
            
            if best_insertion:
                item_idx, k, pos = best_insertion
                item_to_add = unassigned_items.pop(item_idx)
                routes[k].insert(pos, item_to_add)
                loads[k] += sizes[item_to_add]
            else:
                # No feasible insertion found for any remaining item
                return None, float('inf')

        max_dist = 0
        for k in range(m):
            dist, last_stop = 0, depot
            for item_id in routes[k]:
                dist += D[last_stop-1][item_id-1]
                last_stop = item_id
            dist += D[last_stop-1][depot-1]
            if dist > max_dist:
                max_dist = dist
                
        return routes, max_dist

    # ─────────────────────────── Core Solver Logic ───────────────────────────

    def _parse_instance(self, file_path: str):
        """ Parses the instance file to extract problem data. """
        with open(file_path, 'r') as f: content = f.read()
        tokens = re.split(r'\s+', content.strip())
        m, n = int(tokens.pop(0)), int(tokens.pop(0))
        caps = [int(tokens.pop(0)) for _ in range(m)]
        sizes = [0] + [int(tokens.pop(0)) for _ in range(n)] # 1-based indexing for items
        D_flat = [int(t) for t in tokens if t]
        D = [D_flat[i*(n+1):(i+1)*(n+1)] for i in range(n+1)]
        return m, n, caps, sizes, D

    def _build_model(self, m: int, n: int, caps: list, sizes: list, D: list):
        """
        Builds the Z3 optimization model for the CVRP.
        
        Returns:
            - optimizer: The Z3 Optimize object.
            - rho: The objective variable to be minimized.
            - x: Variables for item-to-vehicle assignment.
            - y: Variables for arc traversal.
        """
        optimizer = Optimize()
        depot_idx = n  # 0-based index for the depot in an n+1 matrix

        # --- Decision Variables ---
        # x[k][j]: item j (0 to n-1) is assigned to vehicle k
        x = [[Bool(f"x_{k}_{j}") for j in range(n)] for k in range(m)]
        # y[k][i][j]: vehicle k travels from node i to node j (0 to n)
        y = [[[Bool(f"y_{k}_{i}_{j}") for j in range(n + 1)] for i in range(n + 1)] for k in range(m)]
        # u[k][j]: position of item j in the tour of vehicle k (for Subtour Elimination)
        u = [[Int(f"u_{k}_{j}") for j in range(n)] for k in range(m)]
        # rho: the maximum distance over all routes (our objective)
        rho = Int("rho")

        # --- Constraints ---
        # 1. Capacity Constraints: The total size of items on a vehicle must not exceed its capacity.
        for k in range(m):
            optimizer.add(Sum([If(x[k][j], sizes[j+1], 0) for j in range(n)]) <= caps[k])

        # 2. Assignment Constraints: Each item must be assigned to exactly one vehicle.
        for j in range(n):
            optimizer.add(self._exactly_one([x[k][j] for k in range(m)]))

        # 3. Flow Conservation Constraints
        for k in range(m):
            # Each vehicle can leave the depot at most once.
            optimizer.add(PbLe([(y[k][depot_idx][j], 1) for j in range(n)], 1))
            
            # For each item j, if it's on route k, one arc must enter and one must leave.
            for j in range(n):
                num_incoming = Sum([If(y[k][i][j], 1, 0) for i in range(n + 1) if i != j])
                num_outgoing = Sum([If(y[k][j][i], 1, 0) for i in range(n + 1) if i != j])
                # If x[k][j] is true, then num_incoming/outgoing must be 1. Otherwise, 0.
                optimizer.add(If(x[k][j], num_incoming == 1, num_incoming == 0))
                optimizer.add(If(x[k][j], num_outgoing == 1, num_outgoing == 0))

            # The number of vehicles leaving the depot must equal the number returning.
            optimizer.add(Sum([If(y[k][depot_idx][j], 1, 0) for j in range(n)]) == 
                          Sum([If(y[k][j][depot_idx], 1, 0) for j in range(n)]))

        # 4. Miller-Tucker-Zemlin (MTZ) Subtour Elimination
        for k in range(m):
            for j in range(n):
                optimizer.add(Implies(x[k][j], And(u[k][j] >= 1, u[k][j] <= n)))
                optimizer.add(Implies(Not(x[k][j]), u[k][j] == 0))
                # If an arc from depot to j exists, j is the first stop.
                optimizer.add(Implies(y[k][depot_idx][j], u[k][j] == 1))
                # If an arc from i to j exists, u[j] must be u[i] + 1.
                for i in range(n):
                    if i != j:
                        optimizer.add(Implies(y[k][i][j], u[k][j] == u[k][i] + 1))
        
        # 5. ENHANCEMENT: Add Valid Inequalities to link x and y variables
        # If an arc (i, j) is used by vehicle k, then both items i and j must be assigned to k.
        # This helps the solver by improving constraint propagation.
        for k in range(m):
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    optimizer.add(Implies(y[k][i][j], And(x[k][i], x[k][j])))

        # 6. Objective Function: rho must be greater than or equal to the distance of each route.
        for k in range(m):
            dist = Sum([If(y[k][i][j], D[i][j], 0) for i in range(n + 1) for j in range(n + 1)])
            optimizer.add(rho >= dist)
            
        # 7. Symmetry Breaking: For vehicles with identical capacities, enforce a lexicographical order on their assignments.
        for k in range(m - 1):
            if caps[k] == caps[k+1]:
                optimizer.add(self._lex_leq([x[k][j] for j in range(n)], [x[k+1][j] for j in range(n)]))
        
        # 8. Set the objective for the optimizer
        optimizer.minimize(rho)
        
        return optimizer, rho, x, y

    def _solve_with_optimizer(self, optimizer: Optimize, rho: ArithRef, timeout_ms: int, initial_upper_bound=None):
        """
        Uses the Optimize object to find a solution.

        Args:
            optimizer: The Z3 Optimize object with the model loaded.
            rho: The objective function variable.
            timeout_ms: The time limit for the solver in milliseconds.
            initial_upper_bound: An optional upper bound from a heuristic.

        Returns:
            A tuple (model, is_optimal).
        """
        start_time = time.time()
        
        # Apply the upper bound from the greedy heuristic, if available
        if initial_upper_bound is not None and initial_upper_bound != float('inf'):
            print(f"Applying Greedy Heuristic UB: {initial_upper_bound}")
            optimizer.add(rho <= initial_upper_bound)

        optimizer.set("timeout", max(1, timeout_ms))
        
        status = optimizer.check()
        
        model, is_optimal = None, False
        if status == sat:
            # A satisfying assignment was found. If it didn't time out, it's optimal.
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms < timeout_ms:
                is_optimal = True
            model = optimizer.model()
            print(f"Solution found. Status: SAT. Optimal: {is_optimal}")
        elif status == unknown:
            # Solver timed out. It may have found a non-optimal solution.
            print("Solver timed out (unknown). A sub-optimal solution may be available.")
            try:
                model = optimizer.model() # Z3 can often provide the best model found so far
            except Z3Exception:
                model = None
            is_optimal = False
        else: # unsat
            print("Problem is UNSAT (no solution exists).")
            is_optimal = True # Technically, it's proven that no solution exists.
        
        return model, is_optimal

    def _extract_paths(self, model: ModelRef, y_vars, m: int, n: int, reverse_map=None):
        """ Extracts the solution paths from a Z3 model. """
        if not model: return [[] for _ in range(m)]
        
        depot_idx_0based = n
        if reverse_map is None:
            # If no sorting was used, the map is a direct 1-to-1 mapping.
            # model indices (0..n-1) -> problem item IDs (1..n)
            reverse_map = {i: i + 1 for i in range(n)}
        
        all_routes = []
        for k in range(m):
            arcs = []
            for i in range(n + 1):
                for j in range(n + 1):
                    if is_true(model.eval(y_vars[k][i][j], model_completion=True)):
                        arcs.append((i, j))
            
            if not arcs:
                all_routes.append([]); continue

            tour_map = {i: j for i, j in arcs}
            if depot_idx_0based not in tour_map: # No tour starts from the depot
                all_routes.append([]); continue

            # Reconstruct the path by following the arcs from the depot
            current_node = tour_map[depot_idx_0based]
            route = []
            visited_count = 0 # Safety break for malformed tours
            while current_node != depot_idx_0based and visited_count < n:
                route.append(reverse_map[current_node])
                current_node = tour_map.get(current_node, depot_idx_0based)
                visited_count += 1
            all_routes.append(route)
            
        return all_routes

    def solve(self, instance_file: str, output_dir: str):
        """
        Main entry point to solve an instance file with all configured strategies.
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(instance_file))[0]
        inst_id = re.search(r'\d+', base_name).group(0)
        results = {}

        for name, config in self.configs:
            print(f"----- [SMT Solver Enhanced] -----")
            print(f"Running approach: {name} for instance: {inst_id}")
            
            total_start_time = time.time()
            m, n, caps, original_sizes, original_D = self._parse_instance(instance_file)
            
            use_greedy = config.get("use_greedy", False)
            use_sorting = config.get("use_sorting", False)
            timeout_ms = self.timeout * 1000
            
            sizes_to_solve, D_to_solve = original_sizes, original_D
            # reverse_item_map is crucial for correctly mapping sorted indices back to original item IDs
            reverse_item_map = {i: i + 1 for i in range(n)}

            if use_sorting and n > 0:
                print("Applying item pre-sorting by remoteness...")
                original_indices_0based = list(range(n))
                # Sort items by their round-trip distance from the depot, descending
                original_indices_0based.sort(key=lambda j: original_D[j][n] + original_D[n][j], reverse=True)
                
                sorted_to_original_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(original_indices_0based)}
                reverse_item_map = {new_idx: sorted_to_original_map[new_idx] + 1 for new_idx in range(n)}

                # Re-build sizes and distance matrix according to the new sorted order
                sizes_to_solve = [0] + [original_sizes[sorted_to_original_map[i] + 1] for i in range(n)]
                new_D = [[0] * (n + 1) for _ in range(n + 1)]
                for i in range(n):
                    for j in range(n):
                        orig_i, orig_j = sorted_to_original_map[i], sorted_to_original_map[j]
                        new_D[i][j] = original_D[orig_i][orig_j]
                for i in range(n):
                    orig_i = sorted_to_original_map[i]
                    new_D[i][n], new_D[n][i] = original_D[orig_i][n], original_D[n][orig_i]
                new_D[n][n] = original_D[n][n]
                D_to_solve = new_D

            model, is_optimal = None, False
            
            try:
                presolve_start_time = time.time()
                
                optimizer, rho, x, y = self._build_model(m, n, caps, sizes_to_solve, D_to_solve)
                
                greedy_ub = None
                if use_greedy:
                    _, greedy_ub = self._greedy_heuristic(m, n, caps, sizes_to_solve, D_to_solve)
                
                presolve_time_ms = (time.time() - presolve_start_time) * 1000
                remaining_timeout_ms = timeout_ms - int(presolve_time_ms)
                
                if remaining_timeout_ms > 0:
                    model, is_optimal = self._solve_with_optimizer(optimizer, rho, remaining_timeout_ms, initial_upper_bound=greedy_ub)

            except Exception as e:
                print(f"[SMT Solver] ERROR on instance {base_name} with {name}: {e}")
                traceback.print_exc()

            solve_time_sec = time.time() - total_start_time
            final_obj, final_sol = -1, [[] for _ in range(m)]

            if model:
                final_obj = model.eval(rho).as_long()
                final_sol = self._extract_paths(model, y, m, n, reverse_map=reverse_item_map)
            
            if solve_time_sec >= self.timeout:
                is_optimal = False

            results[name] = {
                "time": min(self.timeout, round(solve_time_sec, 2)),
                "optimal": is_optimal,
                "obj": int(final_obj),
                "sol": final_sol
            }

        out_path = os.path.join(output_dir, f"{inst_id}-enhanced.json")
        with open(out_path, "w") as jf:
            json.dump(results, jf, indent=4)
        print(f"Enhanced SMT results for instance {inst_id} written to {out_path}\n")


# Example of how to run the solver
if __name__ == '__main__':
    # This block allows you to run this file directly for testing.
    # To use it with your `unified_solver.py`, you would still import the class.
    
    # Create a dummy instance file for testing
    instance_content = """
    2 4
    10 10
    2 3 4 5
    0 10 12 8 6
    10 0 2 15 10
    12 2 0 4 8
    8 15 4 0 5
    6 10 8 5 0
    """
    instance_dir = "instances"
    output_dir = "results"
    os.makedirs(instance_dir, exist_ok=True)
    instance_path = os.path.join(instance_dir, "instance_1.txt")
    with open(instance_path, "w") as f:
        f.write(instance_content.strip())

    # Initialize and run the solver
    # Using a short timeout for this example run
    solver = SMT_Solver(timeout=60) 
    solver.solve(instance_path, output_dir)