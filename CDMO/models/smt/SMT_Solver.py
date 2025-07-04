# import time
# import json
# import re
# import math
# import os
# import traceback
# from z3 import *

# class SMT_Solver:
#     """
#     A SMT solver for the MCP problem.
    
#     Some features of the model:
#     1.  **Uses the native Optimization Engine:** Uses Z3's `Optimize` class for more efficient handling
#         of the min-max objective, replacing manual search loops.
#     2.  **Valid Inequalities:** Adds explicit linking constraints between item assignment
#         and arc usage to improve constraint propagation.
#     3.  **Heuristic-Driven:** Includes an "auto" mode that combines pre-sorting and a powerful
#         local search heuristic to provide the solver with an excellent starting point.
#     """

#     def __init__(self, timeout: int = 300):
#         """
#         Initializes the solver with a timeout and a suite of experimental configurations.
#         """
#         self.timeout = timeout
#         # --- Configurations for experiments ---
#         self.configs = [
#             ("smt_auto",             {"use_greedy": True,  "use_sorting": True}),
#             ("smt_base",             {"use_greedy": False, "use_sorting": False}),
#             ("smt_greedy",           {"use_greedy": True,  "use_sorting": False}),
#             ("smt_sorted",           {"use_greedy": False, "use_sorting": True}),
#             ("smt_greedy_sorted",    {"use_greedy": True,  "use_sorting": True}),
#         ]

#     # ─────────────────────────── Helper methods ───────────────────────────
#     @staticmethod
#     def _exactly_one(bools: list) -> BoolRef:
#         """Ensures exactly one of the booleans in the list is true."""
#         return PbEq([(b, 1) for b in bools], 1)
        
#     @staticmethod
#     def _lex_leq(a: list, b: list) -> BoolRef:
#         """Adds a lexicographical ordering constraint (a <= b) to break symmetry."""
#         less = And(Not(a[0]), b[0])
#         equal = a[0] == b[0]
#         for i in range(1, len(a)):
#             less = Or(less, And(equal, Not(a[i]), b[i]))
#             equal = And(equal, a[i] == b[i])
#         return Or(equal, less)

#     # ─────────────────── Advanced Heuristic Pre-Solver ───────────────────
    
#     def _cheapest_insertion_heuristic(self, m: int, n: int, caps: list, sizes: list, D: list) -> (list, int):
#         """
#         Stage 1: Constructs an initial feasible solution using cheapest insertion.
#         Item IDs are 1-based, but all distance matrix access is 0-based.
#         """
#         depot_idx = n # 0-based index for distance matrix
#         routes = [[] for _ in range(m)]
#         loads = [0] * m
#         unassigned_items = list(range(1, n + 1))
        
#         while unassigned_items:
#             best_cost_increase, best_insertion = float('inf'), None
#             for item_idx_in_list, item_id in enumerate(unassigned_items):
#                 for k in range(m):
#                     if loads[k] + sizes[item_id] <= caps[k]:
#                         for pos in range(len(routes[k]) + 1):
#                             current_route = routes[k]
#                             # Convert 1-based item IDs to 0-based matrix indices
#                             prev_node_idx = depot_idx if pos == 0 else current_route[pos-1] - 1
#                             next_node_idx = depot_idx if pos == len(current_route) else current_route[pos] - 1
#                             item_d_idx = item_id - 1
                            
#                             cost_increase = (D[prev_node_idx][item_d_idx] + D[item_d_idx][next_node_idx] - D[prev_node_idx][next_node_idx])
                            
#                             if cost_increase < best_cost_increase:
#                                 best_cost_increase = cost_increase
#                                 best_insertion = (item_idx_in_list, k, pos)
        
#             if best_insertion:
#                 item_idx_in_list, k, pos = best_insertion
#                 item_to_add = unassigned_items.pop(item_idx_in_list)
#                 routes[k].insert(pos, item_to_add)
#                 loads[k] += sizes[item_to_add]
#             else:
#                 return None, float('inf')
        
#         return routes, 0 # Objective is calculated by the calling function

#     def _local_search_heuristic(self, m: int, n: int, caps: list, sizes: list, D: list) -> (list, int):
#         """
#         A powerful two-stage heuristic:
#         1. Constructs a solution using cheapest insertion.
#         2. Improves the solution using a 2-Opt local search.
#         """
#         # --- Stage 1: Construction ---
#         routes, _ = self._cheapest_insertion_heuristic(m, n, caps, sizes, D)
#         if routes is None:
#             return None, float('inf')

#         # --- Stage 2: Improvement (2-Opt) ---
#         depot_idx = n
#         improvement_found = True
#         while improvement_found:
#             improvement_found = False
#             for k in range(m):
#                 if len(routes[k]) < 2: continue

#                 # Create a full tour with 0-based depot/customer indices for calculation
#                 tour = [depot_idx] + [node - 1 for node in routes[k]] + [depot_idx]
                
#                 # Using "best improvement" strategy for 2-opt
#                 best_delta = 0
#                 best_move = None
#                 for i in range(len(tour) - 3):
#                     for j in range(i + 2, len(tour) - 1):
#                         node_i, node_i1 = tour[i], tour[i+1]
#                         node_j, node_j1 = tour[j], tour[j+1]
                        
#                         original_cost = D[node_i][node_i1] + D[node_j][node_j1]
#                         new_cost = D[node_i][node_j] + D[node_i1][node_j1]
#                         delta = new_cost - original_cost

#                         if delta < best_delta:
#                             best_delta = delta
#                             best_move = (i, j)

#                 if best_move:
#                     i, j = best_move
#                     tour[i+1:j+1] = tour[i+1:j+1][::-1] # Reverse segment to apply swap
#                     routes[k] = [node + 1 for node in tour[1:-1]] # Update route (1-based)
#                     improvement_found = True

#         # --- Stage 3: Recalculate Final Objective ---
#         max_dist = 0
#         for k in range(m):
#             dist, last_node_idx = 0, depot_idx
#             for item_id in routes[k]:
#                 item_idx = item_id - 1
#                 dist += D[last_node_idx][item_idx]
#                 last_node_idx = item_idx
#             dist += D[last_node_idx][depot_idx]
#             if dist > max_dist:
#                 max_dist = dist
                
#         print(f"Advanced heuristic found initial solution with max route: {max_dist}")
#         return routes, max_dist

#     # ─────────────────────────── Core Solver Logic ───────────────────────────

#     def _parse_instance(self, file_path: str):
#         with open(file_path, 'r') as f: content = f.read()
#         tokens = re.split(r'\s+', content.strip())
#         m, n = int(tokens.pop(0)), int(tokens.pop(0))
#         caps = [int(tokens.pop(0)) for _ in range(m)]
#         sizes = [0] + [int(tokens.pop(0)) for _ in range(n)]
#         D_flat = [int(t) for t in tokens if t]
#         D = [D_flat[i*(n+1):(i+1)*(n+1)] for i in range(n+1)]
#         return m, n, caps, sizes, D

#     def _build_model(self, m: int, n: int, caps: list, sizes: list, D: list):
#         optimizer = Optimize()
#         depot_idx = n
#         x = [[Bool(f"x_{k}_{j}") for j in range(n)] for k in range(m)]
#         y = [[[Bool(f"y_{k}_{i}_{j}") for j in range(n + 1)] for i in range(n + 1)] for k in range(m)]
#         u = [[Int(f"u_{k}_{j}") for j in range(n)] for k in range(m)]
#         rho = Int("rho")

#         for k in range(m):
#             optimizer.add(Sum([If(x[k][j], sizes[j+1], 0) for j in range(n)]) <= caps[k])
#         for j in range(n):
#             optimizer.add(self._exactly_one([x[k][j] for k in range(m)]))
#         for k in range(m):
#             optimizer.add(PbLe([(y[k][depot_idx][j], 1) for j in range(n)], 1))
#             for j in range(n):
#                 num_incoming = Sum([If(y[k][i][j], 1, 0) for i in range(n + 1) if i != j])
#                 num_outgoing = Sum([If(y[k][j][i], 1, 0) for i in range(n + 1) if i != j])
#                 optimizer.add(If(x[k][j], num_incoming == 1, num_incoming == 0))
#                 optimizer.add(If(x[k][j], num_outgoing == 1, num_outgoing == 0))
#             optimizer.add(Sum([If(y[k][depot_idx][j], 1, 0) for j in range(n)]) == Sum([If(y[k][j][depot_idx], 1, 0) for j in range(n)]))
#         for k in range(m):
#             for j in range(n):
#                 optimizer.add(Implies(x[k][j], And(u[k][j] >= 1, u[k][j] <= n)))
#                 optimizer.add(Implies(Not(x[k][j]), u[k][j] == 0))
#                 optimizer.add(Implies(y[k][depot_idx][j], u[k][j] == 1))
#                 for i in range(n):
#                     if i != j:
#                         optimizer.add(Implies(y[k][i][j], u[k][j] == u[k][i] + 1))
#         for k in range(m):
#             for i in range(n):
#                 for j in range(n):
#                     if i != j:
#                         optimizer.add(Implies(y[k][i][j], And(x[k][i], x[k][j])))
#         for k in range(m):
#             dist = Sum([If(y[k][i][j], D[i][j], 0) for i in range(n + 1) for j in range(n + 1)])
#             optimizer.add(rho >= dist)
#         for k in range(m - 1):
#             if caps[k] == caps[k+1]:
#                 optimizer.add(self._lex_leq([x[k][j] for j in range(n)], [x[k+1][j] for j in range(n)]))
#         optimizer.minimize(rho)
#         return optimizer, rho, x, y

#     def _solve_with_optimizer(self, optimizer: Optimize, rho: ArithRef, timeout_ms: int, initial_upper_bound=None):
#         start_time = time.time()
#         if initial_upper_bound is not None and initial_upper_bound != float('inf'):
#             print(f"Applying Advanced Heuristic UB: {initial_upper_bound}")
#             optimizer.add(rho <= initial_upper_bound)
#         optimizer.set("timeout", max(1, timeout_ms))
#         status = optimizer.check()
#         model, is_optimal = None, False
#         if status == sat:
#             elapsed_ms = (time.time() - start_time) * 1000
#             if elapsed_ms < timeout_ms:
#                 is_optimal = True
#             model = optimizer.model()
#             print(f"Solution found. Status: SAT. Optimal: {is_optimal}")
#         elif status == unknown:
#             print("Solver timed out (unknown). A sub-optimal solution may be available.")
#             try:
#                 model = optimizer.model()
#             except Z3Exception:
#                 model = None
#             is_optimal = False
#         else: # unsat
#             print("Problem is UNSAT (no solution exists).")
#             is_optimal = True
#         return model, is_optimal

#     def _extract_paths(self, model: ModelRef, y_vars, m: int, n: int, reverse_map=None):
#         if not model: return [[] for _ in range(m)]
#         depot_idx_0based = n
#         if reverse_map is None:
#             reverse_map = {i: i + 1 for i in range(n)}
#         all_routes = []
#         for k in range(m):
#             arcs = []
#             for i in range(n + 1):
#                 for j in range(n + 1):
#                     if is_true(model.eval(y_vars[k][i][j], model_completion=True)):
#                         arcs.append((i, j))
#             if not arcs:
#                 all_routes.append([]); continue
#             tour_map = {i: j for i, j in arcs}
#             if depot_idx_0based not in tour_map:
#                 all_routes.append([]); continue
#             current_node = tour_map[depot_idx_0based]
#             route = []
#             visited_count = 0
#             while current_node != depot_idx_0based and visited_count < n:
#                 route.append(reverse_map[current_node])
#                 current_node = tour_map.get(current_node, depot_idx_0based)
#                 visited_count += 1
#             all_routes.append(route)
#         return all_routes

#     def solve(self, instance_file: str, output_dir: str):
#         os.makedirs(output_dir, exist_ok=True)
#         base_name = os.path.splitext(os.path.basename(instance_file))[0]
#         inst_id = re.search(r'\d+', base_name).group(0)
#         results = {}

#         for name, config in self.configs:
#             print(f"----- [SMT Solver Enhanced] -----")
#             print(f"Running approach: {name} for instance: {inst_id}")
            
#             total_start_time = time.time()
#             m, n, caps, original_sizes, original_D = self._parse_instance(instance_file)
            
#             use_greedy = config.get("use_greedy", False)
#             use_sorting = config.get("use_sorting", False)
#             timeout_ms = self.timeout * 1000
            
#             sizes_to_solve, D_to_solve = original_sizes, original_D
#             reverse_item_map = {i: i + 1 for i in range(n)}

#             if use_sorting and n > 0:
#                 print("Applying item pre-sorting by remoteness...")
#                 original_indices_0based = sorted(range(n), key=lambda j: original_D[j][n] + original_D[n][j], reverse=True)
#                 sorted_to_original_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(original_indices_0based)}
#                 reverse_item_map = {new_idx: sorted_to_original_map[new_idx] + 1 for new_idx in range(n)}
#                 sizes_to_solve = [0] + [original_sizes[sorted_to_original_map[i] + 1] for i in range(n)]
#                 new_D = [[0] * (n + 1) for _ in range(n + 1)]
#                 for i in range(n):
#                     for j in range(n):
#                         orig_i, orig_j = sorted_to_original_map[i], sorted_to_original_map[j]
#                         new_D[i][j] = original_D[orig_i][orig_j]
#                 for i in range(n):
#                     orig_i = sorted_to_original_map[i]
#                     new_D[i][n], new_D[n][i] = original_D[orig_i][n], original_D[n][orig_i]
#                 new_D[n][n] = original_D[n][n]
#                 D_to_solve = new_D

#             model, is_optimal = None, False
            
#             try:
#                 presolve_start_time = time.time()
                
#                 optimizer, rho, x, y = self._build_model(m, n, caps, sizes_to_solve, D_to_solve)
                
#                 greedy_ub = None
#                 if use_greedy:
#                     # UPDATED TO USE THE ADVANCED HEURISTIC
#                     _, greedy_ub = self._local_search_heuristic(m, n, caps, sizes_to_solve, D_to_solve)
                
#                 presolve_time_ms = (time.time() - presolve_start_time) * 1000
#                 remaining_timeout_ms = timeout_ms - int(presolve_time_ms)
                
#                 if remaining_timeout_ms > 0:
#                     model, is_optimal = self._solve_with_optimizer(optimizer, rho, remaining_timeout_ms, initial_upper_bound=greedy_ub)

#             except Exception as e:
#                 print(f"[SMT Solver] ERROR on instance {base_name} with {name}: {e}")
#                 traceback.print_exc()

#             solve_time_sec = time.time() - total_start_time
#             final_obj, final_sol = -1, [[] for _ in range(m)]

#             if model:
#                 final_obj = model.eval(rho).as_long()
#                 final_sol = self._extract_paths(model, y, m, n, reverse_map=reverse_item_map)
            
#             if solve_time_sec >= self.timeout:
#                 is_optimal = False

#             results[name] = {
#                 "time": min(self.timeout, round(solve_time_sec, 2)),
#                 "optimal": is_optimal,
#                 "obj": int(final_obj),
#                 "sol": final_sol
#             }

#         out_path = os.path.join(output_dir, f"{inst_id}.json")
#         with open(out_path, "w") as jf:
#             json.dump(results, jf, indent=4)
#         print(f"Enhanced SMT results for instance {inst_id} written to {out_path}\n")


import time
import json
import re
import math
import os
import traceback
from z3 import *

class SMT_Solver:
    """
    A SMT solver for the MCP problem.
    
    Some features of the model:
    1.  **Uses the native Optimization Engine:** Uses Z3's `Optimize` class for more efficient handling
        of the min-max objective, replacing manual search loops.
    2.  **Valid Inequalities:** Adds explicit linking constraints between item assignment
        and arc usage to improve constraint propagation.
    3.  **Heuristic-Driven:** Includes an "auto" mode that combines pre-sorting and a powerful
        local search heuristic to provide the solver with an excellent starting point.
    """

    def __init__(self, timeout: int = 300):
        """
        Initializes the solver with a timeout and a suite of experimental configurations.
        """
        self.timeout = timeout
        # --- Configurations for experiments ---
        self.configs = [
            ("smt_auto",            {"use_greedy": True,  "use_sorting": True}),
            ("smt_base",            {"use_greedy": False, "use_sorting": False}),
            ("smt_greedy",          {"use_greedy": True,  "use_sorting": False}),
            ("smt_sorted",          {"use_greedy": False, "use_sorting": True}),
            ("smt_greedy_sorted",   {"use_greedy": True,  "use_sorting": True}),
        ]

    # ─────────────────────────── Helper methods ───────────────────────────
    @staticmethod
    def _exactly_one(bools: list) -> BoolRef:
        """Ensures exactly one of the booleans in the list is true."""
        return PbEq([(b, 1) for b in bools], 1)
        
    @staticmethod
    def _lex_leq(a: list, b: list) -> BoolRef:
        """Adds a lexicographical ordering constraint (a <= b) to break symmetry."""
        less = And(Not(a[0]), b[0])
        equal = a[0] == b[0]
        for i in range(1, len(a)):
            less = Or(less, And(equal, Not(a[i]), b[i]))
            equal = And(equal, a[i] == b[i])
        return Or(equal, less)

    # ─────────────────── Advanced Heuristic Pre-Solver ───────────────────
    
    def _cheapest_insertion_heuristic(self, m: int, n: int, caps: list, sizes: list, D: list) -> (list, int):
        """
        Stage 1: Constructs an initial feasible solution using cheapest insertion.
        Item IDs are 1-based, but all distance matrix access is 0-based.
        """
        depot_idx = n # 0-based index for distance matrix
        routes = [[] for _ in range(m)]
        loads = [0] * m
        unassigned_items = list(range(1, n + 1))
        
        while unassigned_items:
            best_cost_increase, best_insertion = float('inf'), None
            for item_idx_in_list, item_id in enumerate(unassigned_items):
                for k in range(m):
                    if loads[k] + sizes[item_id] <= caps[k]:
                        for pos in range(len(routes[k]) + 1):
                            current_route = routes[k]
                            # Convert 1-based item IDs to 0-based matrix indices
                            prev_node_idx = depot_idx if pos == 0 else current_route[pos-1] - 1
                            next_node_idx = depot_idx if pos == len(current_route) else current_route[pos] - 1
                            item_d_idx = item_id - 1
                            
                            cost_increase = (D[prev_node_idx][item_d_idx] + D[item_d_idx][next_node_idx] - D[prev_node_idx][next_node_idx])
                            
                            if cost_increase < best_cost_increase:
                                best_cost_increase = cost_increase
                                best_insertion = (item_idx_in_list, k, pos)
            
            if best_insertion:
                item_idx_in_list, k, pos = best_insertion
                item_to_add = unassigned_items.pop(item_idx_in_list)
                routes[k].insert(pos, item_to_add)
                loads[k] += sizes[item_to_add]
            else:
                # This item cannot be assigned to any courier
                return None, float('inf')
        
        return routes, 0 # Objective is calculated by the calling function

    def _local_search_heuristic(self, m: int, n: int, caps: list, sizes: list, D: list) -> (list, int):
        """
        A powerful two-stage heuristic:
        1. Constructs a solution using cheapest insertion.
        2. Improves the solution using a 2-Opt local search.
        """
        # --- Stage 1: Construction ---
        routes, _ = self._cheapest_insertion_heuristic(m, n, caps, sizes, D)
        if routes is None:
            return None, float('inf')

        # --- Stage 2: Improvement (2-Opt) ---
        depot_idx = n
        improvement_found = True
        while improvement_found:
            improvement_found = False
            for k in range(m):
                if len(routes[k]) < 2: continue

                # Create a full tour with 0-based depot/customer indices for calculation
                tour = [depot_idx] + [node - 1 for node in routes[k]] + [depot_idx]
                
                # Using "best improvement" strategy for 2-opt
                best_delta = 0
                best_move = None
                for i in range(len(tour) - 3):
                    for j in range(i + 2, len(tour) - 1):
                        node_i, node_i1 = tour[i], tour[i+1]
                        node_j, node_j1 = tour[j], tour[j+1]
                        
                        original_cost = D[node_i][node_i1] + D[node_j][node_j1]
                        new_cost = D[node_i][node_j] + D[node_i1][node_j1]
                        delta = new_cost - original_cost

                        if delta < best_delta:
                            best_delta = delta
                            best_move = (i, j)

                if best_move:
                    i, j = best_move
                    tour[i+1:j+1] = tour[i+1:j+1][::-1] # Reverse segment to apply swap
                    routes[k] = [node + 1 for node in tour[1:-1]] # Update route (1-based)
                    improvement_found = True

        # --- Stage 3: Recalculate Final Objective ---
        max_dist = self._recalculate_objective(m, n, D, routes)
                
        print(f"Advanced heuristic found initial solution with max route: {max_dist}")
        return routes, max_dist

    # ─────────────────────────── Core Solver Logic ───────────────────────────

    def _parse_instance(self, file_path: str):
        with open(file_path, 'r') as f: content = f.read()
        tokens = re.split(r'\s+', content.strip())
        m, n = int(tokens.pop(0)), int(tokens.pop(0))
        caps = [int(tokens.pop(0)) for _ in range(m)]
        sizes = [0] + [int(tokens.pop(0)) for _ in range(n)] # 1-based indexing for sizes
        D_flat = [int(t) for t in tokens if t]
        D = [D_flat[i*(n+1):(i+1)*(n+1)] for i in range(n+1)]
        return m, n, caps, sizes, D

    def _build_model(self, m: int, n: int, caps: list, sizes: list, D: list):
        optimizer = Optimize()
        depot_idx = n
        x = [[Bool(f"x_{k}_{j}") for j in range(n)] for k in range(m)]
        y = [[[Bool(f"y_{k}_{i}_{j}") for j in range(n + 1)] for i in range(n + 1)] for k in range(m)]
        u = [[Int(f"u_{k}_{j}") for j in range(n)] for k in range(m)]
        rho = Int("rho")

        # Capacity constraints
        for k in range(m):
            optimizer.add(Sum([If(x[k][j], sizes[j+1], 0) for j in range(n)]) <= caps[k])
        
        # Each item assigned to exactly one courier
        for j in range(n):
            optimizer.add(self._exactly_one([x[k][j] for k in range(m)]))

        # Flow conservation constraints
        for k in range(m):
            # Each courier leaves the depot at most once
            optimizer.add(PbLe([(y[k][depot_idx][j], 1) for j in range(n)], 1))
            
            for j in range(n):
                num_incoming = Sum([If(y[k][i][j], 1, 0) for i in range(n + 1) if i != j])
                num_outgoing = Sum([If(y[k][j][i], 1, 0) for i in range(n + 1) if i != j])
                # If item j is assigned to courier k, one arc in, one arc out. Otherwise zero.
                optimizer.add(If(x[k][j], num_incoming == 1, num_incoming == 0))
                optimizer.add(If(x[k][j], num_outgoing == 1, num_outgoing == 0))
            
            # Number of arcs leaving depot equals number of arcs entering depot
            optimizer.add(Sum([If(y[k][depot_idx][j], 1, 0) for j in range(n)]) == Sum([If(y[k][j][depot_idx], 1, 0) for j in range(n)]))

        # Subtour elimination (MTZ)
        for k in range(m):
            for j in range(n):
                optimizer.add(Implies(x[k][j], And(u[k][j] >= 1, u[k][j] <= n)))
                optimizer.add(Implies(Not(x[k][j]), u[k][j] == 0))
                optimizer.add(Implies(y[k][depot_idx][j], u[k][j] == 1))
                for i in range(n):
                    if i != j:
                        optimizer.add(Implies(y[k][i][j], u[k][j] == u[k][i] + 1))
        
        # Linking constraints
        for k in range(m):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        optimizer.add(Implies(y[k][i][j], And(x[k][i], x[k][j])))

        # Objective function constraints
        for k in range(m):
            dist = Sum([If(y[k][i][j], D[i][j], 0) for i in range(n + 1) for j in range(n + 1)])
            optimizer.add(rho >= dist)

        # Symmetry breaking
        for k in range(m - 1):
            if caps[k] == caps[k+1]:
                optimizer.add(self._lex_leq([x[k][j] for j in range(n)], [x[k+1][j] for j in range(n)]))
        
        optimizer.minimize(rho)
        return optimizer, rho, x, y

    def _solve_with_optimizer(self, optimizer: Optimize, rho: ArithRef, timeout_ms: int, initial_upper_bound=None):
        start_time = time.time()
        if initial_upper_bound is not None and initial_upper_bound != float('inf'):
            print(f"Applying Advanced Heuristic UB: {initial_upper_bound}")
            optimizer.add(rho <= initial_upper_bound)
            
        optimizer.set("timeout", max(1, timeout_ms))
        status = optimizer.check()
        model, is_optimal = None, False
        
        if status == sat:
            elapsed_ms = (time.time() - start_time) * 1000
            # Considered optimal only if solver finishes before timeout
            if elapsed_ms < timeout_ms:
                is_optimal = True
            model = optimizer.model()
            print(f"Solution found. Status: SAT. Optimal: {is_optimal}")
        elif status == unknown:
            print("Solver timed out (unknown). A sub-optimal solution may be available.")
            try:
                model = optimizer.model()
            except Z3Exception:
                model = None
            is_optimal = False
        else: # unsat
            print("Problem is UNSAT (no solution exists).")
            is_optimal = True # UNSAT is a proven state
            
        return model, is_optimal

    def _extract_paths(self, model: ModelRef, y_vars, m: int, n: int, reverse_map=None):
        if not model: return [[] for _ in range(m)]
        
        depot_idx_0based = n
        if reverse_map is None:
            reverse_map = {i: i + 1 for i in range(n)} # Default map if no sorting
            
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
            if depot_idx_0based not in tour_map:
                all_routes.append([]); continue # Empty route

            current_node = tour_map[depot_idx_0based]
            route = []
            visited_count = 0
            while current_node != depot_idx_0based and visited_count < n:
                route.append(reverse_map[current_node])
                current_node = tour_map.get(current_node, depot_idx_0based)
                visited_count += 1
            all_routes.append(route)
            
        return all_routes

    def _recalculate_objective(self, m: int, n: int, D: list, solution_paths: list) -> int:
        """
        Recalculates the objective value from a given solution path using pure integer arithmetic
        to ensure consistency with the external checker.
        """
        if not solution_paths:
            return -1

        max_dist = 0
        depot_idx = n  # 0-based index for distance matrix

        for k in range(m):
            route = solution_paths[k]
            if not route:
                continue

            dist = 0
            last_node_idx = depot_idx
            # Item IDs in the solution are 1-based
            for item_id in route:
                item_idx = item_id - 1  # Convert to 0-based for matrix access
                dist += D[last_node_idx][item_idx]
                last_node_idx = item_idx
            
            # Add distance from last item back to depot
            dist += D[last_node_idx][depot_idx]

            if dist > max_dist:
                max_dist = dist
                
        return max_dist

    def solve(self, instance_file: str, output_dir: str):
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
            reverse_item_map = {i: i + 1 for i in range(n)}

            if use_sorting and n > 0:
                print("Applying item pre-sorting by remoteness...")
                # Sort items by distance from depot (remotest first)
                original_indices_0based = sorted(range(n), key=lambda j: original_D[j][n] + original_D[n][j], reverse=True)
                sorted_to_original_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(original_indices_0based)}
                reverse_item_map = {new_idx: sorted_to_original_map[new_idx] + 1 for new_idx in range(n)}
                
                # Remap sizes and distance matrix for the solver
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
                    _, greedy_ub = self._local_search_heuristic(m, n, caps, sizes_to_solve, D_to_solve)
                
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
                # First, extract the paths from the model.
                final_sol = self._extract_paths(model, y, m, n, reverse_map=reverse_item_map)
                
                # *** FIX: Recalculate the objective from the extracted paths and ORIGINAL distance matrix. ***
                # This avoids any solver precision/rounding issues and guarantees consistency.
                final_obj = self._recalculate_objective(m, n, original_D, final_sol)
            
            if solve_time_sec >= self.timeout:
                is_optimal = False

            results[name] = {
                "time": min(self.timeout, round(solve_time_sec, 2)),
                "optimal": is_optimal,
                "obj": int(final_obj),
                "sol": final_sol
            }

        out_path = os.path.join(output_dir, f"{inst_id}.json")
        with open(out_path, "w") as jf:
            json.dump(results, jf, indent=4)
        print(f"Enhanced SMT results for instance {inst_id} written to {out_path}\n")

