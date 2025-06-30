import time
import json
import re
import math
import os
import traceback
from z3 import *

class SMT_Solver:
    """
    An SMT solver for the MCP problem using a powerful GRAPH-BASED formulation
    with multiple configurable search strategies (Linear, Binary) and enhancements.
    """

    def __init__(self, timeout: int = 300):
        self.timeout = timeout
        # --- ENHANCEMENT: Full suite of configurations for experiments ---
        self.configs = [
            ("smt_linear",               {"search_type": "linear", "use_greedy": False, "use_sorting": False}),
            ("smt_linear_greedy",        {"search_type": "linear", "use_greedy": True,  "use_sorting": False}),
            ("smt_binary",               {"search_type": "binary", "use_greedy": False, "use_sorting": False}),
            ("smt_binary_greedy",        {"search_type": "binary", "use_greedy": True,  "use_sorting": False}),
            ("smt_binary_sorted",        {"search_type": "binary", "use_greedy": True,  "use_sorting": True}),
        ]

    # ─────────────────────────── Helper methods ───────────────────────────
    @staticmethod
    def _exactly_one(bools: list, tag: str = "") -> BoolRef:
        return PbEq([(b, 1) for b in bools], 1)
        
    @staticmethod
    def _lex_leq(a: list, b: list) -> BoolRef:
        less = And(Not(a[0]), b[0])
        equal = a[0] == b[0]
        for i in range(1, len(a)):
            less = Or(less, And(equal, Not(a[i]), b[i]))
            equal = And(equal, a[i] == b[i])
        return Or(equal, less)

    # ─────────────────── Greedy Heuristic Pre-Solver ───────────────────
    def _greedy_heuristic(self, m: int, n: int, caps: list, sizes: list, D: list) -> (list, int):
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
                            cost_increase = (D[prev_stop-1][item_id-1] + D[item_id-1][next_stop-1] - D[prev_stop-1][next_stop-1])
                            if cost_increase < best_cost_increase:
                                best_cost_increase = cost_increase
                                best_insertion = (item_idx, k, pos)
            if best_insertion:
                item_idx, k, pos = best_insertion
                item_to_add = unassigned_items.pop(item_idx)
                routes[k].insert(pos, item_to_add)
                loads[k] += sizes[item_to_add]
            else: return None, float('inf')
        max_dist = 0
        for k in range(m):
            dist, last_stop = 0, depot
            for item_id in routes[k]:
                dist += D[last_stop-1][item_id-1]
                last_stop = item_id
            dist += D[last_stop-1][depot-1]
            if dist > max_dist: max_dist = dist
        return routes, max_dist

    # ─────────────────────────── Core Solver Logic ───────────────────────────

    def _parse_instance(self, file_path: str):
        with open(file_path, 'r') as f: content = f.read()
        tokens = re.split(r'\s+', content.strip())
        m, n = int(tokens.pop(0)), int(tokens.pop(0))
        caps = [int(tokens.pop(0)) for _ in range(m)]
        sizes = [0] + [int(tokens.pop(0)) for _ in range(n)]
        D_flat = [int(t) for t in tokens if t]
        D = [D_flat[i*(n+1):(i+1)*(n+1)] for i in range(n+1)]
        return m, n, caps, sizes, D

    def _build_model(self, m, n, caps, sizes, D):
        solver = Solver()
        depot_idx = n 
        
        x = [[Bool(f"x_{k}_{j}") for j in range(n)] for k in range(m)]
        y = [[[Bool(f"y_{k}_{i}_{j}") for j in range(n + 1)] for i in range(n + 1)] for k in range(m)]
        u = [[Int(f"u_{k}_{j}") for j in range(n)] for k in range(m)]
        rho = Int("rho")

        for k in range(m):
            solver.add(Sum([If(x[k][j], sizes[j+1], 0) for j in range(n)]) <= caps[k])

        for j in range(n):
            solver.add(self._exactly_one([x[k][j] for k in range(m)]))

        for k in range(m):
            solver.add(PbLe([(y[k][depot_idx][j], 1) for j in range(n)], 1))
            
            for j in range(n):
                num_incoming = Sum([If(y[k][i][j], 1, 0) for i in range(n + 1) if i != j])
                num_outgoing = Sum([If(y[k][j][i], 1, 0) for i in range(n + 1) if i != j])
                solver.add(If(x[k][j], num_incoming == 1, num_incoming == 0))
                solver.add(If(x[k][j], num_outgoing == 1, num_outgoing == 0))

            solver.add(Sum([If(y[k][depot_idx][j], 1, 0) for j in range(n)]) == Sum([If(y[k][j][depot_idx], 1, 0) for j in range(n)]))

        for k in range(m):
            for j in range(n):
                solver.add(Implies(x[k][j], And(u[k][j] >= 1, u[k][j] <= n)))
                solver.add(Implies(Not(x[k][j]), u[k][j] == 0))
                solver.add(Implies(y[k][depot_idx][j], u[k][j] == 1))
                for i in range(n):
                    if i != j:
                        solver.add(Implies(y[k][i][j], u[k][j] == u[k][i] + 1))

        for k in range(m):
            dist = Sum([If(y[k][i][j], D[i][j], 0) for i in range(n + 1) for j in range(n + 1)])
            solver.add(rho >= dist)
            
        for k in range(m - 1):
            if caps[k] == caps[k+1]:
                solver.add(self._lex_leq(x[k], x[k+1]))
        
        return solver, rho, x, y

    # --- NEW: Re-implemented Linear Search ---
    def _linear_search(self, solver: Solver, rho: ArithRef, timeout_ms: int, initial_upper_bound=None):
        start_time = time.time()
        best_model, is_optimal = None, False
        
        if initial_upper_bound is not None and initial_upper_bound != float('inf'):
            print(f"Applying Greedy UB to Linear Search: {initial_upper_bound}")
            solver.add(rho <= initial_upper_bound)

        time_for_first_sol = timeout_ms
        solver.set("timeout", max(1, time_for_first_sol))
        
        if solver.check() != sat: return None, True 
        best_model = solver.model()
        
        while True:
            elapsed_ms = (time.time() - start_time) * 1000
            remaining_ms = max(1, timeout_ms - int(elapsed_ms))
            if remaining_ms <= 1: break
            
            obj_val = best_model.eval(rho).as_long()
            solver.add(rho < obj_val)
            solver.set("timeout", remaining_ms)
            
            status = solver.check()
            if status == sat:
                best_model = solver.model()
            elif status == unknown:
                is_optimal = False; break
            else: # unsat
                is_optimal = True; break
        return best_model, is_optimal

    def _binary_search(self, solver: Solver, rho: ArithRef, n: int, D: list, timeout_ms: int, initial_upper_bound=None):
        start_time = time.time()
        best_model, is_optimal = None, False
        lower_bound = max([D[i][n] + D[n][i] for i in range(n)]) if n > 0 else 0
        upper_bound = float('inf')

        if initial_upper_bound is not None and initial_upper_bound != float('inf'):
            upper_bound = initial_upper_bound
            solver.add(rho <= upper_bound)
        
        time_for_first_sol = int(timeout_ms / 4) if initial_upper_bound is None else int(timeout_ms)
        solver.set("timeout", max(1, time_for_first_sol))

        check_status = solver.check()
        if check_status == sat:
            initial_model = solver.model()
            smt_ub = initial_model.eval(rho).as_long()
            best_model, upper_bound = initial_model, min(smt_ub, upper_bound)
        elif upper_bound == float('inf'):
            print(f"Could not find an initial solution. Status: {check_status}")
            return None, True

        while lower_bound <= upper_bound:
            elapsed_ms = (time.time() - start_time) * 1000
            remaining_ms = max(1, timeout_ms - int(elapsed_ms))
            if remaining_ms <= 1: break
            mid = lower_bound + (upper_bound - lower_bound) // 2
            if mid > upper_bound: mid = upper_bound
            
            solver.push()
            solver.add(rho <= mid)
            solver.set("timeout", remaining_ms)
            if solver.check() == sat:
                best_model = solver.model()
                upper_bound = best_model.eval(rho).as_long() - 1
            else:
                lower_bound = mid + 1
            solver.pop()

        if (time.time() - start_time) * 1000 < timeout_ms: is_optimal = True
        return best_model, is_optimal
        
    def _extract_paths(self, model: ModelRef, y_vars, m: int, n: int, reverse_map=None):
        if not model: return [[] for _ in range(m)]
        depot_idx_0based = n
        if reverse_map is None: reverse_map = {i: i+1 for i in range(n)}
        
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
                all_routes.append([]); continue

            current_node = tour_map[depot_idx_0based]
            route = []
            visited_count = 0
            while current_node != depot_idx_0based and visited_count < n:
                route.append(reverse_map[current_node])
                current_node = tour_map.get(current_node, depot_idx_0based) # Safe get
                visited_count += 1
            all_routes.append(route)
        return all_routes

    def solve(self, instance_file: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(instance_file))[0]
        inst_id = re.search(r'\d+', base_name).group(0)
        results = {}

        for name, config in self.configs:
            print(f"----- [SMT Solver] -----")
            print(f"Running approach: {name} for instance: {inst_id}")
            
            presolve_start_time = time.time()
            m, n, caps, original_sizes, original_D = self._parse_instance(instance_file)
            
            use_greedy = config.get("use_greedy", False)
            use_sorting = config.get("use_sorting", False)
            timeout_ms = self.timeout * 1000
            
            sizes_to_solve, D_to_solve = original_sizes, original_D
            reverse_item_map = {i: i+1 for i in range(n)}

            if use_sorting and n > 0:
                print("Applying item pre-sorting by remoteness...")
                original_indices_0based = list(range(n))
                original_indices_0based.sort(key=lambda j: original_D[j][n] + original_D[n][j], reverse=True)
                
                sorted_to_original_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(original_indices_0based)}
                reverse_item_map = {orig_idx: new_idx + 1 for new_idx, orig_idx in sorted_to_original_map.items()}

                sizes_to_solve = [0] + [original_sizes[sorted_to_original_map[i]+1] for i in range(n)]
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
            total_start_time = time.time()
            
            try:
                solver, rho, x, y = self._build_model(m, n, caps, sizes_to_solve, D_to_solve)
                greedy_ub = None
                if use_greedy:
                    _, greedy_ub = self._greedy_heuristic(m, n, caps, sizes_to_solve, D_to_solve)
                
                remaining_timeout = timeout_ms - int((time.time() - presolve_start_time) * 1000)
                
                # --- NEW: Dispatcher for search strategies ---
                search_type = config.get("search_type")
                if search_type == "linear":
                    model, is_optimal = self._linear_search(solver, rho, remaining_timeout, initial_upper_bound=greedy_ub)
                elif search_type == "binary":
                    model, is_optimal = self._binary_search(solver, rho, n, D_to_solve, remaining_timeout, initial_upper_bound=greedy_ub)

            except Exception as e:
                print(f"[SMT Solver] ERROR on instance {base_name} with {name}: {e}")
                traceback.print_exc()

            solve_time = time.time() - total_start_time
            final_obj, final_sol = -1, [[] for _ in range(m)]

            if model:
                final_obj = model.eval(rho).as_long()
                final_sol = self._extract_paths(model, y, m, n, reverse_map=reverse_item_map)
            
            if solve_time * 1000 >= timeout_ms: is_optimal = False

            results[name] = {
                "time": min(self.timeout, math.floor(time.time() - presolve_start_time)),
                "optimal": is_optimal,
                "obj": final_obj,
                "sol": final_sol
            }

        out_path = os.path.join(output_dir, f"{inst_id}.json")
        with open(out_path, "w") as jf:
            json.dump(results, jf, indent=4)
        print(f"SMT results for instance {inst_id} written to {out_path}\n")