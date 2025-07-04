import os
import json
import sys
import time
import traceback
from typing import Dict, List, Tuple
from datetime import timedelta
import re
import math # Import math for floor function
from pathlib import Path # Import the Path object

import minizinc
from z3 import is_true, ModelRef, Bool, sat

# We import our OR‑Tools model functions from two files.
# The first import is used when running the first five approaches (the HiGHS variants).
from models.mip.MIP_HiGHS import read_instance as read_instance1, build_and_solve_mcp as build_and_solve_mcp1
# The second import is used for the last two approaches (CBC and SCIP).
from models.mip.MIP_CBC_SCIP import read_instance as read_instance2, build_and_solve_mcp as build_and_solve_mcp2

#Import the methods used for SAT
from models.sat.sat import Instance, load_instance, optimise, lns_optimise

# Import the new SMT Solver class
from models.smt.SMT_Solver import SMT_Solver

# This class handles constraint programming (CP) solving.
class CPSolver:
    @staticmethod
    def read_instances(file_name: str) -> Tuple[int, int, List[int], List[int], List[List[int]]]:
        try:
            with open(file_name, 'r') as file:
                lines = [line.strip() for line in file if line.strip()]
                if len(lines) < 4:
                    raise ValueError("Instance file does not contain enough lines.")
                m = int(lines[0])
                n = int(lines[1])
                l = [int(i) for i in lines[2].split()]
                s = [int(i) for i in lines[3].split()]
                if len(l) != m:
                    raise ValueError(f"Expected {m} courier capacities, found {len(l)}.")
                if len(s) != n:
                    raise ValueError(f"Expected {n} item sizes, found {len(s)}.")
                flat_d = " ".join(lines[4:]).split()
                if len(flat_d) != (n + 1) * (n + 1):
                        raise ValueError(f"Expected {(n + 1) * (n + 1)} distance matrix elements, found {len(flat_d)}.")
                D = [[int(flat_d[i * (n + 1) + j]) for j in range(n + 1)] for i in range(n + 1)]
                return m, n, l, s, D
        except Exception as e:
            print(f"Error reading instance file '{file_name}': {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def reconstruct_route(ns: List[int], n: int, origin: int) -> List[int]:
        route = []
        if not ns: return route
        origin_idx = origin 
        current = ns[origin_idx - 1]
        visited = {origin}
        while current != origin and current not in visited:
            visited.add(current)
            if 1 <= current <= n:
                route.append(current)
            current = ns[current - 1]
        return route

    @staticmethod
    def run_cp_solver(instance_file: str, model_file: str, output_dir: str):
        try:
            m, n, l, s, D = CPSolver.read_instances(instance_file)
            print(f"\nProcessing instance: '{instance_file}'")
            model = minizinc.Model(model_file)
            os.makedirs(output_dir, exist_ok=True)
            all_results = {}
            cp_solvers = {"gecode": minizinc.Solver.lookup("gecode"), "chuffed": minizinc.Solver.lookup("chuffed")}
            configs = [("no-sb", False, False), ("no-sb-imp", False, True), ("sb", True, False), ("sb-imp", True, True)]
            base_name = os.path.splitext(os.path.basename(instance_file))[0]
            inst_id = re.search(r"\d+", base_name).group(0)
            origin = n + 1
            for solver_name, solver in cp_solvers.items():
                if solver is None:
                    print(f"Warning: {solver_name} solver not found.")
                    continue
                for conf_name, use_sb, use_imp in configs:
                    approach_key = f"{solver_name}-{conf_name}"
                    print(f"\nRunning model with {approach_key} configuration...")
                    instance = minizinc.Instance(solver, model)
                    instance["couriers"], instance["items"], instance["courier_capacity"], instance["item_size"], instance["distance_matrix"], instance["USE_SB"], instance["USE_IMP"] = m, n, l, s, D, use_sb, use_imp
                    start_time = time.time()
                    result = instance.solve(timeout=timedelta(seconds=300))
                    solving_time = time.time() - start_time
                    routes, best_obj = [[] for _ in range(m)], -1
                    optimal = (result.status == minizinc.result.Status.OPTIMAL_SOLUTION)
                    if result.solution:
                        best_obj = result.objective
                        if hasattr(result.solution, 'nextStop'):
                            ns_per_courier = result.solution.nextStop
                            routes = [CPSolver.reconstruct_route(ns, n, origin) for ns in ns_per_courier]
                    all_results[approach_key] = {"time": int(min(solving_time, 300)), "optimal": optimal, "obj": best_obj if best_obj is not None else -1, "sol": routes}
            out_path = os.path.join(output_dir, f"{inst_id}.json")
            with open(out_path, 'w') as jf:
                json.dump(all_results, jf, indent=4)
            print(f"All CP approaches for instance {inst_id} written to {out_path}")
        except Exception as e:
            print(f"Error solving instance '{instance_file}': {e}")
            traceback.print_exc()

    def solve(self, instance_file: str, output_dir: str):
        model_file = os.path.join("models", "cp", "cp01.mzn")
        self.run_cp_solver(instance_file, model_file, output_dir)

class MIPSolver:
    def solve(self, instance_file, output_dir, time_limit=300):
        try:
            m, n, capacities, item_sizes, dist = read_instance1(instance_file)
        except Exception as e:
            print("Error reading instance:", e)
            return
        base_name = os.path.splitext(os.path.basename(instance_file))[0]
        inst_id = re.search(r"\d+", base_name).group(0)
        os.makedirs(output_dir, exist_ok=True)
        all_results = {}
        ortools_approaches = ["HiGHS", "HiGHS+SB", "HiGHS+SB+IMPLIED", "HiGHS+SB+IMPLIED+WM", "HiGHS+SB+IMPLIED+WM+CUT"]
        for approach in ortools_approaches:
            print(f"[MIPSolver] Running approach: {approach} for instance: {instance_file}")
            sol = build_and_solve_mcp1(m, n, capacities, item_sizes, dist, time_limit=time_limit, approach=approach)
            key = approach.upper().replace("+", "_")
            inner = sol.get(approach, {})
            all_results[key] = inner
        mip2_approaches = ["CBC", "SCIP"]
        for approach in mip2_approaches:
            print(f"[MIPSolver] Running approach: {approach} for instance: {base_name}")
            res = build_and_solve_mcp2(m, n, capacities, item_sizes, dist, time_limit=time_limit, approach=approach)
            key = approach.upper()
            inner = res.get(key, {})
            all_results[key] = inner
        out_file = os.path.join(output_dir, f"{inst_id}.json")
        with open(out_file, 'w') as jf:
            json.dump(all_results, jf, indent=4)
        print(f"All MIP approaches for instance {inst_id} written to {out_file}")

class SAT_Solver:
    def __init__(self, timeout_per_config: int = 300, lns_iters: int = 20, destroy_frac: float = 0.3):
        self.timeout = timeout_per_config
        self.lns_iters = lns_iters
        self.destroy_frac = destroy_frac
        self.configs: List[Tuple[str, Dict]] = [
            ("linear",        dict(strategy="linear", knn=None, lns=False)),
            ("binary",        dict(strategy="binary", knn=None, lns=False)),
            ("binary_knn6",   dict(strategy="binary", knn=6,    lns=False)),
            ("lns",           dict(strategy="binary", knn=None, lns=True)),
            ("lns_knn6",      dict(strategy="binary", knn=6,    lns=True)),
        ]

    def solve(self, instance_file: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        inst = load_instance(Path(instance_file))
        results = {}
        for name, cfg in self.configs:
            print(f"[SAT_Solver] Running approach: {name} for instance: {os.path.basename(instance_file)}")
            t0 = time.time()
            # Default values in case of timeout or error
            final_obj, final_tours, is_optimal = -1, [[] for _ in range(inst.m)], False
            try:
                # *** FIX: Capture the new 'is_optimal' boolean returned by the functions ***
                if cfg["lns"]:
                    final_obj, final_tours, is_optimal = lns_optimise(
                        inst, timeout=self.timeout, strategy=cfg["strategy"],
                        knn=cfg["knn"], lns_iters=self.lns_iters, destroy_fraction=self.destroy_frac
                    )
                else:
                    final_obj, final_tours, is_optimal = optimise(
                        inst, timeout=self.timeout, strategy=cfg["strategy"], knn=cfg["knn"]
                    )
            except RuntimeError as e:
                print(f"  [Warning] No solution found for {name}: {e}")
            
            elapsed = time.time() - t0
            time_reported = math.floor(elapsed)
            
            # The optimality flag is now correct, but if we timed out, force it to false
            if time_reported >= self.timeout:
                time_reported = self.timeout
                is_optimal = False
            if "lns" in name or "knn" in name:
                is_optimal = False

            solution_1_based = [[item_idx + 1 for item_idx in route] for route in final_tours]
            results[name] = {"time": time_reported, "optimal": is_optimal, "obj": final_obj, "sol": solution_1_based}

        base = os.path.splitext(os.path.basename(instance_file))[0]
        inst_id = re.search(r"\d+", base).group(0)
        out_path = os.path.join(output_dir, f"{inst_id}.json")
        if os.path.exists(out_path):
            with open(out_path, "r") as jf:
                try: existing_results = json.load(jf)
                except json.JSONDecodeError: existing_results = {}
            existing_results.update(results)
            results = existing_results
        with open(out_path, "w") as jf:
            json.dump(results, jf, indent=4)
        print(f"All SAT approaches for instance {inst_id} written to {out_path}")

class UnifiedSolver:
    def __init__(self, solver_type, base_dir=".", output_dir="res"):
        self.solver_type = solver_type.lower()
        self.instances_dir = os.path.join(base_dir, "Instances")
        self.output_dir = os.path.join(base_dir, output_dir, self.solver_type)
        os.makedirs(self.output_dir, exist_ok=True)
        if self.solver_type == "cp": self.solver = CPSolver()
        elif self.solver_type == "mip": self.solver = MIPSolver()
        elif self.solver_type == "smt": self.solver = SMT_Solver(timeout=300)
        elif self.solver_type == "sat": self.solver = SAT_Solver(timeout_per_config=300)
        else: raise ValueError("Invalid solver type: choose 'cp', 'mip', 'smt', or 'sat'.")

    def solve_all_instances(self):
        instance_files = sorted([f for f in os.listdir(self.instances_dir) if f.endswith(".dat")])
        for fn in instance_files:
            path = os.path.join(self.instances_dir, fn)
            try:
                self.solver.solve(path, self.output_dir)
            except Exception as e:
                print(f"FATAL: A critical error occurred while processing {fn} with {self.solver_type} solver. Aborting instance.")
                print(f"Error: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        solver_type = sys.argv[1]
        unified = UnifiedSolver(solver_type, base_dir=".", output_dir="res")
        unified.solve_all_instances()
    else:
        print("Usage: python unified_solver.py <solver_type>")
        print("solver_type can be: cp, mip, smt, sat")
