#
# This script is designed to be a one-stop-shop for runnin varius optimization models (MIP, CP, SAT, SMT)
# on a set of problem instances. Its built to be robust, handlin timeouts and organizing results


import os
import json
import sys
import time
import traceback
from typing import Dict, List, Tuple, Callable, Any
from datetime import timedelta
import re
import math
from pathlib import Path
import multiprocessing
import queue

# --- Model Imports ---
# We're pullin in the actual solver funtions from their own files
# Each file has the logic for a specific kind of model
from models.mip.MIP_HiGHS import read_instance as read_instance1, build_and_solve_mcp as build_and_solve_mcp1
from models.mip.MIP_CBC_SCIP import read_instance as read_instance2, build_and_solve_mcp as build_and_solve_mcp2
from models.sat.sat import load_instance as load_instance_sat, optimise
from models.smt.SMT_Solver import (
    load_instance as load_instance_smt,
    optimise as optimise_smt,
    lns_optimise as lns_optimise_smt,
    z3_optimise as z3_optimise_smt
)
import minizinc

# --- Helper Funtions ---

def log(message):
    """a simple helper to print stuff with a timestamp. Makes the console output way easier to reed."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

def save_results(output_path: str, new_results: Dict):
    """
    This is our centralized function for saving results.
    It safely reads a JSON file if it exists, merges the new results into it,
    and then writes the whole thing back. This stops us from overwriting data from a previous run.
    """
    all_results = {}
    if os.path.exists(output_path):
        log(f"Results file exists at {output_path}. Reading to merge.")
        with open(output_path, "r") as f:
            try:
                # Load whatever is already in the file.
                all_results = json.load(f)
            except json.JSONDecodeError:
                log(f"[WARNING] Could not parse existing JSON file at {output_path}. It will be overwritten.")
    
    # Update the main dictionary with the new results. This will add new keys
    # or overwrite existing ones if we're re-running the same approach.
    all_results.update(new_results)

    try:
        # Write the combined data back to the file.
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=4)
        log(f"Successfully saved/updated results file: {output_path}")
    except Exception as e:
        log(f"[ERROR] Failed to write results to {output_path}: {e}")


def solve_process_wrapper(result_queue: multiprocessing.Queue, solver_func: Callable, args: tuple):
    """
    this funtion is a safety net, it runs a solver in its own process.
    Why? Becuase some solvrs can be a bit wild and unpredictable, maybe they crash or eat all the memory
    runing them seperate like this isolates them, so if one of em fails, it don;t bring down the complet script.
    """
    try:
        log(f"Process {os.getpid()} starting solver function: {solver_func.__name__}")
        solution = solver_func(*args)
        result_queue.put(solution)
        log(f"Process {os.getpid()} finished successfully.")
    except Exception as e:
        log(f"Process {os.getpid()} encountered an exception: {e}")
        traceback.print_exc()
        result_queue.put(e)

def run_with_timeout(solver_func: Callable, args: tuple, timeout: int) -> Any:
    """
    Manages a solver proces and enforces a strict time limit.
    if a solver takes too long, we gotta stop it. this function handles that.
    """
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=solve_process_wrapper, args=(result_queue, solver_func, args))
    
    log(f"Starting solver process for {solver_func.__name__} with a {timeout}s timeout.")
    process.start()
    process.join(timeout)
    
    if process.is_alive():
        log(f"Total time limit of {timeout}s exceeded for {solver_func.__name__}. Terminating process.")
        process.terminate()
        process.join()
        return None
    
    log("Process finished within the time limit.")
    try:
        result = result_queue.get_nowait()
        if isinstance(result, Exception):
            raise result
        return result
    except queue.Empty:
        log("[WARNING] Process finished but the result queue was empty.")
        return None

# --- Solver Classes ---
# Each class below is a 'manager' for a specific type of solver.
# Their job now is JUST to solve, not to save files. They will return their results.

class SMT_Solver:
    """Manages runnin various SMT (Satisfiability Modulo Theories) approches."""
    def __init__(self, lns_iters: int = 20, destroy_frac: float = 0.3):
        self.lns_iters = lns_iters
        self.destroy_frac = destroy_frac
        self.configs: List[Tuple[str, Dict]] = [
            ("smt_binary", {'func': optimise_smt, 'kwargs': {'strategy': 'binary', 'knn': None}}),
            ("smt_linear", {'func': optimise_smt, 'kwargs': {'strategy': 'linear', 'knn': None}}),
            ("smt_z3", {'func': z3_optimise_smt, 'kwargs': {'knn': None}}),
            ("smt_binary_knn6", {'func': optimise_smt, 'kwargs': {'strategy': 'binary', 'knn': 6}}),
            ("smt_z3_knn6", {'func': z3_optimise_smt, 'kwargs': {'knn': 6}}),
            ("smt_lns", {'func': lns_optimise_smt, 'kwargs': {'strategy': 'binary', 'knn': None}}),
            ("smt_lns_knn6", {'func': lns_optimise_smt, 'kwargs': {'strategy': 'binary', 'knn': 6}}),
        ]

    def solve(self, instance_file: str, time_limit: int = 300) -> Dict:
        instance_name = os.path.basename(instance_file)
        log(f"[SMT_Solver] Starting to solve instance: {instance_name}")
        
        try:
            inst = load_instance_smt(Path(instance_file))
        except Exception as e:
            log(f"[SMT_Solver] [ERROR] Failed to load instance {instance_name}: {e}")
            return None # Return nothing if we can't load the file

        results = {}
        for name, cfg in self.configs:
            log(f"[SMT_Solver] Running approach: {name}")
            t0 = time.perf_counter()
            final_obj, final_tours, is_optimal = -1, [[] for _ in range(inst.m)], False
            
            try:
                solver_func = cfg['func']
                solver_kwargs = cfg['kwargs'].copy()
                solver_kwargs['inst'] = inst
                solver_kwargs['timeout'] = time_limit

                if solver_func is lns_optimise_smt:
                    solver_kwargs['lns_iters'] = self.lns_iters
                    solver_kwargs['destroy_fraction'] = self.destroy_frac
                
                final_obj, final_tours, is_optimal = solver_func(**solver_kwargs)

            except RuntimeError as e:
                log(f"[SMT_Solver] [WARNING] No solution found for '{name}': {e}")
            except Exception as e:
                log(f"[SMT_Solver] [ERROR] An unexpected error occurred in '{name}': {e}")
                traceback.print_exc()

            elapsed = time.perf_counter() - t0
            time_reported = math.floor(elapsed)
            
            if time_reported >= time_limit:
                time_reported = time_limit
                is_optimal = False
            if "lns" in name or "knn" in name:
                is_optimal = False

            solution_1_based = [[item_idx + 1 for item_idx in route] for route in final_tours]
            results[name] = {"time": time_reported, "optimal": is_optimal, "obj": final_obj, "sol": solution_1_based}
        
        return results

class CPSolver:
    """Manages runnin the Constraint Programming (CP) model with MiniZinc."""
    @staticmethod
    def read_instances(file_name: str) -> Tuple[int, int, List[int], List[int], List[List[int]]]:
        with open(file_name, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
            if len(lines) < 4: raise ValueError("Instance file does not contain enough data.")
            m, n = int(lines[0]), int(lines[1])
            l, s = [int(i) for i in lines[2].split()], [int(i) for i in lines[3].split()]
            if len(l) != m or len(s) != n: raise ValueError("Mismatch in courier/item counts and capacity/size lists.")
            flat_d = " ".join(lines[4:]).split()
            if len(flat_d) != (n + 1) * (n + 1): raise ValueError("Distance matrix has incorrect number of elements.")
            D = [[int(flat_d[i * (n + 1) + j]) for j in range(n + 1)] for i in range(n + 1)]
            return m, n, l, s, D

    @staticmethod
    def reconstruct_route(ns: List[int], n: int, origin: int) -> List[int]:
        route = []
        if not ns: return route
        current, visited = ns[origin - 1], {origin}
        while current != origin and current not in visited:
            visited.add(current)
            if 1 <= current <= n:
                route.append(current)
            current = ns[current - 1]
        return route

    def solve(self, instance_file: str, time_limit: int = 300) -> Dict:
        model_file = os.path.join("models", "cp", "cp01.mzn")
        instance_name = os.path.basename(instance_file)
        log(f"[CPSolver] Starting to solve instance: {instance_name}")
        
        try:
            m, n, l, s, D = CPSolver.read_instances(instance_file)
            log(f"[CPSolver] [INFO] Instance details - Couriers: {m}, Items: {n}")
            model = minizinc.Model(model_file)
            log(f"[CPSolver] [INFO] Model loaded from {model_file}")
        except Exception as e:
            log(f"[CPSolver] [ERROR] Failed to read instance or load model for {instance_name}: {e}")
            traceback.print_exc()
            return None

        all_results = {}
        cp_solvers = {"gecode": minizinc.Solver.lookup("gecode"), "chuffed": minizinc.Solver.lookup("chuffed")}
        configs = [("no-sb", False, False), ("no-sb-imp", False, True), ("sb", True, False), ("sb-imp", True, True)]
        origin = n + 1

        for solver_name, solver in cp_solvers.items():
            if solver is None:
                log(f"[CPSolver] [WARNING] MiniZinc solver '{solver_name}' not found, skipping.")
                continue
            for conf_name, use_sb, use_imp in configs:
                approach_key = f"{solver_name}-{conf_name}"
                log(f"[CPSolver] Running approach: {approach_key}")
                instance = minizinc.Instance(solver, model)
                instance["couriers"], instance["items"] = m, n
                instance["courier_capacity"], instance["item_size"] = l, s
                instance["distance_matrix"] = D
                instance["USE_SB"], instance["USE_IMP"] = use_sb, use_imp

                start_time = time.time()
                result = instance.solve(timeout=timedelta(seconds=time_limit))
                solving_time = time.time() - start_time
                
                routes, best_obj = [[] for _ in range(m)], -1
                optimal = result.status == minizinc.result.Status.OPTIMAL_SOLUTION

                if result.solution:
                    best_obj = result.objective
                    if hasattr(result.solution, 'nextStop'):
                        routes = [CPSolver.reconstruct_route(ns, n, origin) for ns in result.solution.nextStop]
                
                all_results[approach_key] = {"time": int(min(solving_time, time_limit)), "optimal": optimal, "obj": best_obj if best_obj is not None else -1, "sol": routes}

        return all_results

class MIPSolver:
    """
    Manages the Mixed-Integer Programming (MIP) models usin Google OR-Tools.
    THIS CLASS HAS BEEN FIXED to be more robust.
    """
    def solve(self, instance_file: str, time_limit: int = 300) -> Dict:
        instance_name = os.path.basename(instance_file)
        log(f"[MIPSolver] Starting to solve instance: {instance_name}")
        
        try:
            m, n, capacities, item_sizes, dist = read_instance1(instance_file)
        except Exception as e:
            log(f"[MIPSolver] [ERROR] Could not read instance file '{instance_name}': {e}")
            return None
        
        all_results = {}
        
        ortools_approaches = ["HiGHS", "HiGHS+SB", "HiGHS+SB+IMPLIED", "HiGHS+SB+IMPLIED+WM", "HiGHS+SB+IMPLIED+WM+CUT"]
        for approach in ortools_approaches:
            log(f"[MIPSolver] Running approach: {approach}")
            args = (m, n, capacities, item_sizes, dist, time_limit, approach)
            sol_value = run_with_timeout(build_and_solve_mcp1, args, time_limit)
            
            key = approach.upper().replace("+", "_")
            if sol_value is None:
                log(f"[MIPSolver] [WARNING] No solution returned for approach '{approach}'. Assuming timeout/failure.")
                all_results[key] = {"time": time_limit, "optimal": False, "obj": -1.0, "sol": [], "status": "No solution or timeout"}
            else:
                all_results[key] = sol_value

        mip2_approaches = ["CBC", "SCIP"]
        for approach in mip2_approaches:
            log(f"[MIPSolver] Running approach: {approach}")
            args = (m, n, capacities, item_sizes, dist, time_limit, approach)
            res_value = run_with_timeout(build_and_solve_mcp2, args, time_limit)
            
            key = approach.upper().replace("+", "_")
            if res_value is None:
                log(f"[MIPSolver] [WARNING] No solution returned for approach '{approach}'. Assuming timeout/failure.")
                all_results[key] = {"time": time_limit, "optimal": False, "obj": -1.0, "sol": [], "status": "No solution or timeout"}
            else:
                all_results[key] = res_value
        
        return all_results

class SAT_Solver:
    """Manages runnin the SAT-based model"""
    def __init__(self, timeout_per_config: int = 300):
        self.timeout = timeout_per_config
        self.configs: List[Tuple[str, Dict]] = [("sat_linear", dict(strategy="linear")), ("sat_binary", dict(strategy="binary"))]
        
    def solve(self, instance_file: str, time_limit: int = 300) -> Dict:
        instance_name = os.path.basename(instance_file)
        log(f"[SAT_Solver] Starting to solve instance: {instance_name}")
        
        try:
            inst = load_instance_sat(Path(instance_file))
        except Exception as e:
            log(f"[SAT_Solver] [ERROR] Failed to load instance {instance_name}: {e}")
            return None
        
        results = {}
        for name, cfg in self.configs:
            log(f"[SAT_Solver] Running approach: {name}")
            t0 = time.perf_counter()
            final_obj, final_tours, is_optimal = -1, [[] for _ in range(inst.m)], False
            
            try:
                final_obj, final_tours, is_optimal = optimise(inst, timeout=time_limit, strategy=cfg["strategy"])
            except RuntimeError as e:
                log(f"[SAT_Solver] [WARNING] No solution found for '{name}': {e}")
            except Exception as e:
                log(f"[SAT_Solver] [ERROR] An unexpected error occurred in '{name}': {e}")
                traceback.print_exc()

            elapsed = time.perf_counter() - t0
            time_reported = math.floor(elapsed)
            
            if time_reported >= time_limit:
                time_reported = time_limit
                is_optimal = False

            solution_1_based = [[item_idx + 1 for item_idx in route] for route in final_tours]
            results[name] = {"time": time_reported, "optimal": is_optimal, "obj": final_obj, "sol": solution_1_based}

        return results

class UnifiedSolver:
    """
    This figurs out which solver to use from the command line,
    and now it ALSO handles all the file saving, which is much better.
    """
    def __init__(self, solver_type, base_dir=".", output_dir="res"):
        self.solver_type = solver_type.lower()
        self.instances_dir = os.path.join(base_dir, "Instances")
        
        self.output_dir = os.path.join(base_dir, output_dir, self.solver_type.upper())
        os.makedirs(self.output_dir, exist_ok=True)
        
        solver_map = {"cp": CPSolver, "mip": MIPSolver, "smt": SMT_Solver, "sat": SAT_Solver}
        solver_class = solver_map.get(self.solver_type)
        if solver_class:
            self.solver = solver_class()
        else:
            raise ValueError("Invalid solver type: choose 'cp', 'mip', 'smt', or 'sat'.")

    def solve_all_instances(self):
        """Finds all '.dat' files in the instances directory and solves them one-by-one."""
        instance_files = sorted([f for f in os.listdir(self.instances_dir) if f.endswith(".dat")])
        log(f"[UnifiedSolver] Found {len(instance_files)} instances to solve with '{self.solver_type.upper()}' solver.")
        
        for fn in instance_files:
            path = os.path.join(self.instances_dir, fn)
            try:
                results_dict = self.solver.solve(path)

                if results_dict:
                    base_name = os.path.splitext(fn)[0]
                    inst_id_match = re.search(r"\d+", base_name)
                    
                    if inst_id_match:
                        number_as_string = inst_id_match.group(0)
                        formatted_id = f"{int(number_as_string):02d}"
                    else:
                        formatted_id = base_name
                    
                    out_path = os.path.join(self.output_dir, f"{formatted_id}.json")
                    save_results(out_path, results_dict)
                else:
                    log(f"[UnifiedSolver] [WARNING] Solver for '{fn}' did not return any results to save.")

            except Exception as e:
                log(f"[UnifiedSolver] [FATAL] A critical error occurred while processing {fn}. Aborting this instance.")
                log(f"[UnifiedSolver] [FATAL] Error: {e}")
                traceback.print_exc()

# --- Main Execution Block ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    if len(sys.argv) > 1:
        solver_type_arg = sys.argv[1]
        try:
            unified = UnifiedSolver(solver_type_arg, base_dir=".", output_dir="res")
            unified.solve_all_instances()
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Usage: python unified_solver.py <solver_type>")
        print("solver_type can be one of: cp, mip, smt, sat")