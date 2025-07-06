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

# We import our OR-Tools model functions
from models.mip.MIP_HiGHS import read_instance as read_instance1, build_and_solve_mcp as build_and_solve_mcp1
from models.mip.MIP_CBC_SCIP import read_instance as read_instance2, build_and_solve_mcp as build_and_solve_mcp2
# from models.sat.sat import Instance, load_instance, optimise, lns_optimise
from models.sat.sat import load_instance as load_instance_sat, optimise
from models.smt.SMT_Solver import (
        load_instance as load_instance_smt,
        optimise as optimise_smt,
        lns_optimise as lns_optimise_smt,
        z3_optimise as z3_optimise_smt
    )
import minizinc

def log(message):
    """Helper function to print messages with a timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

def solve_process_wrapper(result_queue: multiprocessing.Queue, solver_func: Callable, args: tuple):
    """
    A generic wrapper to run any solver function in a separate process 
    and put the result or any exception into a queue.
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
    Runs a solver function with a total timeout.
    Returns the solution dictionary or None if the time limit is exceeded.
    """
    result_queue = multiprocessing.Queue()
    
    process = multiprocessing.Process(target=solve_process_wrapper, args=(result_queue, solver_func, args))
    
    log(f"Starting solver process for {solver_func.__name__} with a {timeout}s timeout.")
    process.start()
    
    # Wait for the process to finish, or for the timeout to expire
    process.join(timeout)
    
    if process.is_alive():
        log(f"Total time limit of {timeout}s exceeded for {solver_func.__name__}. Terminating process.")
        process.terminate()
        process.join() # Wait for the process to terminate
        return None # Indicates a timeout
    
    log("Process finished within the time limit.")
    # Safely get the result from the queue
    try:
        result = result_queue.get_nowait()
        if isinstance(result, Exception):
            raise result
        return result
    except queue.Empty:
        log("Warning: Process finished but queue was empty.")
        return None


class SMT_Solver:
        def __init__(self, lns_iters: int = 20, destroy_frac: float = 0.3):
            self.lns_iters = lns_iters
            self.destroy_frac = destroy_frac
            # These configurations call the functions imported from your SMT solver file.
            self.configs: List[Tuple[str, Dict]] = [
                ("smt_binary",      {'func': optimise_smt, 'kwargs': {'strategy': 'binary', 'knn': None}}),
                ("smt_linear",      {'func': optimise_smt, 'kwargs': {'strategy': 'linear', 'knn': None}}),
                ("smt_z3",          {'func': z3_optimise_smt, 'kwargs': {'knn': None}}),
                ("smt_binary_knn6", {'func': optimise_smt, 'kwargs': {'strategy': 'binary', 'knn': 6}}),
                ("smt_z3_knn6",     {'func': z3_optimise_smt, 'kwargs': {'knn': 6}}),
                ("smt_lns",         {'func': lns_optimise_smt, 'kwargs': {'strategy': 'binary', 'knn': None}}),
                ("smt_lns_knn6",    {'func': lns_optimise_smt, 'kwargs': {'strategy': 'binary', 'knn': 6}}),
            ]

        def solve(self, instance_file: str, output_dir: str, time_limit: int = 300):
            os.makedirs(output_dir, exist_ok=True)
            inst = load_instance_smt(Path(instance_file))
            results = {}

            for name, cfg in self.configs:
                log(f"[SMT_Solver] Running approach: {name} for instance: {os.path.basename(instance_file)}")
                t0 = time.perf_counter()
                
                final_obj, final_tours, is_optimal = -1, [[] for _ in range(inst.m)], False
                
                try:
                    # Prepare arguments for the specific function call
                    solver_func = cfg['func']
                    solver_kwargs = cfg['kwargs'].copy()
                    solver_kwargs['inst'] = inst
                    solver_kwargs['timeout'] = time_limit

                    # Add LNS-specific parameters if calling lns_optimise
                    if solver_func is lns_optimise_smt:
                        solver_kwargs['lns_iters'] = self.lns_iters
                        solver_kwargs['destroy_fraction'] = self.destroy_frac
                    
                    final_obj, final_tours, is_optimal = solver_func(**solver_kwargs)

                except RuntimeError as e:
                    log(f"   [Warning] No solution found for {name}: {e}")
                except Exception as e:
                    log(f"   [ERROR] An unexpected error occurred in {name}: {e}")
                    traceback.print_exc()

                elapsed = time.perf_counter() - t0
                time_reported = math.floor(elapsed)
                
                if time_reported >= time_limit:
                    time_reported = time_limit
                    is_optimal = False
                if "lns" in name or "knn" in name:
                    is_optimal = False

                solution_1_based = [[item_idx + 1 for item_idx in route] for route in final_tours]
                results[name] = {
                    "time": time_reported, 
                    "optimal": is_optimal, 
                    "obj": final_obj, 
                    "sol": solution_1_based
                }

            base = os.path.splitext(os.path.basename(instance_file))[0]
            # Ensure inst_id is handled safely if no digits are found
            digits = re.search(r"\d+", base)
            inst_id = digits.group(0) if digits else base
            out_path = os.path.join(output_dir, f"{inst_id}.json")
            
            if os.path.exists(out_path):
                with open(out_path, "r") as jf:
                    try:
                        existing_results = json.load(jf)
                    except json.JSONDecodeError:
                        existing_results = {}
                existing_results.update(results)
                results = existing_results
                
            with open(out_path, "w") as jf:
                json.dump(results, jf, indent=4)
            log(f"All SMT approaches for instance {inst_id} written to {out_path}")
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
                distance_values = []
                # Handle flattened or matrix-style distance inputs
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
        # MiniZinc's arrays are 1-indexed, adjust for 0-indexed Python list
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
            print(f"Couriers: {m}, Items: {n}, Capacities: {l}, Sizes: {s}")
            model = minizinc.Model(model_file)
            print("Model loaded")

            os.makedirs(output_dir, exist_ok=True)
            all_results = {}

            cp_solvers = {
                "gecode": minizinc.Solver.lookup("gecode"),
                "chuffed": minizinc.Solver.lookup("chuffed")
            }
            configs = [
                ("no-sb", False, False),
                ("no-sb-imp", False, True),
                ("sb", True, False),
                ("sb-imp", True, True)
            ]
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
                    instance["couriers"] = m
                    instance["items"] = n
                    instance["courier_capacity"] = l
                    instance["item_size"] = s
                    instance["distance_matrix"] = D
                    instance["USE_SB"] = use_sb
                    instance["USE_IMP"] = use_imp

                    start_time = time.time()
                    result = instance.solve(timeout=timedelta(seconds=300))
                    solving_time = time.time() - start_time
                    
                    routes = [[] for _ in range(m)]
                    best_obj = -1
                    optimal = (result.status == minizinc.result.Status.OPTIMAL_SOLUTION)

                    if result.solution:
                        best_obj = result.objective
                        # Extract routes carefully from result object
                        if hasattr(result.solution, 'nextStop'):
                           ns_per_courier = result.solution.nextStop
                           routes = [CPSolver.reconstruct_route(ns, n, origin) for ns in ns_per_courier]
                    
                    all_results[approach_key] = {
                        "time": int(min(solving_time, 300)),
                        "optimal": optimal,
                        "obj": best_obj if best_obj is not None else -1,
                        "sol": routes
                    }

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
            log(f"Error reading instance '{instance_file}': {e}")
            return
        
        base_name = os.path.splitext(os.path.basename(instance_file))[0]
        inst_id = re.search(r"\d+", base_name).group(0)
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        
        # --- HiGHS Approaches ---
        ortools_approaches = ["HiGHS", "HiGHS+SB", "HiGHS+SB+IMPLIED", "HiGHS+SB+IMPLIED+WM", "HiGHS+SB+IMPLIED+WM+CUT"]
        for approach in ortools_approaches:
            log(f"[MIPSolver] Preparing approach: {approach} for instance: {instance_file}")
            
            args = (m, n, capacities, item_sizes, dist, time_limit, approach)
            sol = run_with_timeout(build_and_solve_mcp1, args, time_limit)
            
            key = approach.upper().replace("+", "_")
            if sol is None: # Timeout occurred
                all_results[key] = {
                    "time": time_limit, "optimal": False, "obj": -1.0, 
                    "sol": [], "status": "Total timeout exceeded"
                }
            else:
                # The returned `sol` is the full dictionary, e.g., {'HIGHS+SB': {...}}
                # We update all_results with it.
                all_results.update(sol)

        # --- CBC/SCIP Approaches ---
        mip2_approaches = ["CBC", "SCIP"]
        for approach in mip2_approaches:
            log(f"[MIPSolver] Preparing approach: {approach} for instance: {base_name}")

            args = (m, n, capacities, item_sizes, dist, time_limit, approach)
            res = run_with_timeout(build_and_solve_mcp2, args, time_limit)
            
            key = approach.upper().replace("+", "_")
            if res is None or res.time>time_limit: # Timeout occurred
                all_results[key] = {
                    "time": time_limit, "optimal": False, "obj": -1.0, 
                    "sol": [], "status": "Total timeout exceeded"
                }
            else:
                all_results.update(res)

        out_file = os.path.join(output_dir, f"{inst_id}.json")
        with open(out_file, 'w') as jf:
            json.dump(all_results, jf, indent=4)
        log(f"All MIP approaches for instance {inst_id} written to {out_file}")


class SAT_Solver:
    def __init__(self, timeout_per_config: int = 300):
        self.timeout = timeout_per_config
        # Configurations are updated to match the new SAT model's capabilities
        self.configs: List[Tuple[str, Dict]] = [
            ("sat_linear", dict(strategy="linear")),
            ("sat_binary", dict(strategy="binary")),
        ]
        
    def solve(self, instance_file: str, output_dir: str, time_limit: int = 300):
        os.makedirs(output_dir, exist_ok=True)
        inst = load_instance_sat(Path(instance_file))
        results = {}

        for name, cfg in self.configs:
            log(f"[SAT_Solver] Running approach: {name} for instance: {os.path.basename(instance_file)}")
            t0 = time.perf_counter()
            
            final_obj, final_tours, is_optimal = -1, [[] for _ in range(inst.m)], False
            
            try:
                # The logic is now simplified, as only the `optimise` function exists.
                final_obj, final_tours, is_optimal = optimise(
                    inst, timeout=time_limit, strategy=cfg["strategy"]
                )
            except RuntimeError as e:
                log(f"   [Warning] No solution found for {name}: {e}")
            except Exception as e:
                log(f"   [ERROR] An unexpected error occurred in {name}: {e}")
                traceback.print_exc()

            elapsed = time.perf_counter() - t0
            time_reported = math.floor(elapsed)
            
            if time_reported >= time_limit:
                time_reported = time_limit
                is_optimal = False

            # The routes from the SAT model are 0-indexed, convert to 1-indexed for output
            solution_1_based = [[item_idx + 1 for item_idx in route] for route in final_tours]
            results[name] = {
                "time": time_reported, 
                "optimal": is_optimal, 
                "obj": final_obj, 
                "sol": solution_1_based
            }

        base = os.path.splitext(os.path.basename(instance_file))[0]
        digits = re.search(r"\d+", base)
        inst_id = digits.group(0) if digits else base
        out_path = os.path.join(output_dir, f"{inst_id}.json")
        
        # Merge with existing results if the JSON file already exists
        if os.path.exists(out_path):
            with open(out_path, "r") as jf:
                try:
                    existing_results = json.load(jf)
                except json.JSONDecodeError:
                    existing_results = {}
            existing_results.update(results)
            results = existing_results
            
        with open(out_path, "w") as jf:
            json.dump(results, jf, indent=4)
        log(f"All SAT approaches for instance {inst_id} written to {out_path}")


class UnifiedSolver:
    def __init__(self, solver_type, base_dir=".", output_dir="res"):
        self.solver_type = solver_type.lower()
        self.instances_dir = os.path.join(base_dir, "Instances")
        self.output_dir = os.path.join(base_dir, output_dir, self.solver_type)
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.solver_type == "cp": self.solver = CPSolver()
        elif self.solver_type == "mip": self.solver = MIPSolver()
        elif self.solver_type == "smt": self.solver = SMT_Solver()
        elif self.solver_type == "sat": self.solver = SAT_Solver()
        else: raise ValueError("Invalid solver type: choose 'cp', 'mip', 'smt', or 'sat'.")

    def solve_all_instances(self):
        instance_files = sorted([f for f in os.listdir(self.instances_dir) if f.endswith(".dat")])
        for fn in instance_files:
            path = os.path.join(self.instances_dir, fn)
            try:

                self.solver.solve(path, self.output_dir)

            except Exception as e:
                log(f"FATAL: A critical error occurred while processing {fn} with {self.solver_type} solver. Aborting instance.")
                log(f"Error: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    # This is required for multiprocessing to work correctly on some platforms
    multiprocessing.freeze_support() 
    
    if len(sys.argv) > 1:
        solver_type_arg = sys.argv[1]
        unified = UnifiedSolver(solver_type_arg, base_dir=".", output_dir="res")
        unified.solve_all_instances()
    else:
        print("Usage: python unified_solver.py <solver_type>")
        print("solver_type can be: cp, mip, smt, sat")