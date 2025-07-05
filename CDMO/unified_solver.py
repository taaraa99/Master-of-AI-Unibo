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
from models.sat.sat import Instance, load_instance, optimise, lns_optimise
from models.smt.SMT_Solver import SMT_Solver
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
        
class CPSolver:
    # This class would also benefit from the same timeout logic.
    # For now, it's a placeholder.
    def solve(self, instance_file, output_dir, time_limit=300):
        log(f"CP Solver for {instance_file} is not fully implemented with timeout.")
        pass

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
        mip2_approaches = ["CBC", "SCIP", "CBC+SB", "SCIP+SB"]
        for approach in mip2_approaches:
            log(f"[MIPSolver] Preparing approach: {approach} for instance: {base_name}")

            args = (m, n, capacities, item_sizes, dist, time_limit, approach)
            res = run_with_timeout(build_and_solve_mcp2, args, time_limit)
            
            key = approach.upper().replace("+", "_")
            if res is None: # Timeout occurred
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
        elif self.solver_type == "smt": self.solver = SMT_Solver()
        elif self.solver_type == "sat": self.solver = SAT_Solver()
        else: raise ValueError("Invalid solver type: choose 'cp', 'mip', 'smt', or 'sat'.")

    def solve_all_instances(self):
        instance_files = sorted([f for f in os.listdir(self.instances_dir) if f.endswith(".dat")])
        for fn in instance_files:
            path = os.path.join(self.instances_dir, fn)
            try:
                self.solver.solve(path, self.output_dir, time_limit=300)
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
