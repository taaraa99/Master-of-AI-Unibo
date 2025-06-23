import os
import json
import sys
import time
import traceback
import importlib.util
from datetime import datetime, timedelta
from typing import Tuple, List
import re

import minizinc
from z3 import *

# We import our OR‑Tools model functions from two files.
# The first import is used when running the first five approaches (the HiGHS variants).
from models.mip.MIP_HiGHS import read_instance as read_instance1, build_and_solve_mcp as build_and_solve_mcp1
# The second import is used for the last two approaches (CBC and SCIP).
from models.mip.MIP_CBC_SCIP import read_instance as read_instance2, build_and_solve_mcp as build_and_solve_mcp2


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
                for line in lines[4:]:
                    row_values = [int(i) for i in line.split()]
                    distance_values.extend(row_values)
                expected_distances = (n + 1) * (n + 1)
                if len(distance_values) != expected_distances:
                    raise ValueError(f"Expected {expected_distances} distance matrix elements, found {len(distance_values)}.")
                D = [distance_values[i*(n+1):(i+1)*(n+1)] for i in range(n+1)]
                return m, n, l, s, D
        except Exception as e:
            print(f"Error reading instance file '{file_name}': {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def reconstruct_route(ns: List[int], n: int, origin: int) -> List[int]:
        route = []
        if len(ns) != origin:
            print(f"Warning: nextStop array length {len(ns)} differs from expected {origin}.")
            return route
        if ns[origin - 1] == origin:
            return route
        current = ns[origin - 1]
        visited = set()
        while current != origin and current not in visited:
            if current < 1 or current > len(ns):
                print(f"Warning: encountered current={current} out of bounds (1..{len(ns)}).")
                break
            visited.add(current)
            if 1 <= current <= n:
                route.append(current)
            if current - 1 < 0 or current - 1 >= len(ns):
                print(f"Warning: index {current-1} out of bounds in nextStop array.")
                break
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

            # ensure output folder exists
            os.makedirs(output_dir, exist_ok=True)

            # we will collect all CP results here
            all_results = {}

            # Define CP solvers.
            cp_solvers = {
                "gecode": minizinc.Solver.lookup("gecode"),
                "chuffed": minizinc.Solver.lookup("chuffed")
            }
            # Define configurations: (config name, USE_SB, USE_IMP)
            configs = [
                ("no-sb", False, False),
                ("no-sb-imp", False, True),
                ("sb", True, False),
                ("sb-imp", True, True)
            ]
            base_name = os.path.splitext(os.path.basename(instance_file))[0]
            # extract instance number
            inst_id = re.search(r"\d+", base_name).group(0)
            origin = n + 1

            # Run each solver/config and accumulate
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
                    runtime_sec = int(min(solving_time, 300))
                    optimal = (result.status == minizinc.result.Status.OPTIMAL_SOLUTION)

                    # parse objective
                    best_obj = -1
                    for line in str(result).splitlines():
                        m_obj = re.match(r"^bestObj = (\d+)", line)
                        if m_obj:
                            best_obj = int(m_obj.group(1))
                            break

                    # reconstruct routes
                    ns_lines = []
                    for line in str(result).splitlines():
                        m_ns = re.match(r"^nextStop for courier (\d+): \[(.*)\]", line)
                        if m_ns:
                            cid = int(m_ns.group(1))
                            tokens = [t.strip() for t in m_ns.group(2).split(",") if t.strip()]
                            ns = []
                            for tk in tokens:
                                if tk.isdigit(): ns.append(int(tk))
                                elif tk.startswith("X_INTRODUCED_"):
                                    num = tk[len("X_INTRODUCED_"):].strip("_")
                                    ns.append(int(num))
                            ns_lines.append((cid, ns))
                    ns_lines.sort(key=lambda x: x[0])
                    routes = [CPSolver.reconstruct_route(ns, n, origin) for _, ns in ns_lines]
                    # pad or trim to m
                    # routes += [[]] * (m - len(routes)) if len(routes) < m else routes[:m]
                    if len(routes) < m:
                        routes += [[]] * (m - len(routes))
                    else:
                        routes = routes[:m]

                    # store result
                    all_results[approach_key] = {
                        "time": runtime_sec,
                        "optimal": optimal,
                        "obj": best_obj,
                        "sol": routes
                    }

            # write one JSON file per instance, named by instance number
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


# This class implements a combined MIP solver that uses two groups of approaches.
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

        # collect all MIP results
        all_results = {}

        # first five HiGHS variants
        ortools_approaches = [
            "HiGHS",
            "HiGHS+SB",
            "HiGHS+SB+IMPLIED",
            "HiGHS+SB+IMPLIED+WM",
            "HiGHS+SB+IMPLIED+WM+CUT"
        ]
        for approach in ortools_approaches:
            print(f"[MIPSolver] Running approach: {approach} for instance: {instance_file}")
            sol = build_and_solve_mcp1(m, n, capacities, item_sizes, dist,
                                       time_limit=time_limit, approach=approach)
            key = approach.upper()
            inner = sol[key] if (isinstance(sol, dict) and key in sol) else sol
            all_results[key] = inner

        # CBC and SCIP
        mip2_approaches = ["CBC", "SCIP"]
        for approach in mip2_approaches:
            print(f"[MIPSolver] Running approach: {approach} for instance: {base_name}")
            res = build_and_solve_mcp2(m, n, capacities, item_sizes, dist,
                                       time_limit=time_limit, approach=approach)
            key = approach.upper()
            inner = res[key] if (isinstance(res, dict) and key in res) else res
            all_results[key] = inner

        # write single JSON named by instance number
        out_file = os.path.join(output_dir, f"{inst_id}.json")
        with open(out_file, 'w') as jf:
            json.dump(all_results, jf, indent=4)
        print(f"All MIP approaches for instance {inst_id} written to {out_file}")


# SATSolver supports SAT-like routing with various heuristics
class SATSolver:
    """
    Delegates to the models.sat.runner module via command-line,
    passing through search, knn, lns, lns-iters, destroy-frac, timeout, and outdir.
    """
    def __init__(self, search: str = 'binary', timeout_s: int = 300,
                 knn: int = None, lns: bool = False,
                 lns_iters: int = None, destroy_frac: float = None,
                 base_outdir: str = 'res/sat'):
        self.search = search
        self.timeout_s = timeout_s
        self.knn = knn
        self.lns = lns
        self.lns_iters = lns_iters
        self.destroy_frac = destroy_frac
        self.base_outdir = base_outdir
        import subprocess
        self.subprocess = subprocess

    def solve(self, instance_file: str, _: str):
        # Build command to call runner
        cmd = [sys.executable, '-m', 'models.sat.runner',
               f'--search={self.search}', f'--timeout={self.timeout_s}',
               f'--outdir={self.base_outdir}']
        if self.knn is not None:
            cmd.append(f'--knn={self.knn}')
        if self.lns:
            cmd.append('--lns')
            if self.lns_iters is not None:
                cmd.append(f'--lns-iters={self.lns_iters}')
            if self.destroy_frac is not None:
                cmd.append(f'--destroy-frac={self.destroy_frac}')
        # Ensure output dir exists
        os.makedirs(self.base_outdir, exist_ok=True)
        # Invoke runner
        print(f"[SATSolver] Running: {' '.join(cmd)}")
        result = self.subprocess.run(cmd, capture_output=True, text=True)
        # Relay output
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)



class UnifiedSolver:
    def __init__(self, solver_type, base_dir, outdir, search, timeout, knn, lns, lns_iters, destroy_frac):
        self.solver_type = solver_type.lower()
        self.instances_dir = os.path.join(base_dir, "Instances")
        self.output_dir = os.path.join(base_dir, outdir, solver_type)
        os.makedirs(self.output_dir, exist_ok=True)
        if self.solver_type == "cp":
            self.solver = CPSolver()
        elif self.solver_type == "mip":
            self.solver = MIPSolver()
        elif self.solver_type == "smt":
            from models.smt.SMT_Solver import SMT_Solver
            if smt_type not in ["sb","nosb"]:
                raise ValueError("Invalid SMT type. Choose 'sb' or 'nosb'.")
            self.solver = SMT_Solver(os.path.join(base_dir, "models",
                                                 "smt"), smt_type)
        elif self.solver_type == "sat":
            self.solver=SATSolver(search=search, timeout_s=timeout,
                                  knn=knn, lns=lns, lns_iters=lns_iters,
                                  destroy_frac=destroy_frac)
        else:
            raise ValueError("Invalid solver type: choose 'cp','mip','smt', or 'sat'.")

    def solve_all_instances(self):
            for fn in os.listdir(self.instances_dir):
                if fn.endswith(".dat"):
                    path = os.path.join(self.instances_dir, fn)
                    self.solver.solve(path, self.output_dir)


    
