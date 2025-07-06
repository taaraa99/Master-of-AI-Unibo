import math
import sys
import os
import json
import time
import multiprocessing
import queue
from datetime import datetime
from ortools.linear_solver import pywraplp

def log(message):
    """Helper function to print messages with a timestamp and process ID."""
    pid_str = f"Process {os.getpid()}"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{pid_str}] {message}", flush=True)

def read_instance(file_path):
    """
    This function loads an MCP instance from a .dat file.
    """
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if len(lines) < 4:
        raise ValueError("Instance file is incomplete.")
    m = int(lines[0])
    n = int(lines[1])
    capacities = list(map(int, lines[2].split()))
    item_sizes = list(map(int, lines[3].split()))
    raw = []
    for line in lines[4:]:
        row = list(map(int, line.split()))
        raw.append(row)
    if len(raw) != (n + 1):
        raise ValueError(f"Distance matrix must have (n+1) rows, but found {len(raw)}.")
        
    dist = [[0]*(n+1) for _ in range(n+1)]
    for j in range(1, n + 1):
        dist[0][j] = raw[n][j - 1]
    for i in range(1, n + 1):
        dist[i][0] = raw[i - 1][n]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                dist[i][j] = raw[i - 1][j - 1]
    return m, n, capacities, item_sizes, dist

def calculate_route_dist(route, dist):
    """Calculates the total distance of a single route."""
    if not route:
        return 0
    total_dist = dist[0][route[0]]
    for i in range(len(route) - 1):
        total_dist += dist[route[i]][route[i+1]]
    total_dist += dist[route[-1]][0]
    return total_dist

def compute_greedy_solution(m, n, capacities, item_sizes, dist):
    """
    An improved greedy insertion heuristic to find a good initial solution.
    """
    routes = [[] for _ in range(m)]
    remaining_caps = list(capacities)
    unassigned_items = list(range(1, n + 1))
    while unassigned_items:
        best_insertion, min_cost_increase = None, float('inf')
        for item_idx, item in enumerate(unassigned_items):
            item_size = item_sizes[item - 1]
            for i in range(m):
                if remaining_caps[i] >= item_size:
                    for pos in range(len(routes[i]) + 1):
                        original_dist = calculate_route_dist(routes[i], dist)
                        new_route = routes[i][:pos] + [item] + routes[i][pos:]
                        new_dist = calculate_route_dist(new_route, dist)
                        cost_increase = new_dist - original_dist
                        if cost_increase < min_cost_increase:
                            min_cost_increase = cost_increase
                            best_insertion = (item, item_idx, i, pos)
        if best_insertion is None:
            log("Warning: Could not assign all items in greedy heuristic.")
            break
        item_to_insert, item_list_idx, route_idx, pos_idx = best_insertion
        routes[route_idx].insert(pos_idx, item_to_insert)
        remaining_caps[route_idx] -= item_sizes[item_to_insert - 1]
        unassigned_items.pop(item_list_idx)
    final_route_dists = [calculate_route_dist(r, dist) for r in routes]
    return routes, max(final_route_dists) if final_route_dists else 0

def build_and_solve_mcp(m, n, capacities, item_sizes, dist, time_limit=300, approach="HiGHS"):
    """
    This function builds and solves the MCP using the MTZ formulation.
    The total time taken for this function is measured.
    """
    start_time = time.time()
    
    log("Creating HiGHS solver.")
    solver = pywraplp.Solver.CreateSolver("highs")
    if not solver:
        raise RuntimeError("Unable to create HiGHS solver with OR-Tools.")
    
    solver_params_str = "presolve=on,heuristics=on,parallel=on"
    solver.SetSolverSpecificParametersAsString(solver_params_str)
    log(f"Solver parameters set to: '{solver_params_str}'")
    solver.SetTimeLimit(time_limit * 1000)

    log("Defining decision variables...")
    a = {(i, j): solver.BoolVar(f"a_{i}_{j}") for i in range(m) for j in range(1, n + 1)}
    x = {(i, j, k): solver.BoolVar(f"x_{i}_{j}_{k}") for i in range(m) for j in range(n + 1) for k in range(n + 1) if j != k}
    used = [solver.BoolVar(f"used_{i}") for i in range(m)]
    y = [solver.NumVar(0, solver.infinity(), f"y_{i}") for i in range(m)]
    z = solver.NumVar(0, solver.infinity(), "z")
    u = {(i, j): solver.NumVar(0, n, f"u_{i}_{j}") for i in range(m) for j in range(1, n + 1)}
    log("Finished defining variables.")

    log("Adding core constraints...")
    for j in range(1, n + 1): solver.Add(solver.Sum(a[i, j] for i in range(m)) == 1)
    for i in range(m): solver.Add(solver.Sum(a[i, j] * item_sizes[j - 1] for j in range(1, n + 1)) <= capacities[i])
    for i in range(m):
        for j in range(1, n + 1):
            solver.Add(solver.Sum(x[i, j, k] for k in range(n + 1) if k != j) == a[i, j])
            solver.Add(solver.Sum(x[i, k, j] for k in range(n + 1) if k != j) == a[i, j])
    for i in range(m):
        solver.Add(solver.Sum(x[i, 0, k] for k in range(1, n + 1)) == used[i])
        solver.Add(solver.Sum(x[i, k, 0] for k in range(1, n + 1)) == used[i])
        for j in range(1, n + 1): solver.Add(used[i] >= a[i, j])
    for i in range(m): solver.Add(y[i] == solver.Sum(dist[j][k] * x[i, j, k] for j in range(n + 1) for k in range(n + 1) if j != k))
    for i in range(m): solver.Add(z >= y[i])
    
    # MTZ (Miller-Tucker-Zemlin) subtour elimination constraints.
    for i in range(m):
        for j in range(1, n + 1):
            solver.Add(u[i, j] >= a[i, j])
            solver.Add(u[i, j] <= n * a[i, j])
        for j in range(1, n + 1):
            for k in range(1, n + 1):
                if j != k:
                    solver.Add(u[(i, j)] - u[(i, k)] + n * x[(i, j, k)] <= n - 1)
    log("Finished adding core constraints.")

    if "WM" in approach:
        log("Computing greedy solution for warm start...")
        _, ub = compute_greedy_solution(m, n, capacities, item_sizes, dist)
        log(f"Greedy upper bound for warm start: {ub}")
        if ub is not None and ub > 0:
            solver.Add(z <= ub)

    log("--- Starting solver ---")
    solver.Minimize(z)
    status = solver.Solve()
    log("--- Solver finished ---")

    total_time_seconds = time.time() - start_time

    solution = {
        "time": round(total_time_seconds, 2),
        "optimal": (status == pywraplp.Solver.OPTIMAL),
        "obj": -1.0,
        "sol": []
    }

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        final_routes = []
        for i in range(m):
            if used[i].solution_value() > 0.5:
                route, current = [], 0
                while True:
                    next_node = -1
                    for k in range(n + 1):
                        if k != current and x[i, current, k].solution_value() > 0.5:
                            next_node = k
                            break
                    if next_node <= 0: break
                    route.append(next_node)
                    current = next_node
                final_routes.append(route)
            else:
                final_routes.append([])
        solution["sol"] = final_routes
        
        # Recalculate objective from the actual solution routes to ensure consistency
        if any(final_routes):
            route_distances = [calculate_route_dist(r, dist) for r in final_routes]
            max_distance = max(route_distances)
            solution["obj"] = round(max_distance, 5)
        else:
            solution["obj"] = 0.0
    
    return {approach.upper(): solution}

def solve_process_wrapper(result_queue, *args):
    """A wrapper to run the solver in a separate process and handle exceptions."""
    try:
        solution = build_and_solve_mcp(*args)
        result_queue.put(solution)
    except Exception as e:
        log(f"Encountered an exception: {e}")
        # Pass the exception back to the main process
        result_queue.put(e)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mcp_solver.py <instance.dat>")
        sys.exit(1)
        
    instance_file = sys.argv[1]
    if not os.path.exists(instance_file):
        print(f"Error: Instance file not found at '{instance_file}'")
        sys.exit(1)

    TOTAL_TIMEOUT = 300  # 5 minutes

    try:
        log(f"Reading instance file: {instance_file}")
        m, n, capacities, item_sizes, dist = read_instance(instance_file)
        log(f"Instance loaded: m={m}, n={n}")
        
        approach_str = "HiGHS+WM"
        
        # Prepare for multiprocessing
        result_queue = multiprocessing.Queue()
        solver_args = (m, n, capacities, item_sizes, dist, TOTAL_TIMEOUT, approach_str)
        
        p = multiprocessing.Process(target=solve_process_wrapper, args=(result_queue,) + solver_args)
        
        log(f"Starting solver process with a total timeout of {TOTAL_TIMEOUT} seconds...")
        p.start()
        
        # Wait for the process to finish or timeout
        p.join(TOTAL_TIMEOUT)
        
        result = None
        if p.is_alive():
            log(f"Total time limit of {TOTAL_TIMEOUT} seconds exceeded. Terminating process.")
            p.terminate()
            p.join()
            
            # Create a timeout result
            result = {
                approach_str.upper(): {
                    "time": TOTAL_TIMEOUT, "optimal": False, "obj": -1.0, 
                    "sol": [], "status": "Total timeout exceeded"
                }
            }
        else:
            log("Process finished within the time limit.")
            # Get the result from the queue
            output = result_queue.get()
            if isinstance(output, Exception):
                # Re-raise the exception caught in the child process
                raise output
            result = output

        print("\n--- Solution ---")
        print(json.dumps(result, indent=4))

    except (ValueError, RuntimeError, queue.Empty) as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
