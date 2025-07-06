import math
import sys
import os
import json
from datetime import datetime
from ortools.linear_solver import pywraplp
import multiprocessing
import queue

def log(message):
    """Helper function to print messages with a timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

def read_instance(file_path):
    """
    Loads an MCP instance from a .dat file.
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
        raise ValueError("Distance matrix must have (n+1) rows.")

    dist = [[0]*(n+1) for _ in range(n+1)]
    for j in range(1, n+1):
        dist[0][j] = raw[n][j-1]
    for i in range(1, n+1):
        dist[i][0] = raw[i-1][n]
    for i in range(1, n+1):
        for j in range(1, n+1):
            dist[i][j] = 0 if i == j else raw[i-1][j-1]
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
    An improved greedy insertion heuristic to find a good initial solution for the warm start.
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

def build_and_solve_mcp(m, n, capacities, item_sizes, dist, time_limit=300, approach="CBC"):
    """
    Builds and solves the MCP using either CBC or SCIP from OR-Tools.
    """
    log(f"Creating {approach.upper()} solver.")
    solver_type_str = approach.split('+')[0].upper()
    if solver_type_str == "CBC":
        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    elif solver_type_str == "SCIP":
        solver = pywraplp.Solver.CreateSolver("SCIP_MIXED_INTEGER_PROGRAMMING")
    else:
        raise ValueError("Unsupported approach: choose CBC or SCIP based approach")
    if not solver:
        raise RuntimeError(f"Unable to create {solver_type_str} solver.")

    solver.SetTimeLimit(time_limit * 1000)

    log("Defining decision variables...")
    a = {(i, j): solver.BoolVar(f"a_{i}_{j}") for i in range(m) for j in range(1, n+1)}
    x = {(i, j, k): solver.BoolVar(f"x_{i}_{j}_{k}") for i in range(m) for j in range(n+1) for k in range(n+1) if j != k}
    used = [solver.BoolVar(f"used_{i}") for i in range(m)]
    y = [solver.NumVar(0, solver.infinity(), f"y_{i}") for i in range(m)]
    z = solver.NumVar(0, solver.infinity(), "z")
    u = {(i, j): solver.NumVar(0, n, f"u_{i}_{j}") for i in range(m) for j in range(1, n+1)}
    log("Finished defining variables.")
    
    log("Adding core constraints...")
    for j in range(1, n+1): solver.Add(solver.Sum(a[(i, j)] for i in range(m)) == 1)
    for i in range(m): solver.Add(solver.Sum(a[(i, j)] * item_sizes[j-1] for j in range(1, n+1)) <= capacities[i])
    for i in range(m):
        for j in range(1, n+1):
            solver.Add(solver.Sum(x[(i, j, k)] for k in range(n+1) if k != j) == a[(i, j)])
            solver.Add(solver.Sum(x[(i, k, j)] for k in range(n+1) if k != j) == a[(i, j)])
    for i in range(m):
        solver.Add(solver.Sum(x[(i, 0, k)] for k in range(1, n+1)) == used[i])
        solver.Add(solver.Sum(x[(i, k, 0)] for k in range(1, n+1)) == used[i])
        for j in range(1, n+1): solver.Add(used[i] >= a[(i, j)])
    for i in range(m): solver.Add(y[i] == solver.Sum(dist[j][k] * x[(i, j, k)] for j in range(n+1) for k in range(n+1) if j != k))
    for i in range(m): solver.Add(z >= y[i])
    for i in range(m):
        for j in range(1, n+1):
            solver.Add(u[(i, j)] >= a[(i, j)])
            solver.Add(u[(i, j)] <= n * a[(i, j)])
            for k in range(1, n+1):
                if j != k: solver.Add(u[(i, j)] - u[(i, k)] + n * x[(i, j, k)] <= n - 1)
    log("Finished adding core constraints.")

    #Symmetry breaking
    for i2 in range(1, m):
        solver.Add(solver.Sum(j * a[(i2 - 1, j)] for j in range(1, n+1)) <= solver.Sum(j * a[(i2, j)] for j in range(1, n+1)))
    

    #Implied constraints
    if any(dist[0][j] + dist[j][0] > 0 for j in range(1, n+1)):
        max_roundtrip = max(dist[0][j] + dist[j][0] for j in range(1, n+1))
        solver.Add(z >= max_roundtrip / 2.0)

    # #adding greedy upperbound
    # _, ub = compute_greedy_solution(m, n, capacities, item_sizes, dist)
    # if ub is not None and ub > 0:
    #         solver.Add(z <= ub)


    solver.Minimize(z)
    log("--- Starting solver ---")
    status = solver.Solve()
    log("--- Solver finished ---")

    solution = { "time": int(solver.WallTime() / 1000.0), "optimal": (status == solver.OPTIMAL), "obj": -1.0, "sol": [] }
    if status in [solver.OPTIMAL, solver.FEASIBLE]:
        solution["obj"] = round(solver.Objective().Value(), 5)
        final_routes = []
        for i in range(m):
            if used[i].solution_value() > 0.5:
                route, current = [], 0
                while True:
                    next_node = -1
                    for kk in range(n+1):
                        if kk != current and x[(i, current, kk)].solution_value() > 0.5:
                            next_node = kk
                            break
                    if next_node <= 0: break
                    route.append(next_node)
                    current = next_node
                final_routes.append(route)
            else:
                final_routes.append([])
        solution["sol"] = final_routes
    return { approach.upper(): solution }

def solve_process_wrapper(result_queue, m, n, capacities, item_sizes, dist, time_limit, approach):
    """A wrapper to run the solver and put the result in a queue."""
    try:
        solution = build_and_solve_mcp(m, n, capacities, item_sizes, dist, time_limit, approach)
        result_queue.put(solution)
    except Exception as e:
        result_queue.put(e)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <instance.dat>")
        sys.exit(1)
        
    instance_file = sys.argv[1]
    if not os.path.exists(instance_file):
        print(f"Error: Instance file not found at '{instance_file}'")
        sys.exit(1)

    TOTAL_TIMEOUT = 300

    try:
        log(f"Reading instance file: {instance_file}")
        m, n, capacities, item_sizes, dist = read_instance(instance_file)
        log(f"Instance loaded: m={m}, n={n}")
        
        # Use all enhancements including the new Warm Start (WM)
        approach_str = "SCIP+SB+IMPLIED+WM"
        
        result_queue = multiprocessing.Queue()
        
        p = multiprocessing.Process(
            target=solve_process_wrapper,
            args=(result_queue, m, n, capacities, item_sizes, dist, TOTAL_TIMEOUT, approach_str)
        )
        
        log(f"Starting solver process with a total timeout of {TOTAL_TIMEOUT} seconds...")
        p.start()
        p.join(TOTAL_TIMEOUT)
        
        if p.is_alive():
            log(f"Total time limit of {TOTAL_TIMEOUT} seconds exceeded. Terminating process.")
            p.terminate()
            p.join()
            
            timeout_solution = {
                approach_str.upper(): {
                    "time": TOTAL_TIMEOUT,
                    "optimal": False,
                    "obj": -1.0,
                    "sol": [],
                    "status": "Total timeout exceeded"
                }
            }
            print("\n--- Solution ---")
            print(json.dumps(timeout_solution, indent=4))
        else:
            log("Process finished within the time limit.")
            result = result_queue.get()
            if isinstance(result, Exception):
                raise result
            
            print("\n--- Solution ---")
            print(json.dumps(result, indent=4))

    except (ValueError, RuntimeError, queue.Empty) as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
