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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def read_instance(file_path):
    """
    This function loads an MCP instance from a .dat file.
    Here, node 0 is used as the starting (origin) point, while nodes 1 through n represent the items.
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
    # Read the distance matrix components
    for line in lines[4:]:
        row = list(map(int, line.split()))
        raw.append(row)
    if len(raw) != (n + 1):
        raise ValueError(f"Distance matrix must have (n+1) rows, but found {len(raw)}.")
        
    # Construct an (n+1)x(n+1) matrix to hold the distances
    dist = [[0]*(n+1) for _ in range(n+1)]
    # Distances from origin (0) to items (1..n)
    for j in range(1, n + 1):
        dist[0][j] = raw[n][j - 1]
    # Distances from items (1..n) to origin (0)
    for i in range(1, n + 1):
        dist[i][0] = raw[i - 1][n]
    # Distances between items
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                dist[i][j] = raw[i - 1][j - 1]
    return m, n, capacities, item_sizes, dist

def compute_greedy_solution(m, n, capacities, item_sizes, dist):
    """
    This function applies a basic greedy strategy to find an initial solution.
    It returns a tuple containing the routes found and the maximum distance among those routes.
    """
    assigned = [False] * (n + 1)
    routes = [[] for _ in range(m)]
    route_dists = [0.0] * m
    rem_caps = capacities[:]
    
    for i in range(m):
        current = 0  # Begin at the origin
        while True:
            best_item = None
            # Find the closest unassigned item that fits in the current courier's capacity
            best_dist = float('inf')
            for j in range(1, n + 1):
                if (not assigned[j]) and (item_sizes[j - 1] <= rem_caps[i]):
                    d = dist[current][j]
                    if d < best_dist:
                        best_dist = d
                        best_item = j
            
            if best_item is None:
                route_dists[i] += dist[current][0]  # Go back to the origin
                break
            
            # Assign the best item found
            routes[i].append(best_item)
            assigned[best_item] = True
            route_dists[i] += dist[current][best_item]
            rem_caps[i] -= item_sizes[best_item - 1]
            current = best_item
            
    # Assign any remaining unassigned items to the courier with the most remaining capacity
    for j in range(1, n + 1):
        if not assigned[j]:
            idx = max(range(m), key=lambda i: rem_caps[i])
            if item_sizes[j - 1] <= rem_caps[idx]:
                routes[idx].append(j)
                assigned[j] = True

    # Recalculate final route distances based on the full routes
    final_route_dists = [0.0] * m
    for i in range(m):
        if routes[i]:
            d = dist[0][routes[i][0]] # Origin to first item
            for k in range(len(routes[i]) - 1):
                d += dist[routes[i][k]][routes[i][k+1]]
            d += dist[routes[i][-1]][0] # Last item to origin
            final_route_dists[i] = d

    return routes, max(final_route_dists) if final_route_dists else 0


def build_and_solve_mcp(m, n, capacities, item_sizes, dist, time_limit=300, approach="HiGHS"):
    """
    This function builds and solves the MCP using an OR-Tools mixed integer programming solver.
    """
    log("Creating HiGHS solver.")
    solver = pywraplp.Solver.CreateSolver("highs")
    if not solver:
        raise RuntimeError("Unable to create HiGHS solver with OR-Tools.")
    
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
    for j in range(1, n + 1):
        solver.Add(solver.Sum(a[i, j] for i in range(m)) == 1, name=f"assign_item_{j}")
    for i in range(m):
        solver.Add(solver.Sum(a[i, j] * item_sizes[j - 1] for j in range(1, n + 1)) <= capacities[i], name=f"capacity_{i}")
    for i in range(m):
        for j in range(1, n + 1):
            solver.Add(solver.Sum(x[i, j, k] for k in range(n + 1) if k != j) == a[i, j], name=f"flow_out_{i}_{j}")
            solver.Add(solver.Sum(x[i, k, j] for k in range(n + 1) if k != j) == a[i, j], name=f"flow_in_{i}_{j}")
    for i in range(m):
        solver.Add(solver.Sum(x[i, 0, k] for k in range(1, n + 1)) == used[i], name=f"start_at_origin_{i}")
        solver.Add(solver.Sum(x[i, k, 0] for k in range(1, n + 1)) == used[i], name=f"end_at_origin_{i}")
        for j in range(1, n + 1):
            solver.Add(used[i] >= a[i, j], name=f"used_logic_{i}_{j}")
    for i in range(m):
        expr = solver.Sum(dist[j][k] * x[i, j, k] for j in range(n + 1) for k in range(n + 1) if j != k)
        solver.Add(y[i] == expr, name=f"route_dist_{i}")
    for i in range(m):
        solver.Add(z >= y[i], name=f"objective_link_{i}")
    for i in range(m):
        for j in range(1, n + 1):
            solver.Add(u[i, j] >= a[i, j])
            solver.Add(u[i, j] <= n * a[i, j])
            for k in range(1, n + 1):
                if j != k:
                    solver.Add(u[i, j] - u[i, k] + n * x[i, j, k] <= n - 1, name=f"mtz_{i}_{j}_{k}")
    log("Finished adding core constraints.")

    log("Adding optional enhancements...")
    if "SB" in approach:
        for i in range(1, m):
            solver.Add(solver.Sum(j * a[i - 1, j] for j in range(1, n + 1)) <= solver.Sum(j * a[i, j] for j in range(1, n + 1)), name=f"symmetry_break_{i}")
    if "CUT" in approach and n > 1:
        b = [solver.BoolVar(f"b_{i}") for i in range(m)]
        L_direct = min(dist[j][k] for j in range(1, n+1) for k in range(1, n+1) if j != k)
        L_round = min(dist[0][j] + dist[j][0] for j in range(1, n+1))
        extra_bound = L_direct + L_round
        for i in range(m):
            solver.Add(solver.Sum(a[i,j] for j in range(1, n+1)) - 1 >= b[i])
            solver.Add(solver.Sum(a[i,j] for j in range(1, n+1)) - 1 <= n * b[i])
            solver.Add(y[i] >= extra_bound * b[i])
    if "IMPLIED" in approach:
        if any(dist[0][j] + dist[j][0] > 0 for j in range(1, n+1)):
             max_roundtrip = max(dist[0][j] + dist[j][0] for j in range(1, n+1))
             solver.Add(z >= max_roundtrip / 2.0)
    if "WM" in approach:
        log("Computing greedy solution for warm start...")
        _, ub = compute_greedy_solution(m, n, capacities, item_sizes, dist)
        log(f"Greedy upper bound for warm start: {ub}")
        if ub is not None and ub > 0:
            solver.Add(z <= ub)
    log("Finished adding optional enhancements.")

    log("--- Starting solver ---")
    solver.Minimize(z)
    status = solver.Solve()
    log("--- Solver finished ---")

    log("Processing solution...")
    solution = {
        "time": int(solver.WallTime() / 1000.0),
        "optimal": (status == pywraplp.Solver.OPTIMAL),
        "obj": -1.0,
        "sol": []
    }

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        solution["obj"] = round(solver.Objective().Value(), 5)
        final_routes = []
        for i in range(m):
            if used[i].solution_value() > 0.5:
                route = []
                current = 0
                while True:
                    next_node = -1
                    for k in range(n + 1):
                        if k != current and x[i, current, k].solution_value() > 0.5:
                            next_node = k
                            break
                    if next_node <= 0:
                        break
                    route.append(next_node)
                    current = next_node
                final_routes.append(route)
            else:
                final_routes.append([])
        solution["sol"] = final_routes
    log("Finished processing solution.")
    
    return {approach.upper(): solution}

def solve_process_wrapper(result_queue, m, n, capacities, item_sizes, dist, time_limit, approach):
    """A wrapper to run the solver and put the result in a queue."""
    try:
        solution = build_and_solve_mcp(m, n, capacities, item_sizes, dist, time_limit, approach)
        result_queue.put(solution)
    except Exception as e:
        # Pass any exception back to the main process
        result_queue.put(e)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mcp_ortools_model.py <instance.dat>")
        sys.exit(1)
        
    instance_file = sys.argv[1]
    if not os.path.exists(instance_file):
        print(f"Error: Instance file not found at '{instance_file}'")
        sys.exit(1)

    TOTAL_TIMEOUT = 300  # 5 minutes total

    try:
        log(f"Reading instance file: {instance_file}")
        m, n, capacities, item_sizes, dist = read_instance(instance_file)
        log(f"Instance loaded: m={m}, n={n}")
        
        approach_str = "HiGHS+SB+IMPLIED+WM+CUT"
        
        result_queue = multiprocessing.Queue()
        
        # Create and start the solver process
        p = multiprocessing.Process(
            target=solve_process_wrapper,
            args=(result_queue, m, n, capacities, item_sizes, dist, TOTAL_TIMEOUT, approach_str)
        )
        
        log(f"Starting solver process with a total timeout of {TOTAL_TIMEOUT} seconds...")
        p.start()
        
        # Wait for the process to finish, with a timeout
        p.join(TOTAL_TIMEOUT)
        
        # Check if the process is still running
        if p.is_alive():
            log(f"Total time limit of {TOTAL_TIMEOUT} seconds exceeded. Terminating process.")
            p.terminate()
            p.join() # Wait for the process to terminate
            
            # Create a timeout result
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
            # Get the result from the queue
            result = result_queue.get()
            if isinstance(result, Exception):
                raise result # Re-raise exception from the child process
            
            print("\n--- Solution ---")
            print(json.dumps(result, indent=4))

    except (ValueError, RuntimeError, queue.Empty) as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
