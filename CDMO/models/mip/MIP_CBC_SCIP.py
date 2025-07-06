import math
import sys
import os
import json
import time
import multiprocessing
import queue
from ortools.linear_solver import pywraplp

def read_instance(file_path):
    """
    Loads an MCP instance from a .dat file.
    Here, node 0 is treated as the starting point (origin),
    while nodes 1 through n represent the items.
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

    # Construct a complete distance matrix with (n+1) rows and columns.
    dist = [[0]*(n+1) for _ in range(n+1)]
    for j in range(1, n+1):
        dist[0][j] = raw[n][j-1]
    for i in range(1, n+1):
        dist[i][0] = raw[i-1][n]
    for i in range(1, n+1):
        for j in range(1, n+1):
            dist[i][j] = 0 if i == j else raw[i-1][j-1]
    return m, n, capacities, item_sizes, dist

def compute_greedy_solution(m, n, capacities, item_sizes, dist):
    """
    Implements a straightforward nearest-neighbor heuristic to generate an upper bound.
    Returns a tuple containing the set of routes and the maximum distance encountered.
    """
    assigned = [False] * (n+1)
    routes = [[] for _ in range(m)]
    route_dists = [0.0] * m
    rem_caps = capacities[:]
    for i in range(m):
        current = 0  # Begin at the origin
        while True:
            best_item = None
            best_dist = float('inf')
            for j in range(1, n+1):
                if (not assigned[j]) and (item_sizes[j-1] <= rem_caps[i]):
                    d = dist[current][j]
                    if d < best_dist:
                        best_dist = d
                        best_item = j
            if best_item is None:
                route_dists[i] += dist[current][0]  # Return to the origin
                break
            routes[i].append(best_item)
            assigned[best_item] = True
            route_dists[i] += dist[current][best_item]
            rem_caps[i] -= item_sizes[best_item-1]
            current = best_item
    for j in range(1, n+1):
        if not assigned[j]:
            idx = max(range(m), key=lambda i: rem_caps[i])
            routes[idx].append(j)
            route_dists[idx] += dist[0][j] + dist[j][0]
            assigned[j] = True
    return routes, max(route_dists)

def build_and_solve_mcp(m, n, capacities, item_sizes, dist, time_limit=300, approach="CBC"):
    """
    Builds and solves the MCP using an OR‑Tools MIP formulation.
    The total execution time is measured, and the solver's internal
    time limit is used as a hint.
    """
    start_time = time.time()

    solver_choice = approach.upper()
    if "CBC" in solver_choice:
        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    elif "SCIP" in solver_choice:
        solver = pywraplp.Solver.CreateSolver("SCIP_MIXED_INTEGER_PROGRAMMING")
    else:
        raise ValueError("Unsupported approach: choose CBC or SCIP")
    if not solver:
        raise RuntimeError(f"Unable to create {solver_choice} solver with OR-Tools.")

    # Set the solver's internal time limit. This is a general method.
    solver.set_time_limit(time_limit * 1000)

    # For SCIP, we can set specific parameters for more control.
    # CBC does not support SetSolverSpecificParametersAsString, so we skip it.
    if "SCIP" in solver_choice:
        try:
            # Correct parameter for SCIP time limit is 'limits/time'
            scip_params = f"limits/time = {time_limit}"
            solver.SetSolverSpecificParametersAsString(scip_params)
        except Exception as e:
            # Silently ignore if parameters can't be set
            pass

    # Optionally, compute a greedy solution to get an initial upper bound.
    routes, ub = None, None
    try:
        routes, ub = compute_greedy_solution(m, n, capacities, item_sizes, dist)
    except Exception as e:
        # Silently ignore if greedy approach fails
        pass

    # Define the decision variables for the optimization model.
    # a[i, j] is 1 if item j is assigned to courier i, 0 otherwise.
    a = {(i, j): solver.BoolVar(f"a_{i}_{j}") for i in range(m) for j in range(1, n + 1)}

    # x[i, j, k] is 1 if courier i travels from node j to node k, 0 otherwise.
    x = {(i, j, k): solver.BoolVar(f"x_{i}_{j}_{k}") for i in range(m) for j in range(n + 1) for k in range(n + 1) if j != k}

    # used[i] is 1 if courier i is used (has at least one item), 0 otherwise.
    used = [solver.BoolVar(f"used_{i}") for i in range(m)]
    # y[i] is the total distance of the route for courier i.
    y = [solver.NumVar(0, solver.infinity(), f"y_{i}") for i in range(m)]
    # z is the maximum route distance among all couriers (the objective).
    z = solver.NumVar(0, solver.infinity(), "z")
    if ub is not None:
        solver.Add(z <= ub)

    # u[i, j] are MTZ variables for subtour elimination.
    u = {(i, j): solver.NumVar(0, n, f"u_{i}_{j}") for i in range(m) for j in range(1, n + 1)}

    # --- CONSTRAINTS ---

    # (A) Each item must be assigned to exactly one courier.
    for j in range(1, n + 1):
        solver.Add(solver.Sum(a[(i, j)] for i in range(m)) == 1)

    # (B) The total size of items for each courier must not exceed their capacity.
    for i in range(m):
        solver.Add(solver.Sum(a[(i, j)] * item_sizes[j - 1] for j in range(1, n + 1)) <= capacities[i])

    # (C) Flow conservation constraints to link assignment (a) and routing (x).
    # If an item j is assigned to courier i, there must be one arc entering and one leaving j for that courier.
    for i in range(m):
        for j in range(1, n + 1):
            solver.Add(solver.Sum(x[(i, j, k)] for k in range(n + 1) if k != j) == a[(i, j)])
            solver.Add(solver.Sum(x[(i, k, j)] for k in range(n + 1) if k != j) == a[(i, j)])

    # (D) Route start/end constraints. If a courier is used, their route must start and end at the origin (0).
    for i in range(m):
        solver.Add(solver.Sum(x[(i, 0, k)] for k in range(1, n + 1)) == used[i])
        solver.Add(solver.Sum(x[(i, k, 0)] for k in range(1, n + 1)) == used[i])
        # Link the 'used' variable to the assignment variables.
        for j in range(1, n + 1):
            solver.Add(used[i] >= a[(i, j)])

    # (E) Calculate the total distance for each courier's route.
    for i in range(m):
        solver.Add(y[i] == solver.Sum(dist[j][k] * x[(i, j, k)]
                                     for j in range(n + 1) for k in range(n + 1) if j != k))

    # (F) The objective variable z must be greater than or equal to each courier's route distance.
    for i in range(m):
        solver.Add(z >= y[i])

    # (G) MTZ (Miller-Tucker-Zemlin) subtour elimination constraints.
    for i in range(m):
        for j in range(1, n + 1):
            solver.Add(u[(i, j)] >= a[(i, j)])
            solver.Add(u[(i, j)] <= n * a[(i, j)])
        for j_inner in range(1, n + 1):
            for k_inner in range(1, n + 1):
                if j_inner != k_inner:
                    solver.Add(u[(i, j_inner)] - u[(i, k_inner)] + n * x[(i, j_inner, k_inner)] <= n - 1)
    
    # (H) Optional: Symmetry-breaking constraints.
    if "SB" in approach.upper():
        for i2 in range(1, m):
            i1 = i2 - 1
            solver.Add(solver.Sum(j * a[(i1, j)] for j in range(1, n + 1))
                       <= solver.Sum(j * a[(i2, j)] for j in range(1, n + 1)))

    # (I) Optional: Implied constraints for a lower bound on the objective.
    if "IMPLIED" in approach.upper():
        max_roundtrip = max(dist[0][j] + dist[j][0] for j in range(1, n + 1))
        solver.Add(z >= max_roundtrip / 2.0)
    
    # Set the objective to minimize the maximum route distance (z).
    solver.Minimize(z)
    status = solver.Solve()

    total_time_seconds = time.time() - start_time

    solution = {
        "time": int(total_time_seconds),
        "optimal": (status == pywraplp.Solver.OPTIMAL),
        "obj": None,
        "sol": []
    }

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        final_routes = []
        for i in range(m):
            route = []
            if used[i].solution_value() > 0.5:
                current = 0
                while True:
                    next_node = None
                    for kk in range(n+1):
                        if kk != current and x[(i, current, kk)].solution_value() > 0.5:
                            next_node = kk
                            break
                    if (not next_node) or next_node == 0:
                        break
                    route.append(next_node)
                    current = next_node
            final_routes.append(route)
        solution["sol"] = final_routes
        raw_obj = max(y[i].solution_value() for i in range(m))
        if abs(raw_obj - round(raw_obj)) < 1e-7:
            raw_obj = round(raw_obj)
        solution["obj"] = float(raw_obj)
    else:
        solution["sol"] = [[] for _ in range(m)]
        solution["obj"] = -1
        
    return { approach.upper(): solution }

def solve_process_wrapper(result_queue, *args):
    """A wrapper to run the solver in a separate process and handle exceptions."""
    try:
        solution = build_and_solve_mcp(*args)
        result_queue.put(solution)
    except Exception as e:
        # Pass the exception back to the main process
        result_queue.put(e)

if __name__ == "__main__":
    # This is required for multiprocessing to work correctly on Windows
    # when the script is frozen into an executable.
    multiprocessing.freeze_support()

    if len(sys.argv) < 2:
        # The original code printed a usage message here.
        # It has been removed as requested.
        sys.exit(1)
        
    instance_file = sys.argv[1]
    # Default to CBC if no approach is specified
    approach_str = sys.argv[2] if len(sys.argv) > 2 else "CBC"

    if not os.path.exists(instance_file):
        # The original code printed an error message here.
        # It has been removed as requested.
        sys.exit(1)

    TOTAL_TIMEOUT = 300  # 5 minutes

    try:
        m, n, capacities, item_sizes, dist = read_instance(instance_file)
        
        # Prepare for multiprocessing
        result_queue = multiprocessing.Queue()
        solver_args = (m, n, capacities, item_sizes, dist, TOTAL_TIMEOUT, approach_str)
        
        p = multiprocessing.Process(target=solve_process_wrapper, args=(result_queue,) + solver_args)
        
        p.start()
        
        # Wait for the process to finish or timeout
        p.join(TOTAL_TIMEOUT)
        
        result = None
        if p.is_alive():
            p.terminate() # First, try a graceful shutdown
            p.join(1) # Wait a moment for it to close
            if p.is_alive():
                p.kill() # Forcefully kill the process
                p.join()
            
            # Create a timeout result
            result = {
                approach_str.upper(): {
                    "time": TOTAL_TIMEOUT, "optimal": False, "obj": -1.0, 
                    "sol": [], "status": "Total timeout exceeded"
                }
            }
        else:
            # Get the result from the queue
            output = result_queue.get()
            if isinstance(output, Exception):
                # Re-raise the exception caught in the child process
                raise output
            result = output

        # The final solution is now in the 'result' variable.
        # The original code printed it to the console.
        # To use the result, you would now process the 'result' dictionary.
        # For example, you could write it to a file:
        # with open("solution.json", "w") as f:
        #     json.dump(result, f, indent=4)


    except (ValueError, RuntimeError, queue.Empty) as e:
        # Error handling without printing
        sys.exit(1)
    except Exception as e:
        # Error handling without printing
        sys.exit(1)
