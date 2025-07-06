import math
import sys
import os
import json
import time
import multiprocessing
import queue
from datetime import datetime
from ortools.linear_solver import pywraplp

# --- Utility Functions ---
# these are some helper functions to make our life easier.

def log(message):
    """
    A simple hepler function to print messages with a timestamp and the process ID.
    its really handy when were running things in parallel to see who is saying what.
    """
    pid_str = f"Process {os.getpid()}"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{pid_str}] {message}", flush=True)

def read_instance(file_path):
    """
    This function is our file reader, it opens up the problem files (the .dat ones),
    which describe the specific multiple courier problem we need to solve. It reads
    teh number of couriers, items, there capacities and sizes, and the all-important
    distance matrix.
    """
    with open(file_path, "r") as f:
        # We read the lines, skipping any empty ones or comments (lines starting with #)
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    # a quick check to make sure teh file isnt obviously broken.
    if len(lines) < 4:
        raise ValueError("Instance file is incomplete.")
    
    # lets grab the basic info:
    m = int(lines[0])  # Number of couriers
    n = int(lines[1])  # Number of items (or locations to visit)
    capacities = list(map(int, lines[2].split()))  # How much each courier can carry
    item_sizes = list(map(int, lines[3].split()))   # How big each item is
    
    # Now, lets tackle the distance matrix. Its a bit tricky cause of its format.
    raw = []
    for line in lines[4:]:
        row = list(map(int, line.split()))
        raw.append(row)
    
    if len(raw) != (n + 1):
        raise ValueError(f"Distance matrix must have (n+1) rows, but found {len(raw)}.")
        
    # Well create a clean, square distance matrix 'dist'.
    # Location 0 is the depot, an locations 1 to n is the customers.
    dist = [[0]*(n+1) for _ in range(n+1)]
    
    # the original format is a bit funky, so were re-organizing it.
    # distances from the depot (node 0) to customers
    for j in range(1, n + 1):
        dist[0][j] = raw[n][j - 1]
    # distances from customers back to the depot
    for i in range(1, n + 1):
        dist[i][0] = raw[i - 1][n]
    # Distances between customers
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                dist[i][j] = raw[i - 1][j - 1]
                
    return m, n, capacities, item_sizes, dist

def calculate_route_dist(route, dist):
    """
    Given a single couriers route (a list of customer locations), this function
    calculate its total length. It start at the depot (0), go through the
    locations in order, and finaly returns to the depot.
    """
    # if the courier has no deliverys, they dont travel at all.
    if not route:
        return 0
    
    # start the journey from the depot to the first customer.
    total_dist = dist[0][route[0]]
    # add the distances between each customer on the rout.
    for i in range(len(route) - 1):
        total_dist += dist[route[i]][route[i+1]]
    # and finaly, the trip from the last customer back to the depot.
    total_dist += dist[route[-1]][0]
    return total_dist

def compute_greedy_solution(m, n, capacities, item_sizes, dist):
    """
    This is a "greedy" way to get a pretty good, but not neccessarily perfect, solution.
    Its like building the routes one step at a time, always making the choice that look
    best at that very moment (the "cheapest" insertion). This helps give our main
    solver a good starting point.
    """
    routes = [[] for _ in range(m)]  # empty routs for each courier
    remaining_caps = list(capacities) # keep track of how much space is left
    unassigned_items = list(range(1, n + 1)) # all items that need a home
    
    # keep going until every item is assigned to a courier.
    while unassigned_items:
        best_insertion, min_cost_increase = None, float('inf')
        
        # Let's check every unassigned item...
        for item_idx, item in enumerate(unassigned_items):
            item_size = item_sizes[item - 1]
            # ...and try to fit it into every courier's route...
            for i in range(m):
                # ...if they have enough capacity...
                if remaining_caps[i] >= item_size:
                    # ...at every possible position in their current route.
                    for pos in range(len(routes[i]) + 1):
                        original_dist = calculate_route_dist(routes[i], dist)
                        
                        # Let's see what the new route would look like.
                        new_route = routes[i][:pos] + [item] + routes[i][pos:]
                        new_dist = calculate_route_dist(new_route, dist)
                        
                        cost_increase = new_dist - original_dist
                        
                        # if this is the best move weve seen so far, lets remember it.
                        if cost_increase < min_cost_increase:
                            min_cost_increase = cost_increase
                            best_insertion = (item, item_idx, i, pos)
    
        # if we couldnt find a place for any item, somthing is wrong.
        if best_insertion is None:
            break
        
        # we found the best move! lets make it happen.
        item_to_insert, item_list_idx, route_idx, pos_idx = best_insertion
        routes[route_idx].insert(pos_idx, item_to_insert)
        remaining_caps[route_idx] -= item_sizes[item_to_insert - 1]
        unassigned_items.pop(item_list_idx)
        
    # once were done, calculate the final distances. the goal is to minimize the longest route.
    final_route_dists = [calculate_route_dist(r, dist) for r in routes]
    return routes, max(final_route_dists) if final_route_dists else 0


# --- The Main Solver ---
# this is where the magic happens. we define the mathmatical model for the problem.

def build_and_solve_mcp(m, n, capacities, item_sizes, dist, time_limit=300, approach="HiGHS"):
    """
    This is the core of our program It builds the mathmatical model for the
    Multiple Courier Problem using the Miller-Tucker-Zemlin (MTZ) formulation
    and then asks the OR-Tools solver (HiGHS) to find the best solution.
    """
    start_time = time.time()
    
    # --- 1. Setting up the Solver ---
    solver = pywraplp.Solver.CreateSolver("highs")
    if not solver:
        raise RuntimeError("Error")
    
    # Give the solver some instructions to help it run faster.
    solver_params_str = "presolve=on,heuristics=on,parallel=on"
    solver.SetSolverSpecificParametersAsString(solver_params_str)
    # and dont let it run forever! well give it a time limit.
    solver.SetTimeLimit(time_limit * 1000) # The solver wants milliseconds.

    # --- 2. Defining our Variables ---
    # These are the "decisions" the solver need to make.
    
    # `a[i, j]` is 1 if courier `i` is assigned to deliver item `j`, and 0 otherwise. simple.
    a = {(i, j): solver.BoolVar(f"a_{i}_{j}") for i in range(m) for j in range(1, n + 1)}
    
    # `x[i, j, k]` is 1 if courier `i` travels direct from location `j` to `k`, and 0 otherwise
    x = {(i, j, k): solver.BoolVar(f"x_{i}_{j}_{k}") for i in range(m) for j in range(n + 1) for k in range(n + 1) if j != k}
    
    # `used[i]` is 1 if courier `i` is used at all (ie has at least one delivery).
    used = [solver.BoolVar(f"used_{i}") for i in range(m)]
    
    # `y[i]` will hold the total distance of the route for courier `i`.
    y = [solver.NumVar(0, solver.infinity(), f"y_{i}") for i in range(m)]
    
    # `z` is our objective!! It represents the maximum distance traveld by any single courier. We want to make this as small as posible.
    z = solver.NumVar(0, solver.infinity(), "z")

    # `u[i, j]` is a helper variable for the MTZ stuff to prevent subtours. It stores the position of item `j` in courier `i`'s tour.
    u = {(i, j): solver.NumVar(0, n, f"u_{i}_{j}") for i in range(m) for j in range(1, n + 1)}

    # --- 3. Defining the Constraints ---
    # these are the rules of the game that the solver must follow.

    # -- Assignment Constraints --
    # "Every item must be deliverd by exactly one courier."
    for j in range(1, n + 1):
        solver.Add(solver.Sum(a[i, j] for i in range(m)) == 1, name=f"item_assigned_{j}")

    # "No courier can be overloaded. the total size of items for a courier must not excede their capacity."
    for i in range(m):
        solver.Add(solver.Sum(a[i, j] * item_sizes[j - 1] for j in range(1, n + 1)) <= capacities[i], name=f"capacity_{i}")

    # -- Routing Flow Constraints --
    # "If a courier is assigned an item, they must travel to it from somewhere and from it to somewhere else"
    for i in range(m):
        for j in range(1, n + 1):
            # The number of arcs entering node j for courier i must equal 1 if a[i,j] is 1, and 0 otherwise.
            solver.Add(solver.Sum(x[i, k, j] for k in range(n + 1) if k != j) == a[i, j], name=f"flow_in_{i}_{j}")
            # The number of arcs leaving node j for courier i must equal 1 if a[i,j] is 1, and 0 otherwise.
            solver.Add(solver.Sum(x[i, j, k] for k in range(n + 1) if k != j) == a[i, j], name=f"flow_out_{i}_{j}")
    
    # -- Depot and Courier Usage Constraints --
    # "Each used courier must start at the depot an end at the depot."
    for i in range(m):
        # `used[i]` is 1 if the courier leaves the depot.
        solver.Add(solver.Sum(x[i, 0, k] for k in range(1, n + 1)) == used[i], name=f"depot_leave_{i}")
        # `used[i]` is 1 if the courier returns to the depot.
        solver.Add(solver.Sum(x[i, k, 0] for k in range(1, n + 1)) == used[i], name=f"depot_return_{i}")
        # A simple check: if a courier is assigned any item, they are considered 'used'.
        for j in range(1, n + 1):
            solver.Add(used[i] >= a[i, j], name=f"usage_link_{i}_{j}")

    # -- Linking Distance, Routes, and the Objective --
    # "lets calculate the total distance for each courier"
    for i in range(m):
        solver.Add(y[i] == solver.Sum(dist[j][k] * x[i, j, k] for j in range(n + 1) for k in range(n + 1) if j != k), name=f"route_dist_{i}")

    # "Our main goal: `z` must be greater than or equal to every couriers individual route distance."
    # By minimizing `z`, we are effectively minimizing the longest route (the minimax objective).
    for i in range(m):
        solver.Add(z >= y[i], name=f"objective_link_{i}")
    
    # -- Subtour Elimination (The MTZ Constraints) --
    # "A couriers route must be a single, continuous tour, not broken into little loops."
    # This is a classic problem in vehicle routing. these constraints prevent solutions where, for example,
    # a courier's path is 1 -> 2 -> 1 without ever visiting the depot.
    for i in range(m):
        for j in range(1, n + 1):
            # `u` can only be non-zero if the item is assigned to the courier.
            solver.Add(u[i, j] >= a[i, j], name=f"mtz_lower_{i}_{j}")
            solver.Add(u[i, j] <= n * a[i, j], name=f"mtz_upper_{i}_{j}")
        # This is the core MTZ logic.
        for j in range(1, n + 1):
            for k in range(1, n + 1):
                if j != k:
                    # If courier `i` travels from `j` to `k`, then the position of `k` in the tour (`u[i,k]`)
                    # must be greater than the position of `j`.
                    solver.Add(u[(i, j)] - u[(i, k)] + n * x[(i, j, k)] <= n - 1, name=f"mtz_subtour_{i}_{j}_{k}")

    # --- 4. Warm Start (Optional, but helpful!) ---
    # If the approach includes "WM" (Warm Start), we use our greedy solution to give the solver a hint.
    if "WM" in approach:
        _, ub = compute_greedy_solution(m, n, capacities, item_sizes, dist)
        if ub is not None and ub > 0:
            # We tell the solver: "Hey, I already found a solution with objective value `ub`.
            # You don't need to look for any solutions worse than this."
            solver.Add(z <= ub)

    # --- 5. Solve! ---
    solver.Minimize(z)
    status = solver.Solve()
    total_time_seconds = time.time() - start_time

    # --- 6. Processing the Results ---
    # Now, let's see what the solver came up with.
    solution = {
        "time": round(total_time_seconds, 2),
        "optimal": (status == pywraplp.Solver.OPTIMAL),
        "obj": -1.0,
        "sol": []
    }

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        final_routes = []
        for i in range(m):
            # Was this courier even used?
            if used[i].solution_value() > 0.5:
                route, current = [], 0 # Start at the depot (0)
                while True:
                    next_node = -1
                    # Find where we go from `current`.
                    for k in range(n + 1):
                        if k != current and x[i, current, k].solution_value() > 0.5:
                            next_node = k
                            break
                    if next_node <= 0: break # We've returned to the depot
                    route.append(next_node)
                    current = next_node
                final_routes.append(route)
            else:
                final_routes.append([]) # This courier did nothing.
        solution["sol"] = final_routes
        
        # Double-check the objective value, just to be sure.
        if any(final_routes):
            route_distances = [calculate_route_dist(r, dist) for r in final_routes]
            max_distance = max(route_distances)
            solution["obj"] = round(max_distance, 5)
        else:
            solution["obj"] = 0.0
    
    return {approach.upper(): solution}

def solve_process_wrapper(result_queue, *args):
    """
    This is a safety wrapper. It runs the solver in a completely separate process.
    If the solver crashes or has a serious error, it wont take down our main program.
    """
    try:
        solution = build_and_solve_mcp(*args)
        result_queue.put(solution)
    except Exception as e:
        log(f"The solver process ran into an exception: {e}")
        # Pass the error back to the main process so it knows what went wrong.
        result_queue.put(e)

# --- Main Execution Block ---
# This is what runs when you execute the script from the command line

if __name__ == "__main__":
    # Make sure the user provided a problem file.
    if len(sys.argv) != 2:
        sys.exit(1)
        
    instance_file = sys.argv[1]
    if not os.path.exists(instance_file):
        sys.exit(1)

    TOTAL_TIMEOUT = 300  # 5 minutes total time limit

    try:
        # 1. Read the problem data.
        m, n, capacities, item_sizes, dist = read_instance(instance_file)
        
        approach_str = "HiGHS+WM" # We're using the HiGHS solver with a Warm-start from our greedy heuristic.
        
        # 2. Set up for safe, timed execution using multiprocessing.
        result_queue = multiprocessing.Queue()
        solver_args = (m, n, capacities, item_sizes, dist, TOTAL_TIMEOUT, approach_str)
        
        # Create a new process to run the solver.
        p = multiprocessing.Process(target=solve_process_wrapper, args=(result_queue,) + solver_args)
        
        p.start()
        
        # 3. Wait for the process to finish, but not longer than our timeout.
        p.join(TOTAL_TIMEOUT)
        
        result = None
        # 4. Check what happened.
        if p.is_alive():
            # If the process is still running, it timed out.
            p.terminate()
            p.join()
            
            # Create a special "timeout" result.
            result = {
                approach_str.upper(): {
                    "time": TOTAL_TIMEOUT, "optimal": False, "obj": -1.0, 
                    "sol": [], "status": "Total timeout exceeded"
                }
            }
        else:
            # The process finished on its own.
            output = result_queue.get()
            if isinstance(output, Exception):
                # If the process sent back an error, well raise it here
                raise output
            result = output

    except (ValueError, RuntimeError, queue.Empty) as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)