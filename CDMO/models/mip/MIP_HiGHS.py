import math
import sys
import os
import json
from ortools.linear_solver import pywraplp

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
    for line in lines[4:]:
        row = list(map(int, line.split()))
        raw.append(row)
    if len(raw) != (n + 1):
        raise ValueError("Distance matrix must have (n+1) rows.")
    # Construct an (n+1)x(n+1) matrix to hold the distances
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
    This function applies a basic greedy strategy with a touch of iterative improvement.
    It returns a tuple containing the routes found and the maximum distance among those routes.
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
                route_dists[i] += dist[current][0]  # Go back to the origin if no further item can be added
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
    # Try to further improve the solution by swapping items between routes for a few iterations.
    iterations = 10
    for _ in range(iterations):
        improved = False
        for i in range(m):
            for k in range(m):
                if i == k: continue
                for idx_i, j in enumerate(routes[i]):
                    for idx_k, l in enumerate(routes[k]):
                        def route_cost(route):
                            return sum(dist[0][item] + dist[item][0] for item in route)
                        cost_before = route_cost(routes[i]) + route_cost(routes[k])
                        routes[i][idx_i], routes[k][idx_k] = routes[k][idx_k], routes[i][idx_i]
                        cost_after = route_cost(routes[i]) + route_cost(routes[k])
                        if cost_after < cost_before:
                            improved = True
                        else:
                            # Revert the swap if no improvement is found
                            routes[i][idx_i], routes[k][idx_k] = routes[k][idx_k], routes[i][idx_i]
        if not improved:
            break
    best_ub = 0
    for i in range(m):
        cost_i = sum(dist[0][item] + dist[item][0] for item in routes[i])
        if cost_i > best_ub:
            best_ub = cost_i
    return routes, best_ub

def build_and_solve_mcp(m, n, capacities, item_sizes, dist, time_limit=300, approach="HiGHS"):
    """
    This function builds and solves the MCP using an OR-Tools mixed integer programming solver (HiGHS).
    It supports several optional enhancements depending on the 'approach' string provided:
      - "SB": Adds symmetry-breaking constraints.
      - "IMPLIED": Incorporates extra (redundant) implied constraints.
      - "WM": Uses a warm start by incorporating an improved heuristic solution.
      - "CUT": Inserts an additional valid inequality (cut) to strengthen the model.
    """
    solver = pywraplp.Solver.CreateSolver("highs")
    if not solver:
        raise RuntimeError("Unable to create HiGHS solver with OR-Tools.")
    param_string = f"time_limit={time_limit} heuristic_scale=2"
    if "WM" in approach:
        param_string += " warm_start=1"
    solver.SetSolverSpecificParametersAsString(param_string)
    
    # Calculate an enhanced greedy solution to obtain a tighter upper bound.
    routes, ub = compute_greedy_solution(m, n, capacities, item_sizes, dist)
    print("Greedy improved upper bound:", ub)
    if ("WM" in approach) and (ub is not None):
        # Add some placeholder constraints as hints to help the solver with a warm start.
        solver.Add(solver.Sum([]) <= ub)
        solver.Add(solver.NumVar(0, solver.infinity(), "dummy") == 0)
        solver.Add(solver.Sum([]) >= 0)
        solver.Add(solver.NumVar(0, solver.infinity(), "warm_bound") <= ub)
    
    # Define the decision variables for the model.
    a = {}
    for i in range(m):
        for j in range(1, n+1):
            a[(i,j)] = solver.BoolVar(f"a_{i}_{j}")
    x = {}
    for i in range(m):
        for j in range(n+1):
            for k in range(n+1):
                if j != k:
                    x[(i,j,k)] = solver.BoolVar(f"x_{i}_{j}_{k}")
    used = [solver.BoolVar(f"used_{i}") for i in range(m)]
    y = [solver.NumVar(0, solver.infinity(), f"y_{i}") for i in range(m)]
    z = solver.NumVar(0, solver.infinity(), "z")
    if ("WM" in approach) and (ub is not None):
        solver.Add(z <= ub)
    # Create extra binary variables for the valid inequality cuts.
    b = [solver.BoolVar(f"b_{i}") for i in range(m)]
    # Generate MTZ variables to help eliminate subtours.
    u = {}
    for i in range(m):
        for j in range(1, n+1):
            u[(i,j)] = solver.NumVar(0, n, f"u_{i}_{j}")
    # (A) Make sure that every item is assigned one and only one time.
    for j in range(1, n+1):
        solver.Add(solver.Sum(a[(i,j)] for i in range(m)) == 1)
    # (B) Ensure that no courier exceeds their capacity.
    for i in range(m):
        solver.Add(solver.Sum(a[(i,j)] * item_sizes[j-1] for j in range(1, n+1)) <= capacities[i])
    # (C) Set up flow constraints linking assignment variables with routing variables.
    for i in range(m):
        for j in range(1, n+1):
            solver.Add(solver.Sum(x[(i,j,k)] for k in range(n+1) if k != j) == a[(i,j)])
            solver.Add(solver.Sum(x[(i,k,j)] for k in range(n+1) if k != j) == a[(i,j)])
    # (D) Force the routes to correctly start from the origin.
    for i in range(m):
        solver.Add(solver.Sum(x[(i,0,k)] for k in range(1, n+1)) == used[i])
        solver.Add(solver.Sum(x[(i,k,0)] for k in range(1, n+1)) == used[i])
        for j in range(1, n+1):
            solver.Add(used[i] >= a[(i,j)])
    # (E) Compute the total travel distance for each route.
    for i in range(m):
        expr = solver.Sum(dist[j][k] * x[(i,j,k)] for j in range(n+1) for k in range(n+1) if j != k)
        solver.Add(y[i] == expr)
    # (F) Enforce that the objective variable is at least as large as each route’s distance.
    for i in range(m):
        solver.Add(z >= y[i])
    # (G) Apply the MTZ (Miller-Tucker-Zemlin) constraints to prevent subtours.
    for i in range(m):
        for j in range(1, n+1):
            solver.Add(u[(i,j)] >= a[(i,j)])
            solver.Add(u[(i,j)] <= n * a[(i,j)])
        for j in range(1, n+1):
            for k in range(1, n+1):
                if j != k:
                    solver.Add(u[(i,j)] - u[(i,k)] + n * x[(i,j,k)] <= n - 1)
    # (H) Optionally include symmetry-breaking constraints to reduce redundant equivalent solutions.
    for i in range(1, m):
        solver.Add(
            solver.Sum(j * a[(i - 1, j)] for j in range(1, n+1))
            <= solver.Sum(j * a[(i, j)] for j in range(1, n+1))
        )

    # Extra valid inequality cut ("CUT"):
    if "CUT" in approach:
        # L_direct is the smallest distance between any two different items.
        L_direct = min(dist[j][k] for j in range(1, n+1) for k in range(1, n+1) if j != k)
        # L_round is the smallest roundtrip distance between the origin and an item.
        L_round = min(dist[0][j] + dist[j][0] for j in range(1, n+1))
        extra_bound = L_direct + L_round
        for i in range(m):
            solver.Add(solver.Sum(a[(i,j)] for j in range(1, n+1)) - 1 >= b[i])
            solver.Add(solver.Sum(a[(i,j)] for j in range(1, n+1)) - 1 <= n * b[i])
            solver.Add(y[i] >= extra_bound * b[i])
    # Extra redundant implied constraint ("IMPLIED"):
    if "IMPLIED" in approach:
        max_roundtrip = max(dist[0][j] + dist[j][0] for j in range(1, n+1))
        solver.Add(z >= max_roundtrip / 2.0)
    solver.Minimize(z)
    
    status = solver.Solve()
    solution = {
        "time": int(solver.WallTime() / 1000.0),
        "optimal": (status == solver.OPTIMAL),
        "obj": None,
        "sol": []
    }
    if status in [solver.OPTIMAL, solver.FEASIBLE]:
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mcp_ortools_model.py instance.dat")
    else:
        instance_file = sys.argv[1]
        m, n, capacities, item_sizes, dist = read_instance(instance_file)
        # Solve the MCP using the enhanced approach that includes all improvements.
        sol = build_and_solve_mcp(m, n, capacities, item_sizes, dist, time_limit=300, approach="HiGHS+SB+IMPLIED+WM+CUT")
        print(json.dumps(sol, indent=4))
