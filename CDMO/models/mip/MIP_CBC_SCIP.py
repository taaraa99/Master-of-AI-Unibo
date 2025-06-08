import math
import sys
import os
import json
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
                route_dists[i] += dist[current][0]  # Return to the origin if no further item can be added
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
    The 'approach' parameter (case-insensitive) defines the solver:
      - "CBC": uses CBC_MIXED_INTEGER_PROGRAMMING.
      - "SCIP": uses SCIP_MIXED_INTEGER_PROGRAMMING.
    This model employs MTZ subtour elimination along with optional enhancements like 
    symmetry-breaking ("SB") and extra implied constraints ("IMPLIED"). The internal time limit 
    is set in milliseconds.
    """
    solver_choice = approach.upper()
    if solver_choice == "CBC":
        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    elif solver_choice == "SCIP":
        solver = pywraplp.Solver.CreateSolver("SCIP_MIXED_INTEGER_PROGRAMMING")
    else:
        raise ValueError("Unsupported approach: choose CBC or SCIP")
    if not solver:
        raise RuntimeError(f"Unable to create {solver_choice} solver with OR-Tools.")

    # Configure the solver's internal time limit (milliseconds)
    solver.set_time_limit(time_limit * 1000)
    if solver_choice == "SCIP":
        try:
            solver.SetSolverSpecificParametersAsString("scip/time_limit = 300")
        except Exception as e:
            print("SCIP parameter setting error:", e)
    else:
        solver.SetSolverSpecificParametersAsString(f"time_limit={time_limit} heuristic_scale=2")

    # Optionally, compute a greedy solution to get an initial upper bound.
    routes, ub = None, None
    try:
        routes, ub = compute_greedy_solution(m, n, capacities, item_sizes, dist)
        print(f"Greedy solution upper bound: {ub}")
    except Exception as e:
        print("Warning: Greedy approach failed:", e)

    # Define the decision variables for the optimization model.
    a = {}
    for i in range(m):
        for j in range(1, n+1):
            a[(i, j)] = solver.BoolVar(f"a_{i}_{j}")

    x = {}
    for i in range(m):
        for j in range(n+1):
            for k in range(n+1):
                if j != k:
                    x[(i, j, k)] = solver.BoolVar(f"x_{i}_{j}_{k}")

    used = [solver.BoolVar(f"used_{i}") for i in range(m)]
    y = [solver.NumVar(0, solver.infinity(), f"y_{i}") for i in range(m)]
    z = solver.NumVar(0, solver.infinity(), "z")
    if ub is not None:
        solver.Add(z <= ub)

    # Define MTZ variables to help eliminate subtours.
    u = {}
    for i in range(m):
        for j in range(1, n+1):
            u[(i, j)] = solver.NumVar(0, n, f"u_{i}_{j}")

    # (A) Ensure every item is assigned exactly once.
    for j in range(1, n+1):
        solver.Add(solver.Sum(a[(i, j)] for i in range(m)) == 1)
    # (B) Make sure each courier's load does not exceed its capacity.
    for i in range(m):
        solver.Add(solver.Sum(a[(i, j)] * item_sizes[j-1] for j in range(1, n+1)) <= capacities[i])
    # (C) Add flow constraints to link assignments with routing choices.
    for i in range(m):
        for j in range(1, n+1):
            solver.Add(solver.Sum(x[(i, j, k)] for k in range(n+1) if k != j) == a[(i, j)])
            solver.Add(solver.Sum(x[(i, k, j)] for k in range(n+1) if k != j) == a[(i, j)])
    # (D) Ensure routes start and end at the origin.
    for i in range(m):
        solver.Add(solver.Sum(x[(i, 0, k)] for k in range(1, n+1)) == used[i])
        solver.Add(solver.Sum(x[(i, k, 0)] for k in range(1, n+1)) == used[i])
        for j in range(1, n+1):
            solver.Add(used[i] >= a[(i, j)])
    # (E) Calculate the total travel distance for each courier's route.
    for i in range(m):
        solver.Add(y[i] == solver.Sum(dist[j][k] * x[(i, j, k)]
                                        for j in range(n+1) for k in range(n+1) if j != k))
    # (F) Constrain z so that it is at least as large as every route's distance.
    for i in range(m):
        solver.Add(z >= y[i])
    # (G) Apply MTZ constraints to prevent the formation of subtours.
    for i in range(m):
        for j in range(1, n+1):
            solver.Add(u[(i, j)] >= a[(i, j)])
            solver.Add(u[(i, j)] <= n * a[(i, j)])
        for j in range(1, n+1):
            for k in range(1, n+1):
                if j != k:
                    solver.Add(u[(i, j)] - u[(i, k)] + n * x[(i, j, k)] <= n - 1)
    # (H) Optional: add symmetry-breaking constraints if "SB" is specified.
    if "SB" in approach.upper():
        for i2 in range(1, m):
            i1 = i2 - 1
            solver.Add(solver.Sum(j * a[(i1, j)] for j in range(1, n+1))
                       <= solver.Sum(j * a[(i2, j)] for j in range(1, n+1)))
    # (I) Optional: include extra implied constraints if "IMPLIED" is specified.
    if "IMPLIED" in approach.upper():
        max_roundtrip = max(dist[0][j] + dist[j][0] for j in range(1, n+1))
        solver.Add(z >= max_roundtrip / 2.0)

    # Set the objective: minimize the maximum route distance (z).
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

