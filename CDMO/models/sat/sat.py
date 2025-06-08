from z3 import *
import time


def sat_model(m, n, s, l, D, timeout_duration=300):
    solver = Optimize()

    # Boolean variables: X[i][j] = True if item j is assigned to courier i
    X = [[Bool(f"X_{i}_{j}") for j in range(n)] for i in range(m)]

    # Each item is assigned to exactly one courier
    for j in range(n):
        solver.add(Or([X[i][j] for i in range(m)]))
        for i1 in range(m):
            for i2 in range(i1 + 1, m):
                solver.add(Or(Not(X[i1][j]), Not(X[i2][j])))

    # Capacity constraints
    for i in range(m):
        solver.add(Sum([If(X[i][j], s[j], 0) for j in range(n)]) <= l[i])

    # -------------------------------------------------------------------
    #
    # For each courier we define:
    #  - An ordering variable order[i][j] for each item j:
    #      order[i][j] = position of item j in courier i’s route (if assigned), else 0.
    #  - The route distance is calculated as:
    #        depot -> first item + sum_{consecutive items} (distance between them)
    #        + last item -> depot.
    #
    # It is assumed that for each item j:
    #    D[j][0] gives the distance from the depot to item j, and
    #    D[j][-1] gives the distance from item j back to the depot.
    # -------------------------------------------------------------------
    order = [[Int(f"order_{i}_{j}") for j in range(n)] for i in range(m)]
    for i in range(m):
        for j in range(n):
            # If item j is assigned to courier i then its order is between 1 and n.
            solver.add(Implies(X[i][j], And(order[i][j] >= 1, order[i][j] <= n)))
            # Otherwise, order is 0.
            solver.add(Implies(Not(X[i][j]), order[i][j] == 0))
    
    # Ensure that for each courier, items assigned have distinct order positions.
    for i in range(m):
        for j in range(n):
            for k in range(j + 1, n):
                solver.add(Implies(And(X[i][j], X[i][k]), order[i][j] != order[i][k]))

    # For each courier, introduce an auxiliary variable to denote the maximum order (i.e., route length)
    max_order = [Int(f"max_order_{i}") for i in range(m)]
    for i in range(m):
        # If courier i is assigned no items, then max_order[i] = 0.
        # Otherwise, max_order[i] is at least the order of any item assigned.
        num_assigned = Sum([If(X[i][j], 1, 0) for j in range(n)])
        solver.add(Or(
            num_assigned == 0,
            And(num_assigned > 0, max_order[i] >= 1, *[Implies(X[i][j], max_order[i] >= order[i][j]) for j in range(n)])
        ))
    
    # Define route distance for each courier.
    route_distance = [Int(f"route_distance_{i}") for i in range(m)]
    for i in range(m):
        # Case: No items assigned → route distance = 0.
        no_item = (Sum([If(X[i][j], 1, 0) for j in range(n)]) == 0)
        # Depot to first item: for any item j with order 1, add D[j][0].
        depot_to_first = Sum([If(And(X[i][j], order[i][j] == 1), D[j][0], 0) for j in range(n)])
        # Last item to depot: for any item j with order equal to max_order, add D[j][-1].
        last_to_depot = Sum([If(And(X[i][j], order[i][j] == max_order[i]), D[j][-1], 0) for j in range(n)])
        # Intermediate distances: for consecutive positions p and p+1, add the distance between the items.
        intermediate = Sum([
            Sum([
                If(And(X[i][j], X[i][k], order[i][j] == p, order[i][k] == p + 1), D[j][k], 0)
                for j in range(n) for k in range(n)
            ])
            for p in range(1, n)  # p goes from 1 to n-1
        ])
        solver.add(route_distance[i] == If(no_item, 0, depot_to_first + intermediate + last_to_depot))
    
    # Global maximum route distance among all couriers.
    max_distance = Int("max_distance")
    for i in range(m):
        solver.add(max_distance >= route_distance[i])
    solver.minimize(max_distance)
    # -------------------------------------------------------------------
    
    # Solve the model
    solver.set(timeout=timeout_duration * 1000)
    start_time = time.time()
    result = solver.check()
    end_time = time.time()
    runtime = int(end_time - start_time)

    solution = {}
    if result is not None:
        model = solver.model()
        assignments = [[j + 1 for j in range(n) if model.evaluate(X[i][j])] for i in range(m)]
        max_dist_value = model.evaluate(max_distance).as_long()
        solution = {
            "time": runtime,
            "optimal": True,
            "obj": max_dist_value,
            "sol": assignments
        }
    else:
        solution = {
            "time": runtime,
            "optimal": False,
            "obj": None,
            "sol": [[] for _ in range(m)] 
        }

    return solution