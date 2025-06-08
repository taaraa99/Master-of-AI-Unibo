#Symmetry breaking with search
from z3 import *
import time
import math

def smt_model_sb(num_couriers, num_items, item_sizes, courier_capacities, distance_matrix, search='binary', timeout_duration=None):
    start_time = time.time()

    def exactly_one(vars):
        return And(AtMost(*vars, 1), AtLeast(*vars, 1))

    # Decision variables
    assign = [[Bool(f"assign_{c}_{i}") for i in range(num_items)] for c in range(num_couriers)]
    travel = [[[Bool(f"travel_{c}_{i}_{j}") for j in range(num_items + 1)] for i in range(num_items + 1)] for c in range(num_couriers)]
    position = [[Int(f"position_{c}_{i}") for i in range(num_items + 1)] for c in range(num_couriers)]

    solver = Solver()

    # Load capacity constraints
    for c in range(num_couriers):
        solver.add(Sum([If(assign[c][i], item_sizes[i], 0) for i in range(num_items)]) <= courier_capacities[c])

    # Each item must be assigned to exactly one courier
    for i in range(num_items):
        solver.add(exactly_one([assign[c][i] for c in range(num_couriers)]))

    # Route constraints: each courier's tour must start and end at the depot
    for c in range(num_couriers):
        solver.add(exactly_one([travel[c][num_items][k] for k in range(num_items)]))
        solver.add(exactly_one([travel[c][k][num_items] for k in range(num_items)]))

        for i in range(num_items):
            solver.add(Sum([If(travel[c][i][k], 1, 0) for k in range(num_items + 1)]) == If(assign[c][i], 1, 0))
            solver.add(Sum([If(travel[c][k][i], 1, 0) for k in range(num_items + 1)]) == If(assign[c][i], 1, 0))

    # No single-node loops
    for c in range(num_couriers):
        for i in range(num_items + 1):
            solver.add(travel[c][i][i] == False)

    # Subtour elimination constraints (Miller-Tucker-Zemlin)
    for c in range(num_couriers):
        for i in range(1, num_items + 1):
            solver.add(And(position[c][i] >= 1, position[c][i] <= num_items))

        for i in range(num_items):
            for j in range(num_items + 1):
                if i != j:
                    solver.add(Implies(travel[c][i][j], position[c][i] + 1 == position[c][j]))

        for k in range(num_items):
            solver.add(Implies(travel[c][num_items][k], position[c][k] == 1))

    # Define the maximum distance traveled by any single courier correctly
    max_travel_distance = Int('max_travel_distance')

    # Compute total distance traveled by each courier
    distances = [Sum([If(travel[c][i][j], distance_matrix[i][j], 0) 
                      for i in range(num_items + 1) for j in range(num_items + 1)]) 
                 for c in range(num_couriers)]

    # Ensure max_travel_distance is the maximum of all individual courier distances
    for c in range(num_couriers):
        solver.add(max_travel_distance >= distances[c])

    # Symmetry Breaking Constraints
    for c in range(num_couriers - 1):
        solver.add(Sum([If(assign[c][i], item_sizes[i], 0) for i in range(num_items)]) 
                   >= Sum([If(assign[c + 1][i], item_sizes[i], 0) for i in range(num_items)]))

    for c in range(num_couriers):
        for i in range(1, num_items):
            for j in range(i):
                solver.add(Implies(And(assign[c][i], assign[c][j]), i > j))

    for c in range(num_couriers):
        for i in range(num_items):
            for j in range(i + 1, num_items):
                solver.add(Implies(And(travel[c][i][j], travel[c][j][i] == False), position[c][i] < position[c][j]))

    # Start the search process
    encoding_time = time.time()
    timeout = encoding_time + timeout_duration

    low, high = 0, sum(max(row) for row in distance_matrix)
    best_solution, best_max_distance = None, None

    if search == 'binary':
        while low <= high:
            mid = (low + high) // 2
            solver.push()
            if time.time() >= timeout:
                break
            solver.set('timeout', int((timeout - time.time()) * 1000))

            solver.add(max_travel_distance <= mid)
            if solver.check() == sat:
                best_solution = solver.model()
                best_max_distance = max([best_solution.evaluate(distances[c]).as_long() for c in range(num_couriers)])
                high = mid - 1
            else:
                low = mid + 1
            solver.pop()

    elif search == 'linear':
        solver.push()
        solver.set('timeout', int((timeout - time.time()) * 1000))
        while solver.check() == sat:
            best_solution = solver.model()
            best_max_distance = max([best_solution.evaluate(distances[c]).as_long() for c in range(num_couriers)])

            if best_max_distance <= low:
                break

            high = best_max_distance - 1
            solver.pop()
            solver.push()
            for c in range(num_couriers):
                solver.add(max_travel_distance <= high)

            if time.time() >= timeout:
                break
            solver.set('timeout', int((timeout - time.time()) * 1000))

    elif search == 'heuristic':
        solver.set('timeout', int((timeout - time.time()) * 1000))
        solver.add(max_travel_distance <= high)
        if solver.check() == sat:
            best_solution = solver.model()
            best_max_distance = max([best_solution.evaluate(distances[c]).as_long() for c in range(num_couriers)])

    end_time = time.time()
    solving_time = timeout_duration if end_time >= timeout else math.floor(end_time - encoding_time)

    if best_solution is None:
        return ("N/A" if solving_time == timeout_duration else "UNSAT", solving_time, None)

    # Extract routes from the best solution
    routes = []
    for c in range(num_couriers):
        arcs = []
        for i in range(num_items + 1):
            for j in range(num_items + 1):
                if is_true(best_solution.evaluate(travel[c][i][j])):
                    arcs.append((i + 1, j + 1))

        route = []
        current_node = num_items + 1
        while arcs:
            for arc in arcs:
                if arc[0] == current_node:
                    route.append(arc[1])
                    current_node = arc[1]
                    arcs.remove(arc)
                    break

        if route and route[-1] == num_items + 1:
            route.pop(-1)

        routes.append(route)

    return (best_max_distance, solving_time, routes)
