from z3 import *
import time
import math

def smt_model_nosb(num_couriers, num_items, item_sizes, courier_capacities, distance_matrix, timeout_duration=None):
    start_time = time.time()

    # Ensures that exactly one of the given Boolean variables is True
    def exactly_one(vars):
        return And(AtMost(*vars, 1), AtLeast(*vars, 1))

    # Decision Variables
    # assign[c][i]: True if courier 'c' is assigned to deliver item 'i'
    assign = [[Bool(f"assign_{c}_{i}") for i in range(num_items)] for c in range(num_couriers)]
    
    # travel[c][i][j]: True if courier 'c' travels from location 'i' to location 'j'
    travel = [[[Bool(f"travel_{c}_{i}_{j}") for j in range(num_items + 1)] for i in range(num_items + 1)] for c in range(num_couriers)]
    
    # position[c][i]: The position of item 'i' in courier 'c’s route (used for cycle elimination)
    position = [[Int(f"position_{c}_{i}") for i in range(num_items + 1)] for c in range(num_couriers)]

    solver = Solver()

    # Constraint 1: Each courier's load should not exceed its capacity
    for c in range(num_couriers):
        solver.add(Sum([If(assign[c][i], item_sizes[i], 0) for i in range(num_items)]) <= courier_capacities[c])

    # Constraint 2: Each item must be assigned to exactly one courier
    for i in range(num_items):
        solver.add(exactly_one([assign[c][i] for c in range(num_couriers)]))

    # Constraint 3: Each courier must start and end at the depot (node 'n')
    for c in range(num_couriers):
        solver.add(exactly_one([travel[c][num_items][k] for k in range(num_items)]))  # Start at depot
        solver.add(exactly_one([travel[c][k][num_items] for k in range(num_items)]))  # Return to depot

        for i in range(num_items):
            # If a courier visits an item, it must leave that item exactly once
            solver.add(Sum([If(travel[c][i][k], 1, 0) for k in range(num_items + 1)]) == If(assign[c][i], 1, 0))
            solver.add(Sum([If(travel[c][k][i], 1, 0) for k in range(num_items + 1)]) == If(assign[c][i], 1, 0))

    # Constraint 4: Prevent self-loops (a courier cannot travel to and from the same node)
    for c in range(num_couriers):
        for i in range(num_items + 1):
            solver.add(travel[c][i][i] == False)

    # Constraint 5: Subtour elimination using Miller-Tucker-Zemlin (MTZ) formulation
    for c in range(num_couriers):
        for i in range(1, num_items + 1):
            solver.add(And(position[c][i] >= 1, position[c][i] <= num_items))  # Ensure valid ordering

        for i in range(num_items):
            for j in range(num_items + 1):
                if i != j:
                    solver.add(Implies(travel[c][i][j], position[c][i] + 1 == position[c][j]))  # Ensure correct order

        for k in range(num_items):
            solver.add(Implies(travel[c][num_items][k], position[c][k] == 1))  # First visited node is position 1

    # Constraint 6: Ensure maximum travel distance is minimized
    max_travel_distance = Int('max_travel_distance')
    distances = [Sum([If(travel[c][i][j], distance_matrix[i][j], 0) for i in range(num_items + 1) for j in range(num_items + 1)]) for c in range(num_couriers)]
    
    for c in range(num_couriers):
        solver.add(distances[c] <= max_travel_distance)

    # Track encoding time
    encoding_time = time.time()
    timeout = encoding_time + timeout_duration

    # Initialize binary search boundaries
    low, high = 0, sum(max(row) for row in distance_matrix)
    best_solution, best_max_distance = None, None

    # Perform Binary Search to Minimize Maximum Travel Distance
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
            high = mid - 1  # Try to minimize further
        else:
            low = mid + 1  # Increase lower bound
        solver.pop()

    # Track solving time
    end_time = time.time()
    solving_time = timeout_duration if end_time >= timeout else math.floor(end_time - encoding_time)

    # If no valid solution is found, return "UNSAT" or "N/A"
    if best_solution is None:
        return ("N/A" if solving_time == timeout_duration else "UNSAT", solving_time, None)

    # Extract optimized routes from the solution
    routes = []
    for c in range(num_couriers):
        arcs = []
        for i in range(num_items + 1):
            for j in range(num_items + 1):
                if is_true(best_solution.evaluate(travel[c][i][j])):
                    arcs.append((i + 1, j + 1))  # Store visited arcs

        # Construct the actual route
        route = []
        current_node = num_items + 1  # Start at the depot (node n+1)
        while arcs:
            for arc in arcs:
                if arc[0] == current_node:
                    route.append(arc[1])  # Add next visited location
                    current_node = arc[1]
                    arcs.remove(arc)
                    break

        # Remove depot from the final route list
        if route and route[-1] == num_items + 1:
            route.pop(-1)

        routes.append(route)

    # Return the best-found solution
    return (best_max_distance, solving_time, routes)
