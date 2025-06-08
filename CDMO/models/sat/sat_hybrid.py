from z3 import *
import time
import math
import random

def create_base_model(m, n, s, l, D, add_objective=True, symmetry_breaking=True):
    """
    Build the base model for the multiple couriers problem.

    Parameters:
      add_objective (bool): if True, creates an Optimize() instance and minimizes the objective;
                            otherwise, uses a plain Solver() (for feasibility checking).
      symmetry_breaking (bool): if True, adds symmetry-breaking constraints.
    """
    # Use Optimize if we need the objective; otherwise use Solver.
    solver = Optimize() if add_objective else Solver()
    depot = n  # depot index

    # Create tour variables: Y[i][0..n+1]
    Y = []
    for i in range(m):
        route = [Int(f"Y_{i}_{p}") for p in range(n + 2)]
        Y.append(route)

    # Fix depot at the start and end.
    for i in range(m):
        solver.add(Y[i][0] == depot)
        solver.add(Y[i][n+1] == depot)

    # For positions 1..n, each Y[i][p] is either -1 (unused) or an item index (0..n-1).
    for i in range(m):
        for p in range(1, n + 1):
            solver.add(Or(Y[i][p] == -1, And(Y[i][p] >= 0, Y[i][p] < n)))

    # Create L_vars[i] for the number of items in courier i's tour.
    L_vars = [Int(f"L_{i}") for i in range(m)]
    for i in range(m):
        solver.add(L_vars[i] >= 0, L_vars[i] <= n)
        for p in range(1, n + 1):
            # Positions up to L[i] must be assigned (not -1); positions after L[i] are -1.
            solver.add(If(p <= L_vars[i], Y[i][p] != -1, Y[i][p] == -1))

    # Each item appears exactly once.
    for j in range(n):
        occurrences = []
        for i in range(m):
            for p in range(1, n + 1):
                occurrences.append(If(Y[i][p] == j, 1, 0))
        solver.add(Sum(occurrences) == 1)

    # Capacity constraints: for each courier i, sum of sizes of assigned items ≤ l[i].
    for i in range(m):
        capacity_terms = []
        for p in range(1, n + 1):
            term = 0
            for j in range(n):
                term += If(Y[i][p] == j, s[j], 0)
            capacity_terms.append(term)
        solver.add(Sum(capacity_terms) <= l[i])

    # Compute route distances.
    route_distance = [Int(f"dist_{i}") for i in range(m)]
    for i in range(m):
        # First leg: depot -> first item (if any item assigned)
        first_leg = Sum([If(And(L_vars[i] >= 1, Y[i][1] == j), D[depot][j], 0) for j in range(n)])
        # Intermediate legs: between consecutive items.
        intermediate = 0
        for p in range(1, n):
            intermediate += Sum([If(And(L_vars[i] > p, Y[i][p] == j, Y[i][p+1] == k), D[j][k], 0)
                                   for j in range(n) for k in range(n)])
        # Last leg: last item -> depot.
        last_leg = Sum([If(L_vars[i] == v, Sum([If(Y[i][v] == j, D[j][depot], 0) for j in range(n)]), 0)
                        for v in range(1, n + 1)])
        solver.add(route_distance[i] == If(L_vars[i] == 0, 0, first_leg + intermediate + last_leg))

    # Define max_distance as the maximum route distance among all couriers.
    max_distance = Int("max_distance")
    for i in range(m):
        solver.add(max_distance >= route_distance[i])

    # Add objective if needed.
    if add_objective:
        solver.minimize(max_distance)

    # -----------------------------
    # Symmetry-Breaking Constraints
    # -----------------------------
    if symmetry_breaking:
        # (1) If courier i+1 is used then courier i must be used.
        for i in range(m - 1):
            solver.add(L_vars[i+1] <= L_vars[i])
        # (2) Force the first courier’s first item to be at most that of other couriers.
        for i in range(1, m):
            solver.add(Implies(And(L_vars[i] > 0, L_vars[0] > 0), Y[0][1] <= Y[i][1]))

    return solver, Y, L_vars, max_distance

def create_model_with_bound(m, n, s, l, D, bound, best_obj=None):
    """
    Creates a feasibility model with the extra constraint max_distance ≤ bound.
    Uses a plain Solver (without an objective) for branch-and-bound.
    """
    solver, Y, L_vars, max_distance = create_base_model(m, n, s, l, D, add_objective=False)
    solver.add(max_distance <= bound)
    if best_obj is not None:
        solver.add(max_distance <= best_obj)
    return solver, Y, L_vars, max_distance

def lns_sat(m, n, s, l, D, timeout_duration=30, stagnation_threshold=5, fix_probability=0.3):
    """
    LNS procedure: iteratively fix a random subset of decisions from the current best solution
    and re-solve the model using a feasibility check.

    Parameters:
      fix_probability (float): the probability with which each decision is fixed,
                               reducing the chance of over-constraining.

    Returns the best solution found and its objective value within timeout_duration seconds.
    """
    start_time = time.time()
    best_solution = None
    best_obj = None
    iteration = 0
    no_improvement_counter = 0

    while time.time() - start_time < timeout_duration:
        iteration += 1
        # Use a feasibility model (without the objective) for LNS iterations.
        solver, Y, L_vars, max_distance = create_base_model(m, n, s, l, D, add_objective=False)

        # If a best solution exists, fix some decisions from it.
        if best_solution is not None:
            for i in range(m):
                if i < len(best_solution) and best_solution[i]:
                    L_best = len(best_solution[i])
                    if random.random() < fix_probability:
                        solver.add(L_vars[i] == L_best)
                    for p in range(1, L_best + 1):
                        if random.random() < fix_probability:
                            solver.add(Y[i][p] == best_solution[i][p - 1])

        remaining = timeout_duration - (time.time() - start_time)
        iteration_timeout = min(10, remaining)
        solver.set(timeout=int(iteration_timeout * 1000))

        if solver.check() == sat:
            model = solver.model()
            current_obj = model.evaluate(max_distance).as_long()
            if best_solution is None or current_obj < best_obj:
                best_obj = current_obj
                sol = []
                for i in range(m):
                    route = []
                    L_val = model.evaluate(L_vars[i]).as_long()
                    for p in range(1, L_val + 1):
                        route.append(model.evaluate(Y[i][p]).as_long())
                    sol.append(route)
                best_solution = sol
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
        else:
            no_improvement_counter += 1

        print(f"LNS iteration {iteration}: best_obj = {best_obj if best_obj is not None else 'N/A'}, elapsed = {time.time() - start_time:.2f}s")
        if no_improvement_counter >= stagnation_threshold:
            break

    return best_solution, best_obj

# def sat_model(m, n, s, l, D, timeout_duration=300, stagnation_threshold=10):
#     """
#     Hybrid approach combining binary search on the objective with LNS refinement and
#     branch-and-bound pruning.
#     """
#     start_time = time.time()

#     print("Starting initial LNS phase...")
#     best_solution, best_obj = lns_sat(m, n, s, l, D, timeout_duration=30, stagnation_threshold=5)
#     if best_solution is None:
#         UB = 10**6  # an arbitrary high bound if no incumbent is found.
#         best_obj = UB
#     else:
#         UB = best_obj
#     LB = 0
#     no_improvement_counter = 0
#     iteration = 0

#     while time.time() - start_time < timeout_duration and LB < UB:
#         iteration += 1
#         mid = (LB + UB) // 2
#         print(f"Binary search iteration {iteration}: trying bound {mid}")

#         remaining = timeout_duration - (time.time() - start_time)
#         iteration_timeout = min(30, remaining)
#         solver, Y, L_vars, max_distance = create_model_with_bound(m, n, s, l, D, mid, best_obj)
#         solver.set(timeout=int(iteration_timeout * 1000))

#         if solver.check() == sat:
#             model = solver.model()
#             current_obj = model.evaluate(max_distance).as_long()
#             if current_obj < best_obj:
#                 best_obj = current_obj
#                 sol = []
#                 for i in range(m):
#                     route = []
#                     L_val = model.evaluate(L_vars[i]).as_long()
#                     for p in range(1, L_val + 1):
#                         route.append(model.evaluate(Y[i][p]).as_long())
#                     sol.append(route)
#                 best_solution = sol
#                 UB = best_obj
#                 no_improvement_counter = 0
#                 print(f"Found improved solution with objective {current_obj}")
#                 # Refine with a short LNS run.
#                 refined_sol, refined_obj = lns_sat(m, n, s, l, D, timeout_duration=iteration_timeout, stagnation_threshold=3)
#                 if refined_sol is not None and refined_obj < best_obj:
#                     best_solution = refined_sol
#                     best_obj = refined_obj
#                     UB = best_obj
#                     print(f"Refined incumbent to objective {refined_obj}")
#             else:
#                 no_improvement_counter += 1
#             UB = min(UB, current_obj)
#         else:
#             LB = mid + 1
#             no_improvement_counter += 1

#         print(f"Iteration {iteration}: LB = {LB}, UB = {UB}, best_obj = {best_obj}, elapsed = {time.time() - start_time:.2f}s")
#         if no_improvement_counter >= stagnation_threshold:
#             print("No improvement for several iterations. Terminating early.")
#             break

#     return best_solution, best_obj

def sat_model(m, n, s, l, D, timeout_duration=300, stagnation_threshold=10):
    """
    Hybrid approach combining binary search on the objective with LNS refinement and
    branch-and-bound pruning.

    Returns:
      {
          "time": runtime,
          "optimal": True/False,
          "obj": best_max_distance,
          "sol": assignments
      }
    """

    start_time = time.time()

    print("Starting initial LNS phase...")
    best_solution, best_obj = lns_sat(m, n, s, l, D, timeout_duration=30, stagnation_threshold=5)
    
    if best_solution is None:
        UB = 10**6  # Large upper bound when no initial solution is found
        best_obj = None
    else:
        UB = best_obj
    LB = 0
    no_improvement_counter = 0
    iteration = 0

    while time.time() - start_time < timeout_duration and LB < UB:
        iteration += 1
        mid = (LB + UB) // 2
        print(f"Binary search iteration {iteration}: trying bound {mid}")

        remaining = timeout_duration - (time.time() - start_time)
        iteration_timeout = min(30, remaining)
        solver, Y, L_vars, max_distance = create_model_with_bound(m, n, s, l, D, mid, best_obj)
        solver.set(timeout=int(iteration_timeout * 1000))

        if solver.check() == sat:
            model = solver.model()
            current_obj = model.evaluate(max_distance).as_long()

            assignments = [[model.evaluate(Y[i][p]).as_long() + 1 for p in range(1, model.evaluate(L_vars[i]).as_long() + 1)] for i in range(m)]

            best_obj = current_obj
            best_solution = assignments
            UB = best_obj
            no_improvement_counter = 0

            print(f"Found improved solution with objective {current_obj}")

            # Refine with LNS
            refined_sol, refined_obj = lns_sat(m, n, s, l, D, timeout_duration=iteration_timeout, stagnation_threshold=3)
            if refined_sol is not None and refined_obj < best_obj:
                best_solution = refined_sol
                best_obj = refined_obj
                UB = best_obj
                print(f"Refined incumbent to objective {refined_obj}")

        else:
            LB = mid + 1
            no_improvement_counter += 1

        print(f"Iteration {iteration}: LB = {LB}, UB = {UB}, best_obj = {best_obj}, elapsed = {time.time() - start_time:.2f}s")
        if no_improvement_counter >= stagnation_threshold:
            print("No improvement for several iterations. Terminating early.")
            break

    runtime = int(time.time() - start_time)
    # if best_solution is not None:
    #     model = solver.model()
    #     assignments = [[j + 1 for j in range(n) if model.evaluate(X[i][j])] for i in range(m)]
    #     max_dist_value = model.evaluate(max_distance).as_long()
    #     solution = {
    #         "time": runtime,
    #         "optimal": True,
    #         "obj": best_obj,
    #         "sol": best_solution
    #     }
    # else:
    #     solution = {
    #         "time": runtime,
    #         "optimal": False,
    #         "obj": None,
    #         "sol": None
    #     }
    if best_solution is not None:
        solution = {
        "time": runtime,
        "optimal": best_solution,
        "obj": best_obj,
        "sol": best_solution
        }
    else:
        solution = {
        "time": runtime,
        "optimal": None,
        "obj": None,
        "sol": [[] for _ in range(m)] 
        }
    

    return solution
