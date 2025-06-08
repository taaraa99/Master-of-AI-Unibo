# Functions for building the solver and reconstructing tours.

from typing import Tuple, Dict, Set, Optional, List
import random
from z3 import Solver, Bool, Int, If, Implies, Not, Or, And, PbEq, PbLe, ModelRef, sat, is_true
from .instance import Instance
from time import perf_counter


def build_solver(
    inst: Instance,
    B: int,
    knn: Optional[int] = None
) -> Tuple[Solver, Dict[Tuple[int,int,int], Bool], Dict[Tuple[int,int], Bool]]:
    """
    Build and return a Z3 Solver for the given instance, with the constraint that
    each courier's total traveled distance ≤ B, and optionally edges restricted
    to k-nearest neighbors (kNN).

    Returns:
       - s: the Z3 Solver with all clauses asserted,
       - e_vars: dict mapping (i,a,b) → Bool e[i,a,b],
       - v_vars: dict mapping (i,j) → Bool v[i,j].
    """
    m, n, dep, D, cap, siz = inst.m, inst.n, inst.depot, inst.D, inst.cap, inst.size

    # Compute neighbor sets for kNN (or full graph if knn=None)
    nodes = list(range(n+1))
    # if knn is None:
    #     neighbors: Dict[int, Set[int]] = {a: set(nodes) for a in nodes}
    # else:
    #     neighbors = {a: set() for a in nodes}
    #     for a in nodes:
    #         dists = sorted((D[a][b], b) for b in nodes if b != a)
    #         for _, b in dists[:knn]:
    #             neighbors[a].add(b)
    #     # ensure depot connects to all nodes
    #     for a in nodes:
    #         neighbors[a].add(dep)
    #         neighbors[dep].add(a)
    if knn is None:
        neighbors = {a: set(nodes) for a in nodes}
    else:
        neighbors = {a: set() for a in nodes}
        # build symmetric kNN graph
        for a in nodes:
            dists = sorted((D[a][b], b) for b in nodes if b != a)
            for _, b in dists[:knn]:
                neighbors[a].add(b)
                neighbors[b].add(a)
        # ensure depot connects to all nodes
        for a in nodes:
            neighbors[a].add(dep)
            neighbors[dep].add(a)

    s = Solver()
    v_vars: Dict[Tuple[int,int], Bool] = {}
    e_vars: Dict[Tuple[int,int,int], Bool] = {}
    u_vars: Dict[Tuple[int,int], Int] = {}

    # 1) Assignment variables: v[i,j]
    for i in range(m):
        for j in range(n):
            v_vars[(i,j)] = Bool(f"v_{i}_{j}")

    # 2) Routing edge variables: e[i,a,b]
    for i in range(m):
        for a in nodes:
            for b in neighbors[a]:
                if a != b:
                    e_vars[(i,a,b)] = Bool(f"e_{i}_{a}_{b}")

    # 3) MTZ ordering variables u[i,k] for k=1..n
    for i in range(m):
        for k_idx in range(1, n+1):
            u_vars[(i,k_idx)] = Int(f"u_{i}_{k_idx}")

    # A) Assignment & capacity constraints
    for j in range(n):
        s.add(PbEq([(v_vars[(i,j)], 1) for i in range(m)], 1))
    for i in range(m):
        s.add(PbLe([(v_vars[(i,j)], siz[j]) for j in range(n)], cap[i]))

    # B) Routing & MTZ subtour-elimination
    for i in range(m):
        used_i = Bool(f"used_{i}")
        s.add(used_i == Or(*[v_vars[(i,j)] for j in range(n)]))

        out_dep = [(e_vars[(i, dep, b)], 1)
                   for b in neighbors[dep] if b != dep and (i,dep,b) in e_vars]
        in_dep  = [(e_vars[(i, a, dep)], 1)
                   for a in neighbors[dep] if a != dep and (i,a,dep) in e_vars]

        s.add(Implies(used_i, PbEq(out_dep, 1)))
        s.add(Implies(used_i, PbEq(in_dep, 1)))
        s.add(Implies(Not(used_i), PbEq(out_dep, 0)))
        s.add(Implies(Not(used_i), PbEq(in_dep, 0)))

        for j in range(n):
            ins  = [(e_vars[(i, a, j)], 1)
                    for a in neighbors[j] if (i,a,j) in e_vars]
            outs = [(e_vars[(i, j, b)], 1)
                    for b in neighbors[j] if (i,j,b) in e_vars]

            s.add(Implies(
                v_vars[(i,j)],
                And(
                    PbEq(ins,   1),
                    PbEq(outs,  1),
                    u_vars[(i, j+1)] >= 1,
                    u_vars[(i, j+1)] <= n
                )
            ))
            s.add(Implies(
                Not(v_vars[(i,j)]),
                And(
                    PbEq(ins,    0),
                    PbEq(outs,   0),
                    u_vars[(i, j+1)] == 0
                )
            ))

        for (ii,a,b), lit in e_vars.items():
            if ii != i:
                continue
            if a < n and b < n:
                s.add(Implies(
                    lit,
                    u_vars[(i, b+1)] == u_vars[(i, a+1)] + 1
                ))
            if a == dep and b < n:
                s.add(Implies(
                    lit,
                    u_vars[(i, b+1)] == 1
                ))

        dist_terms: List[Tuple[Bool,int]] = []
        for (ii,a,b), lit in e_vars.items():
            if ii == i:
                dist_terms.append((lit, inst.D[a][b]))
        s.add(PbLe(dist_terms, B))

    # C) Symmetry-breaking among equal-capacity couriers
    first_node = [Int(f"first_node_{i}") for i in range(inst.m)]
    for i in range(inst.m):
        head_cases = [
            If(e_vars[(i, inst.depot, b)], b+1, 0)
            for b in neighbors[inst.depot]
            if b != inst.depot and (i,inst.depot,b) in e_vars
        ]
        unused_case = If(
            Not(Or(*[v_vars[(i,j)] for j in range(inst.n)])),
            inst.n + 1, 0
        )
        s.add(first_node[i] == unused_case + sum(head_cases))

    for i in range(inst.m - 1):
        if inst.cap[i] == inst.cap[i+1]:
            s.add(Implies(
                inst.cap[i] == inst.cap[i+1],
                first_node[i] <= first_node[i+1]
            ))

    for i in range(m-1):
        if cap[i] == cap[i+1]:
            lex_clauses = []
            for k in range(n):
                prefix_eq = And(*[
                    Or(
                        And(v_vars[(i,j)], v_vars[(i+1,j)]),
                        And(Not(v_vars[(i,j)]), Not(v_vars[(i+1,j)]))
                    ) for j in range(k)
                ]) if k>0 else True
                diff_here = And(Not(v_vars[(i,k)]), v_vars[(i+1,k)])
                lex_clauses.append(And(prefix_eq, diff_here))
            s.add(Implies(cap[i] == cap[i+1], Or(*lex_clauses)))

    return s, e_vars, v_vars


def reconstruct_route(
    inst: Instance,
    model: ModelRef,
    e_vars: Dict[Tuple[int,int,int], Bool],
    courier: int
) -> List[int]:
    """
    Walks the chosen edges for `courier` from depot->...->depot,
    returning the sequence of item-nodes (0-based).
    """
    dep = inst.depot
    succ = [
        b for (i,a,b), lit in e_vars.items()
        if i == courier and a == dep
        and is_true(model.eval(lit, model_completion=True))
    ]
    if not succ:
        return []
    route: List[int] = []
    curr = succ[0]
    while curr != dep:
        route.append(curr)
        nexts = [
            b for (i,a,b), lit in e_vars.items()
            if i == courier and a == curr
            and is_true(model.eval(lit, model_completion=True))
        ]
        if not nexts:
            break
        curr = nexts[0]
    return route


def per_courier_distance(
    inst: Instance,
    model: ModelRef,
    e_vars: Dict[Tuple[int,int,int], Bool]
) -> List[int]:
    dist = [0] * inst.m
    for (i, a, b), lit in e_vars.items():
        if is_true(model.eval(lit, model_completion=True)):
            dist[i] += inst.D[a][b]
    return dist


def item_lists(
    inst: Instance,
    model: ModelRef,
    v_vars: Dict[Tuple[int,int], Bool]
) -> List[List[int]]:
    tours = [[] for _ in range(inst.m)]
    for (i, j), lit in v_vars.items():
        if is_true(model.eval(lit, model_completion=True)):
            tours[i].append(j)
    return tours



def optimise(
    inst: Instance,
    timeout: int = 300,
    strategy: str = "binary",
    knn: Optional[int] = None
) -> Tuple[int, List[List[int]]]:
    """
    Returns (optimal_B, tours). strategy: "binary" or "linear"
    knn: if provided, prune edges to k-nearest neighbors
    """
    max_d = max(max(row) for row in inst.D)
    UB = sum(inst.size) * max_d
    t0 = perf_counter()
    grace = 2  # seconds before timeout to trigger fallback

    if strategy == "binary":
        # 1) seed fallback: prove feasibility at UB
        s0, e0, v0 = build_solver(inst, UB, knn)
        rem = timeout - (perf_counter() - t0)
        s0.set("timeout", int(rem * 1000))
        if s0.check() != sat:
            raise RuntimeError("No feasible solution even at UB")
        m0 = s0.model()
        best_model, best_evars, best_vvars, best_B = m0, e0, v0, UB

        # 2) narrow search to [0, best_B]
        low, high = 0, best_B

        while low <= high:
            # bail out early if within 'grace' seconds of the timeout
            if perf_counter() - t0 >= timeout - grace:
                tours = [
                    reconstruct_route(inst, best_model, best_evars, i)
                    for i in range(inst.m)
                ]
                return best_B, tours

            mid = (low + high) // 2
            s, e_vars, v_vars = build_solver(inst, mid, knn)
            rem = timeout - (perf_counter() - t0)
            if rem <= 0:
                break
            s.set("timeout", int(rem * 1000))

            if s.check() == sat:
                m = s.model()
                best_model = m
                best_evars = e_vars
                best_vvars = v_vars
                best_B = mid
                high = mid - 1
            else:
                low = mid + 1

        # If we never found a better model in loop, best_model is at least UB
        tours = [
            reconstruct_route(inst, best_model, best_evars, i)
            for i in range(inst.m)
        ]
        return best_B, tours

    # linear search
    s0, e0, v0 = build_solver(inst, UB, knn)
    rem = timeout - (perf_counter() - t0)
    if rem <= 0:
        raise RuntimeError("Timeout before feasibility check")
    s0.set("timeout", int(rem * 1000))
    if s0.check() != sat:
        raise RuntimeError("No feasible solution even at UB")
    m0 = s0.model()
    dist0 = per_courier_distance(inst, m0, e0)
    B0 = max(dist0)
    current_B = B0
    best_model = m0
    best_evars = e0
    best_vvars = v0

    while current_B > 0:
        if perf_counter() - t0 >= timeout - grace:
            break
        cand = current_B - 1
        s, e_vars, v_vars = build_solver(inst, cand, knn)
        rem = timeout - (perf_counter() - t0)
        if rem <= 0:
            break
        s.set("timeout", int(rem * 1000))
        if s.check() == sat:
            m = s.model()
            best_model = m
            best_evars = e_vars
            best_vvars = v_vars
            current_B = cand
        else:
            break

    tours = [
        reconstruct_route(inst, best_model, best_evars, i)
        for i in range(inst.m)
    ]
    return current_B, tours



def lns_optimise(
    inst: Instance,
    timeout: int = 300,
    strategy: str = "binary",
    lns_iters: int = 20,
    destroy_fraction: float = 0.3,
    knn: Optional[int] = None
) -> Tuple[int, List[List[int]]]:
    """
    Perform an initial optimise(), then do LNS refinements:
      - randomly unassign ~destroy_fraction of items,
      - re-solve with distance bound = best_B - 1,
      - accept if strictly better.
    """
    t0 = perf_counter()
    best_B, best_tours = optimise(inst, timeout, strategy, knn)

    for _ in range(lns_iters):
        if perf_counter() - t0 >= timeout:
            break

        k = max(1, int(inst.n * destroy_fraction))
        to_unassign = set(random.sample(range(inst.n), k))

        s, e_vars, v_vars = build_solver(inst, best_B - 1, knn)
        for i, route in enumerate(best_tours):
            for j in route:
                if j not in to_unassign:
                    s.add(v_vars[(i, j)])

        rem = timeout - (perf_counter() - t0)
        if rem <= 0:
            break
        s.set("timeout", int(rem * 1000))

        if s.check() == sat:
            m = s.model()
            dists = per_courier_distance(inst, m, e_vars)
            new_B = max(dists)
            if new_B < best_B:
                best_B = new_B
                best_tours = [
                    reconstruct_route(inst, m, e_vars, i)
                    for i in range(inst.m)
                ]

    return best_B, best_tours
