#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smt_search.py — Loop over all .dat instances and solve via SMT
              with options for:
                  • binary, linear, or native Z3 optimisation,
                  • Large Neighborhood Search (LNS) refinement,
                  • k-Nearest-Neighbor (kNN) edge pruning.
"""
from __future__ import annotations
import argparse
import sys
import math
import json
import random
from pathlib import Path
from time import perf_counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set


from z3 import (
    Bool, Int, Solver, Optimize, If, Implies, Not, Or, And,
    PbEq, PbLe, sat, unknown, ModelRef, is_true
)

# ────────────────────────────────────────────────────────────────────────────────
# Data structure for instances
# just a little data holder for our problem, makes passing stuff around easier.
# ────────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class Instance:
    m: int
    n: int
    cap: List[int]
    size: List[int]
    D: List[List[int]]

    @property
    def depot(self) -> int:
        return self.n

# ────────────────────────────────────────────────────────────────────────────────
# Loading instances from .dat files
# this just reads the .dat text file and stuffs it into our Instance object
# ────────────────────────────────────────────────────────────────────────────────
def load_instance(p: Path) -> Instance:
    tok = p.read_text().split()
    it = iter(tok)
    m = int(next(it))
    n = int(next(it))
    cap  = [int(next(it)) for _ in range(m)]
    size = [int(next(it)) for _ in range(n)]
    flat = [int(next(it)) for _ in range((n+1)*(n+1))]
    D = [flat[r*(n+1):(r+1)*(n+1)] for r in range(n+1)]
    return Instance(m, n, cap, size, D)

# ────────────────────────────────────────────────────────────────────────────────
# Build Z3 solver/optimizer with SMT encoding + optional kNN pruning
# this is the heart of the model, where we tell Z3 all the rules of the game.
# ────────────────────────────────────────────────────────────────────────────────
def _populate_common_constraints(
    s: Solver | Optimize,
    inst: Instance,
    knn: Optional[int] = None
) -> Tuple[Dict[Tuple[int,int,int], Bool], Dict[Tuple[int,int], Bool]]:
    m, n, dep, D, cap, siz = inst.m, inst.n, inst.depot, inst.D, inst.cap, inst.size
    nodes = list(range(n+1))

    # --- k-Nearest-Neighbor Pruning ---
    # A neat trick to make things faster. if we use knn, we dont look at all
    # possible edges, just the ones between neighbors that are close to each other.
    # Cuts down the problem size a lot.
    if knn is None:
        neighbors: Dict[int, Set[int]] = {a: set(nodes) for a in nodes}
    else:
        neighbors = {a: set() for a in nodes}
        for a in nodes:
            # here we're sorting all other nodes by distance to find the closest ones
            dists = sorted((D[a][b], b) for b in nodes if b != a)
            for _, b in dists[:knn]:
                neighbors[a].add(b)
        # always make sure the depot is a neighbor so we can get back home
        for a in nodes:
            neighbors[a].add(dep)
            neighbors[dep].add(a)

    # --- Defining our variables ---
    # these are all the things Z3 gets to decide for us.
    v_vars: Dict[Tuple[int,int], Bool] = {} # is item j on truck i? yes or no.
    e_vars: Dict[Tuple[int,int,int], Bool] = {} # is truck i going from a to b? yes or no.
    u_vars: Dict[Tuple[int,int], Int] = {}   # this ones for stopping subtours. whats the position of a stop in a tour.

    # ok lets actually create the z3 variables now
    for i in range(m):
        for j in range(n):
            v_vars[(i,j)] = Bool(f"v_{i}_{j}")

    for i in range(m):
        for a in nodes:
            for b in neighbors[a]: # Only creat edges to neighbors
                if a != b:
                    e_vars[(i,a,b)] = Bool(f"e_{i}_{a}_{b}")

    for i in range(m):
        for k_idx in range(1, n+1):
            u_vars[(i,k_idx)] = Int(f"u_{i}_{k_idx}")

    # --- Assignment and Capacity Constraints ---
    # Each item has to be picked up by exactly one truck. no more, no less.
    for j in range(n):
        s.add(PbEq([(v_vars[(i,j)], 1) for i in range(m)], 1))
    
    # a truck cant carry more than its capacity. obvious really.
    for i in range(m):
        s.add(PbLe([(v_vars[(i,j)], siz[j]) for j in range(n)], cap[i]))

    # --- Flow and Subtour Elimination Constraints ---
    for i in range(m):
        # just a helper variable to see if truck i is even used.
        used_i = Bool(f"used_{i}")
        s.add(used_i == Or(*[v_vars[(i,j)] for j in range(n)]))
        
        # how many times does truck i leave the depot or come back to it
        out_dep = [(e_vars[(i, dep, b)], 1) for b in neighbors[dep] if b != dep and (i,dep,b) in e_vars]
        in_dep  = [(e_vars[(i, a, dep)], 1) for a in neighbors[dep] if a != dep and (i,a,dep) in e_vars]
        
        # if a truck is used, it must leave the depot exactly once.
        s.add(Implies(used_i, PbEq(out_dep, 1)))
        # and it must come back to the depot exactly once.
        s.add(Implies(used_i, PbEq(in_dep, 1)))
        # if a truck isnt used, it shouldnt go anywhere.
        s.add(Implies(Not(used_i), PbEq(out_dep, 0)))
        s.add(Implies(Not(used_i), PbEq(in_dep, 0)))

        # now for the customers...
        for j in range(n):
            # for a customer, if a truck visits it, it must arrive from somewhere and leave to somewhere else.
            ins  = [(e_vars[(i, a, j)], 1) for a in neighbors[j] if (i,a,j) in e_vars]
            outs = [(e_vars[(i, j, b)], 1) for b in neighbors[j] if (i,j,b) in e_vars]
            
            # if truck i visits customer j, then one edge must come in, and one edge must go out.
            s.add(Implies(v_vars[(i,j)], And(PbEq(ins, 1), PbEq(outs, 1), u_vars[(i, j+1)] >= 1, u_vars[(i, j+1)] <= n)))
            # if a truck doesnt visit j, no edges should connect to it for that truck.
            s.add(Implies(Not(v_vars[(i,j)]), And(PbEq(ins, 0), PbEq(outs, 0), u_vars[(i, j+1)] == 0)))
        
        # --- The magic subtour constraint (MTZ) ---
        # This stops the solver from making silly little loops that dont include the depot.
        for (ii,a,b), lit in e_vars.items():
            if ii != i: continue
            # if we go from customer a to customer b, b's position number must be one higher than a's.
            if a < n and b < n: s.add(Implies(lit, u_vars[(i, b+1)] == u_vars[(i, a+1)] + 1))
            # and if we go from the depot to a node, that node is the first stop.
            if a == dep and b < n: s.add(Implies(lit, u_vars[(i, b+1)] == 1))

    # --- Symmetry Breaking Constraints ---
    # This is a clever way to make the solver's life easier.
    first_node = [Int(f"first_node_{i}") for i in range(inst.m)]
    for i in range(inst.m):
        # figuring out which customer is first for each truck. just makes the next rule possible.
        head_cases = [If(e_vars[(i, inst.depot, b)], b+1, 0) for b in neighbors[inst.depot] if b != inst.depot and (i,inst.depot,b) in e_vars]
        unused_case = If(Not(Or(*[v_vars[(i,j)] for j in range(inst.n)])), inst.n + 2, 0) # give unused trucks a big number
        s.add(first_node[i] == unused_case + sum(head_cases))
    
    # The actual symmetry breaking. if two trucks have the same capacity, we dont care which is which.
    # so we force one to take the 'earlier' route. stops the solver exploring pointless copies of the same solution.
    for i in range(inst.m - 1):
        if inst.cap[i] == inst.cap[i+1]: s.add(first_node[i] <= first_node[i+1])
    
    return e_vars, v_vars

# This one builds a model for the binary/linear search. 
# It just needs to answer 'is a solution with cost B possible?'
def build_solver(
    inst: Instance,
    B: int,
    knn: Optional[int] = None
) -> Tuple[Solver, Dict[Tuple[int,int,int], Bool], Dict[Tuple[int,int], Bool]]:
    s = Solver()
    e_vars, v_vars = _populate_common_constraints(s, inst, knn)
    
    # this is the big one for this function, we add the total distance limit for each truck
    for i in range(inst.m):
        dist_terms: List[Tuple[Bool,int]] = []
        for (ii,a,b), lit in e_vars.items():
            if ii == i: dist_terms.append((lit, inst.D[a][b]))
        s.add(PbLe(dist_terms, B))
    return s, e_vars, v_vars

# This one is for the 'z3' mode. Instead of asking yes/no, we just tell
# it to find the best (smallest) distance possible.
def build_optimizer(
    inst: Instance,
    knn: Optional[int] = None
) -> Tuple[Optimize, Dict[Tuple[int,int,int], Bool], Dict[Tuple[int,int], Bool]]:
    opt = Optimize()
    e_vars, v_vars = _populate_common_constraints(opt, inst, knn)
    
    objective = Int("max_dist")
    # The objective is to minimize the longest tour of any single truck
    for i in range(inst.m):
        dist_terms = [inst.D[a][b] * e_vars[(i,a,b)] for (ii,a,b) in e_vars if ii == i]
        opt.add(objective >= sum(dist_terms))

    # here we just say... please minimize this for us. much easier.
    opt.minimize(objective)
    return opt, e_vars, v_vars

# ────────────────────────────────────────────────────────────────────────────────
# Solution reconstruction and utility
# ────────────────────────────────────────────────────────────────────────────────
# after we have a solution, this adds up the distances for each truck.
def per_courier_distance(inst: Instance, model: ModelRef, e_vars: Dict[Tuple[int,int,int], Bool]) -> List[int]:
    dist = [0] * inst.m
    for (i, a, b), lit in e_vars.items():
        if is_true(model.eval(lit, model_completion=True)): dist[i] += inst.D[a][b]
    return dist

# takes the mess of true/false edge variables from the solver and turns it 
# back into a nice list, a proper route that a person can read.
def reconstruct_route(inst: Instance, model: ModelRef, e_vars: Dict[Tuple[int,int,int], Bool], courier: int) -> List[int]:
    dep = inst.depot
    succ = [b for (i,a,b), lit in e_vars.items() if i == courier and a == dep and is_true(model.eval(lit, model_completion=True))]
    if not succ: return []
    route: List[int] = []
    curr = succ[0]
    visited = {dep, curr}
    route.append(curr)
    while curr != dep:
        next_nodes = [b for (i,a,b), lit in e_vars.items() if i == courier and a == curr and is_true(model.eval(lit, model_completion=True))]
        if not next_nodes or next_nodes[0] in visited: break
        curr = next_nodes[0]
        visited.add(curr)
        if curr != dep: route.append(curr)
    return route

# ────────────────────────────────────────────────────────────────────────────────
# Optimisation functions
# ────────────────────────────────────────────────────────────────────────────────
# The all-in-one optimization function. Just give it the problem and let Z3 do its thing.
def z3_optimise(
    inst: Instance,
    timeout: int = 300,
    knn: Optional[int] = None
) -> Tuple[int, List[List[int]], bool]:
    """
    Finds an optimal solution using Z3's native Optimize engine.
    Returns: (objective, solution_tours, was_proven_optimal)
    """
    opt, e_vars, v_vars = build_optimizer(inst, knn)
    opt.set("timeout", max(1, timeout * 1000))

    check_result = opt.check()
    
    if check_result not in [sat, unknown]:
        raise RuntimeError(f"Z3 optimisation failed with status: {check_result}")

    model = opt.model()
    final_dists = per_courier_distance(inst, model, e_vars)
    final_obj = max(final_dists) if final_dists else 0
    final_tours = [reconstruct_route(inst, model, e_vars, i) for i in range(inst.m)]
    
    # With Optimize, a 'sat' result means the solution is proven optimal.
    # An 'unknown' result means the timeout was hit, so it's not proven optimal.
    was_proven_optimal = (check_result == sat)
    
    return final_obj, final_tours, was_proven_optimal

# This is our manual optimization loop. We keep asking z3 'can you do it for less?' 
# (binary search) or 'find me any solution' and then try to beat it (linear search).
def optimise(
    inst: Instance,
    timeout: int = 300,
    strategy: str = "binary",
    knn: Optional[int] = None
) -> Tuple[int, List[List[int]], bool]:
    """
    Finds an optimal solution for the instance using iterative SAT calls.
    Returns: (objective, solution_tours, was_proven_optimal)
    """
    max_d = max(max(row) for row in inst.D if row) if inst.D else 0
    UB = sum(inst.D[inst.depot]) + sum(max_d for _ in range(inst.n))
    t0 = perf_counter()
    best_model: Optional[ModelRef] = None
    best_evars: Dict[Tuple[int,int,int], Bool] = {}
    search_completed = False

    if strategy == "binary":
        low, high = 0, UB
        while low <= high:
            if perf_counter() - t0 >= timeout: break
            mid = (low + high) // 2
            s, e_vars, v_vars = build_solver(inst, mid, knn)
            rem_ms = max(1, int((timeout - (perf_counter() - t0)) * 1000))
            s.set("timeout", rem_ms)
            if s.check() == sat:
                best_model = s.model()
                best_evars = e_vars
                high = mid - 1
            else:
                low = mid + 1
        if perf_counter() - t0 < timeout:
            search_completed = True
    else: # linear search
        s_init, e_init, _ = build_solver(inst, UB, knn)
        rem_ms = max(1, int((timeout - (perf_counter() - t0)) * 1000))
        if rem_ms <= 0: raise RuntimeError("Timeout before initial feasibility check.")
        s_init.set("timeout", rem_ms)

        if s_init.check() != sat:
            raise RuntimeError("Linear search: No feasible solution found even at loose UB.")

        best_model = s_init.model()
        best_evars = e_init
        dists_init = per_courier_distance(inst, best_model, best_evars)
        current_B = max(dists_init)

        while True:
            if perf_counter() - t0 >= timeout:
                search_completed = False
                break

            candidate_B = current_B - 1
            if candidate_B < 0:
                search_completed = True
                break

            s_iter, e_iter, _ = build_solver(inst, candidate_B, knn)
            rem_ms = max(1, int((timeout - (perf_counter() - t0)) * 1000))
            if rem_ms <= 0:
                search_completed = False
                break
            s_iter.set("timeout", rem_ms)

            if s_iter.check() == sat:
                best_model = s_iter.model()
                best_evars = e_iter
                new_dists = per_courier_distance(inst, best_model, best_evars)
                current_B = max(new_dists)
            else:
                # if it's UNSAT, it means the last solution we found was the best possible.
                search_completed = True
                break

    if best_model is None: raise RuntimeError("No feasible solution found within timeout.")
    final_dists = per_courier_distance(inst, best_model, best_evars)
    final_obj = max(final_dists) if final_dists else 0
    final_tours = [reconstruct_route(inst, best_model, best_evars, i) for i in range(inst.m)]
    return final_obj, final_tours, search_completed

# Large Neighborhood Search. A fancy way to try and improve a good solution. 
# we break a little part of it and ask z3 to fix it, hoping it finds a better way.
def lns_optimise(
    inst: Instance,
    timeout: int = 300,
    strategy: str = "binary",
    lns_iters: int = 20,
    destroy_fraction: float = 0.3,
    knn: Optional[int] = None
) -> Tuple[int, List[List[int]], bool]:
    t0 = perf_counter()
    best_obj, best_tours, optimal_search = optimise(inst, timeout, strategy, knn)
    if not optimal_search:
        # if the first search didnt find the optimal, we cant promise LNS will either
        return best_obj, best_tours, False

    for _ in range(lns_iters):
        if perf_counter() - t0 >= timeout:
            optimal_search = False
            break
        all_assigned_items = [item for tour in best_tours for item in tour]
        if not all_assigned_items: break
        # 'destroy' a random chunk of the solution
        k = max(1, int(len(all_assigned_items) * destroy_fraction))
        to_unassign = set(random.sample(all_assigned_items, k))
        # and then try to find a better way to do it
        s, e_vars, v_vars = build_solver(inst, best_obj - 1, knn)
        # lock in the part of the solution we're keeping
        for i, route in enumerate(best_tours):
            for item_j in route:
                if item_j not in to_unassign: s.add(v_vars[(i, item_j)])
        
        rem_ms = max(1, int((timeout - (perf_counter() - t0)) * 1000))
        s.set("timeout", rem_ms)
        if s.check() == sat:
            # cool, we found a better way!
            m = s.model()
            new_dists = per_courier_distance(inst, m, e_vars)
            new_obj = max(new_dists) if new_dists else 0
            if new_obj < best_obj:
                best_obj = new_obj
                best_tours = [reconstruct_route(inst, m, e_vars, i) for i in range(inst.m)]
    return best_obj, best_tours, optimal_search

# this is the main entry point that runs when you call the script from the command line
def main() -> None:
    # just setting up the command line arguments so you can run the script with different options
    parser = argparse.ArgumentParser(
        description="Loop over .dat instances and solve via SMT with binary/linear, LNS and kNN"
    )
    parser.add_argument(
        "instances", nargs="*", help=".dat files or glob patterns (e.g. inst*.dat)"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="per-instance time limit in seconds (default: 300)"
    )
    parser.add_argument(
        "--search", choices=["binary", "linear", "z3"], default="binary",
        help="search strategy: binary, linear, or z3 (native optimization)"
    )
    parser.add_argument(
        "--lns", action="store_true",
        help="apply Large Neighborhood Search refinement (not compatible with --search z3)"
    )
    parser.add_argument(
        "--lns-iters", type=int, default=20,
        help="LNS iterations (default: 20)"
    )
    parser.add_argument(
        "--destroy-frac", type=float, default=0.3,
        help="fraction of assignments to destroy in LNS (default: 0.3)"
    )
    parser.add_argument(
        "--knn", type=int,
        help="number of nearest neighbors for pruning edges (kNN)"
    )

    args = parser.parse_args()

    # you can't use LNS with the 'z3' optimizer, doesnt make sense
    if args.lns and args.search == 'z3':
        print("[ERROR] LNS is not compatible with the 'z3' search strategy.", file=sys.stderr)
        sys.exit(1)

    # find all the instance files to run
    patterns = args.instances if args.instances else ["inst*.dat"]
    files: List[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            files.extend(sorted(p.glob("*.dat")))
        else:
            files.extend(sorted(Path(".").glob(pat)))
    files = sorted(set(files))

    if not files:
        print("No instance files found.", file=sys.stderr)
        sys.exit(1)

    # where to save the results
    smt_res_dir = Path("res") / "SMT"
    smt_res_dir.mkdir(parents=True, exist_ok=True)

    # The main loop, go through each file and solve it
    for f in files:
        header = f"=== Solving {f.name}"
        if args.lns: header += " + LNS"
        if args.knn is not None: header += f" (kNN={args.knn})"
        print(header)

        start = perf_counter()
        inst = load_instance(f)
        opt_val, tours, optimal = -1, [[] for _ in range(inst.m)], False

        try:
            if args.lns:
                opt_val, tours, optimal = lns_optimise(
                    inst, timeout=args.timeout, strategy=args.search,
                    lns_iters=args.lns_iters, destroy_fraction=args.destroy_frac, knn=args.knn
                )
            elif args.search == 'z3':
                opt_val, tours, optimal = z3_optimise(
                    inst, timeout=args.timeout, knn=args.knn
                )
            else:
                opt_val, tours, optimal = optimise(
                    inst, timeout=args.timeout, strategy=args.search, knn=args.knn
                )
        except RuntimeError as e:
            print(f"[ERROR] {f.name}: {e}", file=sys.stderr)

        elapsed = perf_counter() - start
        
        t_int = math.floor(elapsed)
        if t_int >= args.timeout and not optimal:
            t_int = args.timeout
            optimal = False

        # the model uses 0-indexed items, but the problem wants 1-indexed, so fix that
        sol = [[item_idx + 1 for item_idx in route] for route in tours]

        # figure out what to call this run in the output file
        approach = args.search
        if args.lns: approach = "lns"
        if args.knn is not None: approach += f"_knn{args.knn}"

        record = {
            "time":    t_int,
            "optimal": optimal,
            "obj":     opt_val,
            "sol":     sol
        }

        print(f"=== {f.name} result ===")
        print(json.dumps({approach: record}, indent=2))
        print(f"(solved in {t_int}s, optimal={optimal}, obj={opt_val})\n")

        # save everything to a json file
        digits = "".join(filter(str.isdigit, f.stem))
        idx = int(digits) if digits else f.stem
        out_file = smt_res_dir / f"{idx}.json"

        if out_file.exists():
            with open(out_file, 'r') as jf:
                try:
                    full = json.load(jf)
                except json.JSONDecodeError:
                    full = {}
        else:
            full = {}

        full[approach] = record
        out_file.write_text(json.dumps(full, indent=2))
        print(f"→ Updated {out_file}\n")


if __name__ == "__main__":
    main()