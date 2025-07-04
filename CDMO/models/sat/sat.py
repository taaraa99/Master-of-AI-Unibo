#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sat_search.py — Loop over all .dat instances and solve via pure‑SAT
             with options for:
               • binary or linear search,
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
    Bool, Int, Solver, If, Implies, Not, Or, And,
    PbEq, PbLe, sat, ModelRef, is_true
)

# ────────────────────────────────────────────────────────────────────────────────
# Data structure for instances
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
# Build Z3 solver with SAT encoding + optional kNN pruning
# ────────────────────────────────────────────────────────────────────────────────
def build_solver(
    inst: Instance,
    B: int,
    knn: Optional[int] = None
) -> Tuple[Solver, Dict[Tuple[int,int,int], Bool], Dict[Tuple[int,int], Bool]]:
    m, n, dep, D, cap, siz = inst.m, inst.n, inst.depot, inst.D, inst.cap, inst.size
    nodes = list(range(n+1))
    if knn is None:
        neighbors: Dict[int, Set[int]] = {a: set(nodes) for a in nodes}
    else:
        neighbors = {a: set() for a in nodes}
        for a in nodes:
            dists = sorted((D[a][b], b) for b in nodes if b != a)
            for _, b in dists[:knn]:
                neighbors[a].add(b)
        for a in nodes:
            neighbors[a].add(dep)
            neighbors[dep].add(a)

    s = Solver()
    v_vars: Dict[Tuple[int,int], Bool] = {}
    e_vars: Dict[Tuple[int,int,int], Bool] = {}
    u_vars: Dict[Tuple[int,int], Int] = {}

    for i in range(m):
        for j in range(n):
            v_vars[(i,j)] = Bool(f"v_{i}_{j}")

    for i in range(m):
        for a in nodes:
            for b in neighbors[a]:
                if a != b:
                    e_vars[(i,a,b)] = Bool(f"e_{i}_{a}_{b}")

    for i in range(m):
        for k_idx in range(1, n+1):
            u_vars[(i,k_idx)] = Int(f"u_{i}_{k_idx}")

    for j in range(n):
        s.add(PbEq([(v_vars[(i,j)], 1) for i in range(m)], 1))
    for i in range(m):
        s.add(PbLe([(v_vars[(i,j)], siz[j]) for j in range(n)], cap[i]))

    for i in range(m):
        used_i = Bool(f"used_{i}")
        s.add(used_i == Or(*[v_vars[(i,j)] for j in range(n)]))
        out_dep = [(e_vars[(i, dep, b)], 1) for b in neighbors[dep] if b != dep and (i,dep,b) in e_vars]
        in_dep  = [(e_vars[(i, a, dep)], 1) for a in neighbors[dep] if a != dep and (i,a,dep) in e_vars]
        s.add(Implies(used_i, PbEq(out_dep, 1)))
        s.add(Implies(used_i, PbEq(in_dep, 1)))
        s.add(Implies(Not(used_i), PbEq(out_dep, 0)))
        s.add(Implies(Not(used_i), PbEq(in_dep, 0)))
        for j in range(n):
            ins  = [(e_vars[(i, a, j)], 1) for a in neighbors[j] if (i,a,j) in e_vars]
            outs = [(e_vars[(i, j, b)], 1) for b in neighbors[j] if (i,j,b) in e_vars]
            s.add(Implies(v_vars[(i,j)], And(PbEq(ins, 1), PbEq(outs, 1), u_vars[(i, j+1)] >= 1, u_vars[(i, j+1)] <= n)))
            s.add(Implies(Not(v_vars[(i,j)]), And(PbEq(ins, 0), PbEq(outs, 0), u_vars[(i, j+1)] == 0)))
        for (ii,a,b), lit in e_vars.items():
            if ii != i: continue
            if a < n and b < n: s.add(Implies(lit, u_vars[(i, b+1)] == u_vars[(i, a+1)] + 1))
            if a == dep and b < n: s.add(Implies(lit, u_vars[(i, b+1)] == 1))
        dist_terms: List[Tuple[Bool,int]] = []
        for (ii,a,b), lit in e_vars.items():
            if ii == i: dist_terms.append((lit, inst.D[a][b]))
        s.add(PbLe(dist_terms, B))

    first_node = [Int(f"first_node_{i}") for i in range(inst.m)]
    for i in range(inst.m):
        head_cases = [If(e_vars[(i, inst.depot, b)], b+1, 0) for b in neighbors[inst.depot] if b != inst.depot and (i,inst.depot,b) in e_vars]
        unused_case = If(Not(Or(*[v_vars[(i,j)] for j in range(inst.n)])), inst.n + 2, 0)
        s.add(first_node[i] == unused_case + sum(head_cases))
    for i in range(inst.m - 1):
        if inst.cap[i] == inst.cap[i+1]: s.add(first_node[i] <= first_node[i+1])
    return s, e_vars, v_vars

def per_courier_distance(inst: Instance, model: ModelRef, e_vars: Dict[Tuple[int,int,int], Bool]) -> List[int]:
    dist = [0] * inst.m
    for (i, a, b), lit in e_vars.items():
        if is_true(model.eval(lit, model_completion=True)): dist[i] += inst.D[a][b]
    return dist

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

def optimise(
    inst: Instance,
    timeout: int = 300,
    strategy: str = "binary",
    knn: Optional[int] = None
) -> Tuple[int, List[List[int]], bool]:
    """
    Finds an optimal solution for the instance.
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
        # Step 1: Find an initial feasible solution to get a good starting upper bound.
        s_init, e_init, _ = build_solver(inst, UB, knn)
        rem_ms = max(1, int((timeout - (perf_counter() - t0)) * 1000))
        if rem_ms <= 0: raise RuntimeError("Timeout before initial feasibility check.")
        s_init.set("timeout", rem_ms)

        if s_init.check() != sat:
            raise RuntimeError("Linear search: No feasible solution found even at loose UB.")
        
        # Step 2: Get the actual distance of this first solution. This is our starting best-known objective.
        best_model = s_init.model()
        best_evars = e_init
        dists_init = per_courier_distance(inst, best_model, best_evars)
        current_B = max(dists_init)
        
        # Step 3: Iteratively try to find a solution with a strictly better objective.
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
                # Found a better solution. Get its true cost to guide the next search.
                best_model = s_iter.model()
                best_evars = e_iter
                new_dists = per_courier_distance(inst, best_model, best_evars)
                current_B = max(new_dists)
            else:
                # UNSAT for candidate_B means `current_B` is the optimal value.
                search_completed = True
                break

    if best_model is None: raise RuntimeError("No feasible solution found within timeout.")
    final_dists = per_courier_distance(inst, best_model, best_evars)
    final_obj = max(final_dists) if final_dists else 0
    final_tours = [reconstruct_route(inst, best_model, best_evars, i) for i in range(inst.m)]
    return final_obj, final_tours, search_completed

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
        return best_obj, best_tours, False

    for _ in range(lns_iters):
        if perf_counter() - t0 >= timeout:
            optimal_search = False
            break
        all_assigned_items = [item for tour in best_tours for item in tour]
        if not all_assigned_items: break
        k = max(1, int(len(all_assigned_items) * destroy_fraction))
        to_unassign = set(random.sample(all_assigned_items, k))
        s, e_vars, v_vars = build_solver(inst, best_obj - 1, knn)
        for i, route in enumerate(best_tours):
            for item_j in route:
                if item_j not in to_unassign: s.add(v_vars[(i, item_j)])
        rem_ms = max(1, int((timeout - (perf_counter() - t0)) * 1000))
        s.set("timeout", rem_ms)
        if s.check() == sat:
            m = s.model()
            new_dists = per_courier_distance(inst, m, e_vars)
            new_obj = max(new_dists) if new_dists else 0
            if new_obj < best_obj:
                best_obj = new_obj
                best_tours = [reconstruct_route(inst, m, e_vars, i) for i in range(inst.m)]
    return best_obj, best_tours, optimal_search

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loop over .dat instances and solve via SAT with binary/linear, LNS and kNN"
    )
    parser.add_argument(
        "instances", nargs="*", help=".dat files or glob patterns (e.g. inst*.dat)"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="per-instance time limit in seconds (default: 300)"
    )
    parser.add_argument(
        "--search", choices=["binary", "linear"], default="binary",
        help="search strategy: binary (default) or linear"
    )
    parser.add_argument(
        "--lns", action="store_true",
        help="apply Large Neighborhood Search refinement after initial solve"
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

    sat_res_dir = Path("res") / "SAT"
    sat_res_dir.mkdir(parents=True, exist_ok=True)

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
            else:
                opt_val, tours, optimal = optimise(
                    inst, timeout=args.timeout, strategy=args.search, knn=args.knn
                )
        except RuntimeError as e:
            print(f"[ERROR] {f.name}: {e}", file=sys.stderr)
        
        elapsed = perf_counter() - start
        
        t_int = math.floor(elapsed)
        if t_int >= args.timeout:
            t_int = args.timeout
            optimal = False

        sol = [[item_idx + 1 for item_idx in route] for route in tours]

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

        digits = "".join(filter(str.isdigit, f.stem))
        idx = int(digits) if digits else f.stem
        out_file = sat_res_dir / f"{idx}.json"

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
