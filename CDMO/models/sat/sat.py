#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sat_search.py — Loop over all .dat instances and solve via pure‑SAT
             with options for:
               • binary or linear search,
               • Large Neighborhood Search (LNS) refinement,
               • k-Nearest-Neighbor (kNN) edge pruning.

Usage:
    python sat_search.py [--search <binary|linear>] [--timeout <seconds>]
                         [--lns] [--lns-iters <int>] [--destroy-frac <float>]
                         [--knn <k>] [instances...]

If no instances are specified, defaults to “inst*.dat” in the current directory.
If --knn is omitted, the full distance graph is used.
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
    m: int             # number of couriers
    n: int             # number of items
    cap: List[int]     # cap[i] = capacity of courier i
    size: List[int]    # size[j] = size of item j
    D: List[List[int]] # distance matrix of size (n+1)×(n+1); index n = depot

    @property
    def depot(self) -> int:
        return self.n  # depot is node index n


# ────────────────────────────────────────────────────────────────────────────────
# Loading instances from .dat files
# ────────────────────────────────────────────────────────────────────────────────
def load_instance(p: Path) -> Instance:
    """
    Read a .dat file with format:
      - First two tokens: m n
      - Next m tokens: capacities
      - Next n tokens: sizes
      - Next (n+1)*(n+1) tokens: flat distance matrix
    """
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
    if knn is None:
        neighbors: Dict[int, Set[int]] = {a: set(nodes) for a in nodes}
    else:
        neighbors = {a: set() for a in nodes}
        for a in nodes:
            dists = sorted((D[a][b], b) for b in nodes if b != a)
            for _, b in dists[:knn]:
                neighbors[a].add(b)
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

                 #lex
    for i in range(m-1):
        if cap[i] == cap[i+1]:
            # enforce v[i,*] <=_lex v[i+1,*]
            lex_clauses = []
            for k in range(n):
                # prefix equal
                prefix_eq = And(*[
                    Or(
                        And(v_vars[(i,j)], v_vars[(i+1,j)]),
                        And(Not(v_vars[(i,j)]), Not(v_vars[(i+1,j)]))
                    ) for j in range(k)
                ]) if k>0 else True
                # at k: courier i has 0, i+1 has 1
                diff_here = And(Not(v_vars[(i,k)]), v_vars[(i+1,k)])
                lex_clauses.append(And(prefix_eq, diff_here))
            s.add(Implies(cap[i] == cap[i+1], Or(*lex_clauses)))

    return s, e_vars, v_vars


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
    Returns (optimal_B, tours). strategy: "binary" or "linear" knn: if provided, prune edges to k-nearest neighbors
    """
    max_d = max(max(row) for row in inst.D)
    UB = sum(inst.size) * max_d
    t0 = perf_counter()
    grace = 2  # seconds before timeout to trigger fallback

    if strategy == "binary":
        low, high = 0, UB
        best_model: Optional[ModelRef] = None
        best_evars: Dict[Tuple[int,int,int], Bool] = {}
        best_vvars: Dict[Tuple[int,int], Bool] = {}
        best_B: Optional[int] = None

        while low <= high:
            # bail out early if within 'grace' seconds of the timeout
            if perf_counter() - t0 >= timeout - grace and best_model is not None:
                return best_B, item_lists(inst, best_model, best_vvars)

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

        if best_model is None:
            final_B = low
            s, e_vars, v_vars = build_solver(inst, final_B, knn)
            s.set("timeout", 0)
            if s.check() == sat:
                m = s.model()
                best_model = m
                best_evars = e_vars
                best_vvars = v_vars
                best_B = final_B
            else:
                raise RuntimeError("No feasible solution even at UB")

        return best_B, item_lists(inst, best_model, best_vvars)

    # linear search
    # Step 1: find feasible at UB
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

    # Step 2: decrement until UNSAT
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

    return current_B, item_lists(inst, best_model, best_vvars)


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
                best_tours = item_lists(inst, m, v_vars)

    return best_B, best_tours




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

    # collect all .dat files
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

    # make sure the top‐level res/SAT folder exists
    sat_res_dir = Path("res") / "SAT"
    sat_res_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        # 1) print header
        header = f"=== Solving {f.name}"
        if args.lns:
            header += " + LNS"
        if args.knn is not None:
            header += f" (kNN={args.knn})"
        header += " ==="
        print(header)

        # 2) solve
        start = perf_counter()
        inst = load_instance(f)
        try:
            if args.lns:
                opt_val, tours = lns_optimise(
                    inst,
                    timeout=args.timeout,
                    strategy=args.search,
                    lns_iters=args.lns_iters,
                    destroy_fraction=args.destroy_frac,
                    knn=args.knn
                )
            else:
                opt_val, tours = optimise(
                    inst,
                    timeout=args.timeout,
                    strategy=args.search,
                    knn=args.knn
                )
        except RuntimeError as e:
            print(f"[ERROR] {f.name}: no feasible solution: {e}", file=sys.stderr)
            continue
        elapsed = perf_counter() - start

        # 3) prepare JSON entry
        t_int   = int(elapsed) if elapsed < args.timeout else args.timeout
        optimal = (t_int < args.timeout)
        sol     = [[j+1 for j in sorted(route)] for route in tours]

        approach = "sat_" + args.search
        if args.lns:
            approach += "_lns"
        if args.knn is not None:
            approach += f"_knn{args.knn}"

        record = {
            "time":    t_int,
            "optimal": optimal,
            "obj":     opt_val,
            "sol":     sol
        }

        # 4) print JSON to stdout
        print(f"=== {f.name} result ===")
        print(json.dumps({approach: record}, indent=2))
        print(f"(solved in {t_int}s, optimal={optimal}, obj={opt_val})\n")

        # 5) merge into res/SAT/<instance_index>.json
        #    e.g. inst03.dat → 3.json
        digits = "".join(filter(str.isdigit, f.stem))
        idx    = int(digits) if digits else f.stem
        out_file = sat_res_dir / f"{idx}.json"

        if out_file.exists():
            full = json.loads(out_file.read_text())
        else:
            full = {}

        full[approach] = record
        out_file.write_text(json.dumps(full, indent=2))

        print(f"→ Updated {out_file}\n")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
