"""
pure_sat_model.py — A pure SAT model for the MCP problem.
This version avoids integer variables.
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
import traceback

from z3 import (
    Bool, Solver, If, Implies, Not, Or, And,
    PbEq, PbLe, sat, ModelRef, is_true
)

# Helper for lexicographical less-than-or-equal constraint
def LexLe(a: list, b: list):
    """Encodes that list of bools a is lexicographically less-than-or-equal-to b."""
    return Or([And([a[i] == b[i] for i in range(k)] + [Not(a[k]), b[k]]) for k in range(len(a))] + [And([a[i] == b[i] for i in range(len(a))])])

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
# Build Z3 solver with a pure SAT encoding
# ────────────────────────────────────────────────────────────────────────────────
def build_solver_pure_sat(
    inst: Instance,
    B: int
) -> Tuple[Solver, Dict[Tuple[int,int,int], Bool], Dict[Tuple[int,int], Bool]]:
    m, n, dep, D, cap, siz = inst.m, inst.n, inst.depot, inst.D, inst.cap, inst.size
    nodes = list(range(n + 1))
    items = list(range(n))
    positions = list(range(n))

    # All nodes are considered neighbors
    neighbors: Dict[int, Set[int]] = {a: set(nodes) for a in nodes}

    s = Solver()
    v_vars: Dict[Tuple[int, int], Bool] = { (i, j): Bool(f"v_{i}_{j}") for i in range(m) for j in items }
    e_vars: Dict[Tuple[int, int, int], Bool] = { (i, a, b): Bool(f"e_{i}_{a}_{b}") for i in range(m) for a in nodes for b in neighbors[a] if a != b }
    p_vars: Dict[Tuple[int, int, int], Bool] = { (i, j, k): Bool(f"p_{i}_{j}_{k}") for i in range(m) for j in items for k in positions }

    # 1.Each item must be in exactly one positon (i, k) across all couriers
    for j in items:
        s.add(PbEq([(p_vars[i, j, k], 1) for i in range(m) for k in positions], 1))

    # 2. each position (i, k) can hold at most one item.
    for i in range(m):
        for k in positions:
            s.add(PbLe([(p_vars[i, j, k], 1) for j in items], 1))

    # 3. Path contiguity: if pos k is used, pos k-1 must be used.
    for i in range(m):
        for k in positions[1:]:
            pos_k_filled = Or([p_vars[i, j, k] for j in items])
            pos_k_minus_1_filled = Or([p_vars[i, j, k-1] for j in items])
            s.add(Implies(pos_k_filled, pos_k_minus_1_filled))

    # 4. Link p_vars to v_vars (for capacity check)..
    for i in range(m):
        for j in items:
            s.add(v_vars[i, j] == Or([p_vars[i, j, k] for k in positions]))

    # 5. Capacity constraint (uses derived v_vars
    for i in range(m):
        s.add(PbLe([(v_vars[i, j], siz[j]) for j in items], cap[i]))

    # 6.Link p_vars to e_vars (for distance check)
    for i in range(m):
        for a in nodes:
            for b in nodes:
                if a == b: continue
                
                # Define the logic that would make an edge true
                edge_logic: Bool
                if a == dep:
                    # Edge from depot to item 'b'
                    if b < n: # b must be an item
                        edge_logic = p_vars[i, b, 0]
                    else:
                        edge_logic = Bool(False)
                elif b == dep:
                    # Edge from item 'a' to depot
                    if a < n: # a must be an item
                        is_last = Or(
                            [And(p_vars[i, a, k], Not(Or([p_vars[i, j, k+1] for j in items]))) for k in positions[:-1]] + 
                            [p_vars[i, a, n-1]]
                        )
                        edge_logic = is_last
                    else:
                        edge_logic = Bool(False)
                else:
                    # Edge between two items 'a' and 'b'
                    is_consecutive = Or([And(p_vars[i, a, k], p_vars[i, b, k+1]) for k in positions[:-1]])
                    edge_logic = is_consecutive
                
                # Now, link this logic to the actual e_var
                if (i, a, b) in e_vars:
                    s.add(e_vars[i, a, b] == edge_logic)
                else:
                    s.add(Not(edge_logic))

    # 7. Distance constraint (uses derived e_vars)
    for i in range(m):
        dist_terms = [(e_vars[i, a, b], D[a][b]) for (i_loop, a, b) in e_vars if i_loop == i]
        s.add(PbLe(dist_terms, B))

    # --- Pure SAT Symmetry Breaking ---
    for i in range(m - 1):
        if cap[i] == cap[i+1]:
            s.add(LexLe([v_vars[i, j] for j in items], [v_vars[i+1, j] for j in items]))

    return s, e_vars, v_vars

def per_courier_distance(inst: Instance, model: ModelRef, e_vars: Dict[Tuple[int,int,int], Bool]) -> List[int]:
    dist = [0] * inst.m
    for (i, a, b), lit in e_vars.items():
        if is_true(model.eval(lit, model_completion=True)):
            dist[i] += inst.D[a][b]
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
    return route # Return 0-indexed items

def optimise(
    inst: Instance,
    timeout: int = 300,
    strategy: str = "binary"
) -> Tuple[int, List[List[int]], bool]:
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
            s, e_vars, _ = build_solver_pure_sat(inst, mid)
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
        s_init, e_init, _ = build_solver_pure_sat(inst, UB)
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
            s_iter, e_iter, _ = build_solver_pure_sat(inst, candidate_B)
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
                search_completed = True
                break

    if best_model is None: raise RuntimeError("No feasible solution found within timeout.")
    final_dists = per_courier_distance(inst, best_model, best_evars)
    final_obj = max(final_dists) if final_dists else 0
    final_tours = [reconstruct_route(inst, best_model, best_evars, i) for i in range(inst.m)]
    return final_obj, final_tours, search_completed

def main() -> None:
    parser = argparse.ArgumentParser(description="Solve MCP instances with a pure SAT model.")
    parser.add_argument("instances", nargs="*", default=["inst*.dat"], help=".dat files or glob patterns")
    parser.add_argument("--timeout", type=int, default=300, help="Per-instance time limit (s)")
    parser.add_argument("--search", choices=["binary", "linear"], default="binary", help="Search strategy")
    args = parser.parse_args()

    files: List[Path] = []
    for pat in args.instances:
        p = Path(pat)
        if p.is_dir():
            files.extend(sorted(p.glob("*.dat")))
        else:
            base_dir = p.parent if p.parent != Path('.') else Path('.')
            files.extend(sorted(base_dir.glob(p.name)))
    files = sorted(list(set(files)))

    if not files:
        print("No instance files found.", file=sys.stderr)
        sys.exit(1)

    for f in files:
        try:
            inst = load_instance(f)
            opt_val, tours, optimal = optimise(
                inst, timeout=args.timeout, strategy=args.search
            )
            
            sol = [[item_idx + 1 for item_idx in route] for route in tours]
            print(f"Result: obj={opt_val}, optimal={optimal}, sol={sol}")

        except RuntimeError as e:
            print(f"[ERROR] {f.name}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred on {f.name}: {e}", file=sys.stderr)
            traceback.print_exc()
        print("-" * 20)

if __name__ == "__main__":
    main()