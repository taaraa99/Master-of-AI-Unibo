#!/usr/bin/env python3
"""
Batch SMT-based solver for the Multiple Couriers Problem (MCP) using Z3.
Processes multiple .dat/.txt instances with binary/linear/optimize search.
Writes per-instance JSON under res/SMT/<index>.json and prints results to stdout.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
import math
from z3 import Solver, Optimize, Bool, Int, If, PbEq, Sum, And, Or, Not, is_true, sat


def parse_instance(path):
    """
    Reads a MCP instance file (m, n, capacities, weights, distance matrix).
    """
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    m = int(lines[0]); n = int(lines[1])
    capacities = list(map(int, lines[2].split()))
    weights    = list(map(int, lines[3].split()))
    D = [list(map(int, row.split())) for row in lines[4:4+n+1]]
    return m, n, capacities, weights, D


def build_smt_routing(m, n, weights, capacities, D,
                       search='binary', timeout_s=300):
    start = perf_counter()
    origin = n
    # Decision vars
    x = [[ Bool(f"x_{c}_{i}") for i in range(n)] for c in range(m)]
    y = [[[ Bool(f"y_{c}_{i}_{j}") for j in range(n+1)] for i in range(n+1)] for c in range(m)]
    u = [[ Int(f"u_{c}_{i}") for i in range(n+1)] for c in range(m)]
    max_dist = Int('max_dist')

    def add_constraints(slv):
        # capacity
        for c in range(m):
            slv.add(Sum([If(x[c][i], weights[i], 0) for i in range(n)]) <= capacities[c])
        # assign exactly once
        for i in range(n):
            slv.add(PbEq([(x[c][i], 1) for c in range(m)], 1))
        # flow and depot
        for c in range(m):
            slv.add(PbEq([(y[c][origin][j],1) for j in range(n)],1))
            slv.add(PbEq([(y[c][i][origin],1) for i in range(n)],1))
            for i in range(n):
                slv.add(PbEq([(y[c][i][j],1) for j in range(n+1)], If(x[c][i],1,0)))
                slv.add(PbEq([(y[c][j][i],1) for j in range(n+1)], If(x[c][i],1,0)))
        # no self loops
        for c in range(m):
            for i in range(n+1): slv.add(Not(y[c][i][i]))
        # MTZ subtours
        for c in range(m):
            for i in range(1,n+1): slv.add(And(u[c][i]>=1, u[c][i]<=n))
            for i in range(n+1):
                for j in range(n+1):
                    if i!=j: slv.add(If(y[c][i][j], u[c][i]+1==u[c][j], True))
            for j in range(n): slv.add(If(y[c][origin][j], u[c][j]==1, True))
        # max distance constraint
        for c in range(m):
            expr = Sum([If(y[c][i][j], D[i][j], 0)
                        for i in range(n+1) for j in range(n+1)])
            slv.add(expr <= max_dist)

    # Select search mode
    if search == 'optimize':
        solver = Optimize()
        solver.set('timeout', timeout_s*1000)
        add_constraints(solver)
        solver.minimize(max_dist)
        sat_res = solver.check()
        model = solver.model() if sat_res == sat else None
    else:
        solver = Solver(); solver.set('timeout', timeout_s*1000)
        add_constraints(solver)
        low, high = 0, sum(max(row) for row in D)
        model = None;
        if search == 'linear':
            for val in range(low, high+1):
                solver.push(); solver.add(max_dist <= val)
                if solver.check() == sat:
                    model = solver.model(); solver.pop(); break
                solver.pop()
        else:  # binary
            L, H = low, high
            while L <= H:
                mid = (L+H)//2
                solver.push(); solver.add(max_dist <= mid)
                if solver.check() == sat:
                    model = solver.model(); H = mid-1
                else:
                    L = mid+1
                solver.pop()
    elapsed = perf_counter() - start
    t = min(int(elapsed), timeout_s)
    if not model:
        return None, t, None
    best = model[max_dist].as_long()
    # extract tours
    tours = []
    for c in range(m):
        route = []
        cur = origin
        while True:
            nxt = None
            for j in range(n+1):
                if is_true(model.evaluate(y[c][cur][j])):
                    nxt = j; break
            if nxt is None or nxt == origin:
                break
            route.append(nxt+1)
            cur = nxt
        tours.append(route)
    return best, t, tours


def main():
    parser = argparse.ArgumentParser(
        description="Batch solve MCP instances via SMT (Z3)"
    )
    parser.add_argument(
        "instances", nargs="*",
        default=["Instances"],
        help=".dat/.txt files or glob patterns (e.g. inst*.dat)"
    )
    parser.add_argument(
        "--search", choices=["binary","linear","optimize"],
        default="binary", help="search strategy"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="time limit per instance (seconds)"
    )
    parser.add_argument(
        "--outdir", default="res/SMT",
        help="directory for JSON results"
    )
    args = parser.parse_args()

    patterns = args.instances if args.instances else ["inst*.dat"]
    files = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            files.extend(sorted(p.glob("*.*")))
        else:
            files.extend(sorted(Path('.').glob(pat)))
    files = sorted({f for f in files if f.is_file()})
    if not files:
        print("No instance files found.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        # header = f"=== SMT {args.search} on {f.name} ==="
        # print(header)
        best, t, tours = build_smt_routing(
            *parse_instance(f),
            search=args.search, timeout_s=args.timeout
        )
        optimal = (t < args.timeout and best is not None)
        sol = tours if tours else [[] for _ in range(len(parse_instance(f)[0]))]
        approach = f"smt_{args.search}"
        record = {"time": t, "optimal": optimal,
                  "obj": best or 0, "sol": sol}
        # print result
        print(json.dumps({approach: record}, indent=2))
        # merge into file
        digits = ''.join(filter(str.isdigit, f.stem)) or f.stem
        out_file = out_dir / f"{digits}.json"
        if out_file.exists():
            full = json.loads(out_file.read_text())
        else:
            full = {}
        full[approach] = record
        out_file.write_text(json.dumps(full, indent=2))
        print(f"→ Updated {out_file}\n")

if __name__ == "__main__":
    main()