#!/usr/bin/env python3
"""
smt_linear.py  –  SMT-based solver for the Multiple Couriers Planning (MCP) problem,
with JSON output compliant with CDMO, batch-mode over directories, and numeric output filenames.

Usage (single instance):
    python smt_linear.py inst01.dat
    python smt_linear.py inst01.dat --approach z3_smt --output results/01.json

Usage (batch mode):
    python smt_linear.py --input-dir instances/ --output-dir results/ --approach z3_smt

Options:
    instance      path to a single .dat instance (mutually exclusive with --input-dir)
    --input-dir   directory containing .dat instances (mutually exclusive with instance)
    --output      path of JSON file for single-instance mode (prints to stdout if omitted)
    --output-dir  directory in which to write one JSON per instance (defaults to cwd)
    --timeout     wall-clock seconds per instance (default 300)
    --depot-first rotate the first row/column to the end before modelling
    --approach    name of this approach in the JSON (default "smt_z3")
"""
import argparse, time, sys, re, json, math, os, glob
from z3 import *

# ───────────────────────────  Mini helpers  ────────────────────────────────
def exactly_one(bools, tag=""):
    atleast = Or(bools)
    if len(bools) == 1:
        return atleast
    s = [Bool(f"s_{tag}_{i}") for i in range(len(bools) - 1)]
    ladder = [Or(Not(bools[0]), s[0]), Or(Not(bools[-1]), Not(s[-1]))]
    for i in range(1, len(bools) - 1):
        ladder += [
            Or(Not(bools[i]), s[i]),
            Or(Not(bools[i]), Not(s[i-1])),
            Or(Not(s[i-1]), s[i]),
        ]
    return And(atleast, And(ladder))

def lex_leq(a, b):
    less  = a[0] <  b[0]
    equal = a[0] == b[0]
    for i in range(1, len(a)):
        less  = Or(less,  And(equal, a[i] < b[i]))
        equal = And(equal, a[i] == b[i])
    return Or(equal, less)

# ───────────────────────────  Robust parser  ───────────────────────────────
_split = re.compile(r"[,\s]+")
def _next_numbers(fh):
    for raw in fh:
        line = raw.split("#", 1)[0].strip()
        if line:
            return [int(t) for t in _split.split(line) if t]
    raise ValueError("unexpected EOF")

def parse_instance(path, depot_first):
    with open(path) as fh:
        nums = _next_numbers(fh)
        m, n = nums if len(nums) == 2 else (nums[0], _next_numbers(fh)[0])
        caps  = _next_numbers(fh)
        sizes = [0] + _next_numbers(fh)
        if len(caps) != m or len(sizes) != n + 1:
            raise ValueError("capacity/size mismatch")
        D = [_next_numbers(fh) for _ in range(n + 1)]
        for r, row in enumerate(D):
            if len(row) != n + 1:
                raise ValueError(f"distance row {r} has length {len(row)}")
        if depot_first:
            depot_row = D.pop(0)
            for row in D:
                row.append(row.pop(0))
            D.append(depot_row)
    return m, n, caps, sizes, D  # depot is at index n

# ───────────────────────────  Model building  ──────────────────────────────
def build_solver(m, n, caps, sizes, D, timeout_ms):
    s = Solver()
    s.set("timeout", timeout_ms)
    depot = n + 1
    x   = [[Int(f"x_{k}_{i}") for i in range(n + 1)] for k in range(m)]
    ld  = [Int(f"load_{k}") for k in range(m)]
    dst = [Int(f"dist_{k}") for k in range(m)]
    rho = Int("rho")

    # domains + force tail-depot
    for k in range(m):
        for i in range(n + 1):
            s.add(And(1 <= x[k][i], x[k][i] <= depot))
        for i in range(n):
            s.add(Implies(x[k][i] == depot, x[k][i+1] == depot))
        s.add(x[k][0] != depot)

    # each item exactly once
    for itm in range(1, n + 1):
        s.add(exactly_one(
            [x[k][i] == itm for k in range(m) for i in range(n + 1)],
            tag=f"item{itm}"
        ))

    # loads
    def size_of(v):
        res = 0
        for it in range(1, n + 1):
            res = If(v == it, sizes[it], res)
        return res

    for k in range(m):
        s.add(ld[k] == Sum([size_of(x[k][i]) for i in range(n + 1)]))
        s.add(ld[k] <= caps[k])

    # distance(a,b)
    def d(a, b):
        terms = []
        for p in range(1, n + 2):
            for q in range(1, n + 2):
                row = n if p == depot else p - 1
                col = n if q == depot else q - 1
                terms.append(If(And(a == p, b == q), D[row][col], 0))
        return Sum(terms)

    for k in range(m):
        legs, prev = [], depot
        for i in range(n + 1):
            legs.append(d(prev, x[k][i]))
            prev = x[k][i]
        legs.append(d(prev, depot))
        s.add(dst[k] == Sum(legs))

    # objective = minimize the maximum route distance
    maxd = dst[0]
    for d_i in dst[1:]:
        maxd = If(d_i > maxd, d_i, maxd)
    s.add(rho == maxd)

    # symmetry breaking when capacities equal
    for k in range(m - 1):
        if caps[k] == caps[k + 1]:
            s.add(lex_leq(x[k], x[k + 1]))

    return s, rho, x

# ───────────────────────────  Linear search  ───────────────────────────────
def linear_search(sol, rho, timeout_ms):
    start = time.time()
    best = None
    optimal = True
    while True:
        st = sol.check()
        if st == sat:
            best = sol.model()
            val  = best.eval(rho).as_long()
            elapsed = (time.time() - start) * 1000
            sol.set("timeout", max(timeout_ms - int(elapsed), 1))
            sol.add(rho < val)
        else:
            if best is None:
                return None, False
            if st == unknown and sol.reason_unknown() == "timeout":
                optimal = False
            break
    return best, optimal

# ───────────────────────────  Extract paths  ───────────────────────────────
def paths(model, x, m, n, depot):
    res = []
    for k in range(m):
        seq = [model.eval(x[k][i]).as_long() for i in range(n + 1)]
        if depot in seq:
            seq = seq[:seq.index(depot)]
        res.append(seq)  # 1-based
    return res

# ───────────────────────────  Main  ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", nargs="?", help="path to a single .dat file")
    ap.add_argument("--input-dir", dest="input_dir",
                    help="directory containing .dat instances")
    ap.add_argument("--output", help="JSON file path for single-instance mode")
    ap.add_argument("--output-dir", dest="output_dir",
                    help="directory to write one JSON per instance")
    ap.add_argument("--timeout",   type=int, default=300,
                    help="seconds per instance (default 300)")
    ap.add_argument("--depot-first", action="store_true",
                    help="rotate the first row/col to the end before modelling")
    ap.add_argument("--approach",  default="smt_z3",
                    help="name of this approach in the JSON")
    args = ap.parse_args()

    # Batch mode?
    if args.input_dir:
        IN  = args.input_dir
        OUT = args.output_dir or "."
        os.makedirs(OUT, exist_ok=True)
        files = sorted(glob.glob(os.path.join(IN, "*.dat")))
        for path in files:
            base = os.path.basename(path)
            mnum = re.search(r"(\d+)(?=\.dat$)", base)
            key  = mnum.group(1) if mnum else os.path.splitext(base)[0]

            start = time.time()
            try:
                m, n, caps, sizes, D = parse_instance(path, args.depot_first)
                solver, rho, x = build_solver(m, n, caps, sizes, D, args.timeout*1000)
                model, optimal = linear_search(solver, rho, args.timeout*1000)
            except Exception as e:
                print(f"[ERROR] {base}: {e}", file=sys.stderr)
                model = None
                optimal = False
            secs = time.time() - start
            t = min(int(math.floor(secs)), args.timeout)

            if model is None:
                sol_list = [[] for _ in range(m)]
                obj_val   = None
                opt_flag  = False
            else:
                obj_val  = model.eval(rho).as_long()
                sol_list = paths(model, x, m, n, n+1)
                opt_flag = optimal

            result = {
                args.approach: {
                    "time":    t,
                    "optimal": opt_flag,
                    "obj":     obj_val,
                    "sol":     sol_list
                }
            }

            out_path = os.path.join(OUT, f"{key}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Wrote {out_path}")
        return

    # Single-instance mode
    if not args.instance:
        ap.error("Please specify either an instance file or --input-dir")

    start = time.time()
    try:
        m, n, caps, sizes, D = parse_instance(args.instance, args.depot_first)
        solver, rho, x = build_solver(m, n, caps, sizes, D, args.timeout*1000)
        model, optimal = linear_search(solver, rho, args.timeout*1000)
    except Exception as e:
        print(f"[ERROR] {args.instance}: {e}", file=sys.stderr)
        model = None
        optimal = False

    secs = time.time() - start
    t = min(int(math.floor(secs)), args.timeout)

    if model is None:
        sol_list = [[] for _ in range(m)]
        obj_val   = None
        opt_flag  = False
    else:
        obj_val  = model.eval(rho).as_long()
        sol_list = paths(model, x, m, n, n+1)
        opt_flag = optimal

    result = {
        args.approach: {
            "time":    t,
            "optimal": opt_flag,
            "obj":     obj_val,
            "sol":     sol_list
        }
    }

    out = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
    else:
        print(out)

if __name__ == "__main__":
    main()
