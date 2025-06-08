import argparse
import sys
import os
import json
from pathlib import Path
from time import perf_counter
from typing import List

# Ensure the current directory (this script's folder) is on PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from .instance import load_instance
from .solver import optimise, lns_optimise
from .utils import save_result
from typing import List, Tuple
from .instance import load_instance, Instance




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
        # prepare sub‐instances



        
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
        # sol     = [[j+1 for j in sorted(route)] for route in tours]
        sol = [[node+1 for node in route] for route in tours]

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