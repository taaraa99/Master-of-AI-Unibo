import argparse
import os
from unified_solver import UnifiedSolver

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('solver', choices=['cp','mip','smt','sat'])
    parser.add_argument('--search', choices=['binary','linear','optimize'], default='binary')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--knn', type=int)
    parser.add_argument('--lns', action='store_true')
    parser.add_argument('--lns-iters', type=int)
    parser.add_argument('--destroy-frac', type=float)
    parser.add_argument('--outdir', default='res')
    parser.add_argument('instances', nargs='*', default=['Instances/*.dat'])
    args=parser.parse_args()
    unified=UnifiedSolver(args.solver, base_dir='.', outdir=args.outdir,
                           search=args.search, timeout=args.timeout,
                           knn=args.knn, lns=args.lns,
                           lns_iters=args.lns_iters,
                           destroy_frac=args.destroy_frac)
    unified.solve_all_instances()

if __name__ == "__main__":
    main()
