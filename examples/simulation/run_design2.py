"""
Run multiple Design 2 replicates and write summary CSV.

Usage (from cpit package root):
  python -m examples.simulation.run_design2 --n_reps 100 --base_seed 42
"""

import argparse
from pathlib import Path

from .design2 import PIT_SEED, run_design2_replicate
from .runner import ALPHAS, TAU_GRID, aggregate_replicates, aggregate_local_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Design 2 replicates.")
    parser.add_argument("--n_reps", type=int, default=100)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument(
        "--summary_path",
        type=str,
        default="results/simulation/design2/design2_rep_summary.csv",
    )
    parser.add_argument("--progress_path", type=str, default="")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "gam"])
    args = parser.parse_args()

    summary_path = Path(args.summary_path)
    progress_path = (
        Path(args.progress_path)
        if args.progress_path
        else summary_path.parent / "design2_rep_progress.csv"
    )

    print(
        f"Design 2: {args.n_reps} replicates "
        f"(base_seed={args.base_seed}, mode={args.mode})",
        flush=True,
    )

    replicate_results = []
    for r in range(args.n_reps):
        seed = args.base_seed + r
        print(f"Replicate {r + 1}/{args.n_reps} (seed={seed})...", flush=True)
        res = run_design2_replicate(seed, pit_seed=PIT_SEED, alphas=ALPHAS, tau_grid=TAU_GRID, mode=args.mode)
        replicate_results.append(res)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_replicates(replicate_results, mode=args.mode).to_csv(progress_path)
        print(f"  Progress saved ({len(replicate_results)} reps).", flush=True)

    aggregate_replicates(replicate_results, mode=args.mode).to_csv(summary_path)
    local_path = summary_path.parent / summary_path.name.replace(
        "_rep_summary", "_local_diag"
    )
    aggregate_local_diagnostics(replicate_results).to_csv(local_path, index=False)
    print(f"Saved summary to {summary_path} (n_reps={args.n_reps}).")
    print(f"Saved local diagnostics to {local_path}.")


if __name__ == "__main__":
    main()
