#!/usr/bin/env python3
"""KT-2 Robustness Grid: Replicability Test Across Seeds and Dimensions

Runs the decisive 1-step test across a predeclared grid to demonstrate
that the negative result is robust across model initializations and scales.

Grid:
  - k = 1 (decisive test only)
  - dim ∈ {8, 12, 16}
  - seeds ∈ {0..9} (10 seeds per dim)
  - Total: 3 × 10 = 30 runs

Outputs:
  - results/kt2_robustness_grid.json (full provenance + raw data + summary)
  - results/kt2_robustness_grid.csv (tabular summary)

Usage:
    python -m experiments.kt2_robustness_grid
    python -m experiments.kt2_robustness_grid --dims 8 12 16 --seeds 10
    python -m experiments.kt2_robustness_grid --quiet
"""

import argparse
import json
import os
import csv
from typing import List, Dict, Any
import numpy as np

from experiments.kt2_locality_falsifier import (
    create_model,
    run_decisive_1step,
    get_provenance_metadata,
    CI_THRESHOLD,
    DEFAULT_HIDDEN,
)


def run_robustness_grid(
    dims: List[int] = None,
    n_seeds: int = 10,
    output_dir: str = "results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run robustness grid across dims and seeds.

    Args:
        dims: List of matrix dimensions to test
        n_seeds: Number of seeds per dimension
        output_dir: Directory for output files
        verbose: Print progress

    Returns:
        Dict with full results, summary stats, and provenance
    """

    if dims is None:
        dims = [8, 12, 16]

    if verbose:
        print("\n" + "="*70)
        print("KT-2 ROBUSTNESS GRID")
        print("="*70)
        print(f"\n  Grid: dim ∈ {dims}, seeds ∈ [0..{n_seeds-1}]")
        print(f"  Total runs: {len(dims) * n_seeds}")

    results = []
    seeds = list(range(n_seeds))

    run_count = 0
    total_runs = len(dims) * n_seeds

    for dim in dims:
        for seed in seeds:
            run_count += 1
            if verbose:
                print(f"\n[{run_count}/{total_runs}] dim={dim}, seed={seed}")

            # Create model with this seed and dim
            model = create_model(seed=seed, dim=dim, hidden=DEFAULT_HIDDEN)

            # Run decisive test (suppress its verbose output)
            result = run_decisive_1step(model, verbose=False)

            # Extract data for this run
            row = {
                "dim": dim,
                "seed": seed,
                "verdict": result["verdict"],
            }

            # Extract per-constraint CI and pass/fail
            for constraint, data in result["decisive_table"].items():
                row[f"ci_{constraint}"] = data["ci_1step"]
                row[f"below_threshold_{constraint}"] = data["below_threshold"]

            results.append(row)

            if verbose:
                verdict_symbol = "✗" if row["verdict"] == "FALSIFIED" else "✓"
                print(f"  {verdict_symbol} {row['verdict']}")

    # Compute summary statistics
    all_verdicts = [r["verdict"] for r in results]
    failure_count = sum(1 for v in all_verdicts if v == "FALSIFIED")
    failure_rate = failure_count / len(all_verdicts)

    # Per-constraint CI statistics
    constraint_stats = {}
    constraint_names = [k.replace("ci_", "") for k in results[0].keys() if k.startswith("ci_")]

    for constraint in constraint_names:
        ci_values = [r[f"ci_{constraint}"] for r in results]
        constraint_stats[constraint] = {
            "median_ci": float(np.median(ci_values)),
            "iqr_ci": float(np.percentile(ci_values, 75) - np.percentile(ci_values, 25)),
            "mean_ci": float(np.mean(ci_values)),
            "std_ci": float(np.std(ci_values)),
            "n_below_threshold": sum(1 for ci in ci_values if ci < CI_THRESHOLD),
            "n_total": len(ci_values),
        }

    summary = {
        "grid_spec": {
            "dims": dims,
            "seeds": seeds,
            "n_runs": len(results),
        },
        "failure_rate": failure_rate,
        "failure_count": failure_count,
        "total_count": len(all_verdicts),
        "threshold": CI_THRESHOLD,
        "constraint_stats": constraint_stats,
    }

    if verbose:
        print("\n" + "="*70)
        print("GRID SUMMARY")
        print("="*70)
        print(f"\n  Failure rate: {failure_rate:.1%} ({failure_count}/{len(all_verdicts)})")
        print(f"\n  Per-constraint CI statistics:")
        for constraint, stats in constraint_stats.items():
            print(f"    {constraint:10s}: median={stats['median_ci']:.3f}, "
                  f"IQR={stats['iqr_ci']:.3f}, "
                  f"n<τ={stats['n_below_threshold']}/{stats['n_total']}")

    # Build final output
    output = {
        "protocol_id": "KT-2",
        "test": "robustness_grid",
        "meta": get_provenance_metadata("robustness-grid"),
        "summary": summary,
        "results": results,
    }

    return output


def save_grid_results(data: Dict[str, Any], output_dir: str = "results"):
    """Save grid results to JSON and CSV.

    Args:
        data: Results dict from run_robustness_grid
        output_dir: Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # JSON (full provenance + raw data + summary)
    json_path = os.path.join(output_dir, "kt2_robustness_grid.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    # CSV (tabular per-run rows)
    csv_path = os.path.join(output_dir, "kt2_robustness_grid.csv")
    with open(csv_path, "w", newline="") as f:
        if data["results"]:
            writer = csv.DictWriter(f, fieldnames=data["results"][0].keys())
            writer.writeheader()
            writer.writerows(data["results"])
    print(f"  Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="KT-2 Robustness Grid: Test replicability across seeds and dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dims", type=int, nargs="+", default=[8, 12, 16],
                        help="Dimensions to test (default: 8 12 16)")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds to test per dimension (default: 10)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()

    results = run_robustness_grid(
        dims=args.dims,
        n_seeds=args.seeds,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    save_grid_results(results, args.output_dir)

    if not args.quiet:
        print("\n" + "="*70)
        print("ROBUSTNESS GRID COMPLETE")
        print("="*70)


if __name__ == "__main__":
    main()
