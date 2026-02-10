"""CLI script: precompute benchmark results to .npz files.

Usage:
    PYTHONPATH=. python3 benchmarks/precompute.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.sweep import sweep_turn_rate
from benchmarks.corpus import generate_synthetic_corpus, load_real_corpus
from benchmarks.runner import run_corpus
from benchmarks.configs import (
    FILTER_NAMES, make_kf, make_ekf, make_ukf, make_pf,
)
from benchmarks.metrics import computation_time
from kalman_manim.data.generators import generate_linear_trajectory


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "benchmark_results",
)


def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def precompute_sweep():
    """Run turn_rate sweep and save results."""
    print("=== Sweep: RMSE vs turn_rate ===")
    t0 = time.time()
    results = sweep_turn_rate(
        turn_rates=np.linspace(0, 0.5, 25),
        n_trials_per_rate=50,
        n_steps=60,
        base_seed=42,
    )
    elapsed = time.time() - t0
    print(f"  Sweep completed in {elapsed:.1f}s")

    path = os.path.join(OUTPUT_DIR, "sweep_results.npz")
    np.savez(
        path,
        turn_rates=results["turn_rates"],
        filter_names=np.array(results["filter_names"]),
        rmse=results["rmse"],
        mean_rmse=results["mean_rmse"],
        std_rmse=results["std_rmse"],
    )
    print(f"  Saved to {path}")
    return results


def precompute_corpus():
    """Run all filters on synthetic + real corpus and save."""
    print("=== Corpus benchmark ===")

    print("  Generating synthetic corpus...")
    synthetic = generate_synthetic_corpus(n_per_regime=30, base_seed=100)
    print(f"  {len(synthetic)} synthetic trajectories")

    print("  Loading real corpus...")
    real = load_real_corpus(min_steps=20, noise_std=0.5, base_seed=200)
    print(f"  {len(real)} real trajectories")

    corpus = synthetic + real
    print(f"  Total corpus: {len(corpus)} trajectories")

    t0 = time.time()
    results = run_corpus(corpus, pf_particles=300)
    elapsed = time.time() - t0
    print(f"  Corpus benchmark completed in {elapsed:.1f}s")

    # Pack results for npz
    n_traj = len(corpus)
    rmse_data = np.full((n_traj, len(FILTER_NAMES)), np.nan)
    mae_data = np.full((n_traj, len(FILTER_NAMES)), np.nan)
    regimes = []

    for i, r in enumerate(results["per_trajectory"]):
        regimes.append(corpus[i].get("regime", "unknown"))
        if r is not None:
            for fi, fname in enumerate(FILTER_NAMES):
                if fname in r:
                    rmse_data[i, fi] = r[fname]["rmse"]
                    mae_data[i, fi] = r[fname]["mae"]

    path = os.path.join(OUTPUT_DIR, "corpus_results.npz")
    np.savez(
        path,
        filter_names=np.array(FILTER_NAMES),
        rmse=rmse_data,
        mae=mae_data,
        regimes=np.array(regimes),
        # Summary stats
        **{
            f"summary_{k}_{stat}": v
            for k, vals in results["summary"].items()
            for stat, v in vals.items()
            if isinstance(v, (int, float))
        },
    )
    print(f"  Saved to {path}")
    return results


def precompute_timing():
    """Measure computation time for each filter."""
    print("=== Timing benchmark ===")

    # Use a standard 60-step linear trajectory for fair comparison
    data = generate_linear_trajectory(n_steps=60, dt=0.5, seed=999)
    meas = data["measurements"]
    x0 = data["true_states"][0]
    dt = data["dt"]

    timing = {}
    for name in FILTER_NAMES:
        if name == "KF":
            factory = lambda: make_kf(dt, x0)
        elif name == "EKF":
            factory = lambda: make_ekf(dt, x0)
        elif name == "UKF":
            factory = lambda: make_ukf(dt, x0)
        elif name == "PF":
            factory = lambda: make_pf(dt, x0, n_particles=300, seed=42)
        else:
            continue

        result = computation_time(factory, meas, n_runs=10)
        timing[name] = result
        print(f"  {name}: {result['mean_s']*1000:.2f}ms ± {result['std_s']*1000:.2f}ms")

    path = os.path.join(OUTPUT_DIR, "timing_results.npz")
    np.savez(
        path,
        filter_names=np.array(FILTER_NAMES),
        mean_s=np.array([timing[n]["mean_s"] for n in FILTER_NAMES]),
        std_s=np.array([timing[n]["std_s"] for n in FILTER_NAMES]),
    )
    print(f"  Saved to {path}")
    return timing


def main():
    _ensure_dir()
    print(f"Output directory: {OUTPUT_DIR}\n")

    precompute_sweep()
    print()
    precompute_corpus()
    print()
    precompute_timing()
    print("\nAll benchmarks complete.")


if __name__ == "__main__":
    main()
