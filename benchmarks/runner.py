"""Run filters on trajectories and collect metrics."""

from __future__ import annotations

import numpy as np

from benchmarks.configs import make_all_filters, FILTER_NAMES
from benchmarks.metrics import position_rmse, position_mae, per_step_errors, nees


def run_single_trajectory(
    trajectory: dict,
    filter_names: list[str] | None = None,
    pf_particles: int = 300,
    pf_seed: int = 42,
) -> dict:
    """Run all (or selected) filters on one trajectory and return metrics.

    Parameters
    ----------
    trajectory : dict
        Must have keys: true_states, measurements, dt.
        true_states must have at least 4 columns [x, y, vx, vy].
    filter_names : list or None
        Subset of FILTER_NAMES to run. None = all.
    pf_particles : int
        Number of particles for PF.
    pf_seed : int
        Seed for PF.

    Returns
    -------
    dict mapping filter_name -> {rmse, mae, per_step, nees_mean}.
    """
    true_states = trajectory["true_states"]
    measurements = trajectory["measurements"]
    dt = trajectory["dt"]

    # Ensure 4-column state for filter initialization
    x0 = true_states[0]
    if len(x0) > 4:
        x0 = x0[:4]

    filters = make_all_filters(
        dt=dt, x0=x0, pf_particles=pf_particles, pf_seed=pf_seed,
    )

    if filter_names is None:
        filter_names = FILTER_NAMES

    # Ensure true_states has at least 4 cols for metric computation
    ts_4col = true_states[:, :4] if true_states.shape[1] > 4 else true_states

    results = {}
    for name in filter_names:
        filt = filters[name]
        res = filt.run(measurements)
        rmse = position_rmse(res["x_estimates"], ts_4col)
        mae = position_mae(res["x_estimates"], ts_4col)
        errors = per_step_errors(res["x_estimates"], ts_4col)
        nees_vals = nees(res["x_estimates"], res["P_estimates"], ts_4col)
        results[name] = {
            "rmse": rmse,
            "mae": mae,
            "per_step": errors,
            "nees_mean": float(np.nanmean(nees_vals)),
        }

    return results


def run_corpus(
    corpus: list[dict],
    filter_names: list[str] | None = None,
    pf_particles: int = 300,
    pf_seed: int = 42,
) -> dict:
    """Run filters on an entire corpus and aggregate results.

    Returns
    -------
    dict with keys:
        per_trajectory : list of per-trajectory result dicts
        summary : dict mapping filter_name -> {mean_rmse, std_rmse, mean_mae, ...}
    """
    if filter_names is None:
        filter_names = FILTER_NAMES

    per_trajectory = []
    for i, traj in enumerate(corpus):
        try:
            result = run_single_trajectory(
                traj,
                filter_names=filter_names,
                pf_particles=pf_particles,
                pf_seed=pf_seed + i,
            )
            per_trajectory.append(result)
        except Exception:
            # Skip failed trajectories (e.g. too short, singular matrices)
            per_trajectory.append(None)

    # Aggregate
    summary = {}
    for name in filter_names:
        rmses = [
            r[name]["rmse"]
            for r in per_trajectory
            if r is not None and name in r
        ]
        maes = [
            r[name]["mae"]
            for r in per_trajectory
            if r is not None and name in r
        ]
        summary[name] = {
            "mean_rmse": float(np.mean(rmses)) if rmses else float("nan"),
            "std_rmse": float(np.std(rmses)) if rmses else float("nan"),
            "mean_mae": float(np.mean(maes)) if maes else float("nan"),
            "std_mae": float(np.std(maes)) if maes else float("nan"),
            "n_trajectories": len(rmses),
        }

    return {
        "per_trajectory": per_trajectory,
        "summary": summary,
    }
