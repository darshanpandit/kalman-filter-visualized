"""Benchmark metrics for evaluating filter performance."""

from __future__ import annotations

import time

import numpy as np


def position_rmse(x_estimates: list, true_states: np.ndarray) -> float:
    """Root Mean Square Error of position estimates.

    Parameters
    ----------
    x_estimates : list of np.ndarray
        Filter output (each element has at least 2 components: x, y).
    true_states : np.ndarray (N+1, >=2)
        True states. x_estimates[k] corresponds to true_states[k+1].

    Returns
    -------
    float — RMSE in position units.
    """
    est = np.array([x[:2] for x in x_estimates])
    true_pos = true_states[1 : len(est) + 1, :2]
    return float(np.sqrt(np.mean(np.sum((est - true_pos) ** 2, axis=1))))


def position_mae(x_estimates: list, true_states: np.ndarray) -> float:
    """Mean Absolute Error of position estimates."""
    est = np.array([x[:2] for x in x_estimates])
    true_pos = true_states[1 : len(est) + 1, :2]
    return float(np.mean(np.linalg.norm(est - true_pos, axis=1)))


def per_step_errors(x_estimates: list, true_states: np.ndarray) -> np.ndarray:
    """Per-step Euclidean position errors.

    Returns
    -------
    np.ndarray (N,) — error at each step.
    """
    est = np.array([x[:2] for x in x_estimates])
    true_pos = true_states[1 : len(est) + 1, :2]
    return np.linalg.norm(est - true_pos, axis=1)


def nees(
    x_estimates: list,
    P_estimates: list,
    true_states: np.ndarray,
) -> np.ndarray:
    """Normalized Estimation Error Squared (filter consistency check).

    For a consistent filter, NEES should average to the state dimension.

    Returns
    -------
    np.ndarray (N,) — NEES at each step.
    """
    n_steps = len(x_estimates)
    values = np.zeros(n_steps)
    for k in range(n_steps):
        x_est = np.array(x_estimates[k])
        x_true = true_states[k + 1]
        # Use only position components that match the estimate dimension
        dim = min(len(x_est), len(x_true))
        err = x_est[:dim] - x_true[:dim]
        P = np.array(P_estimates[k])[:dim, :dim]
        try:
            P_inv = np.linalg.inv(P)
            values[k] = float(err @ P_inv @ err)
        except np.linalg.LinAlgError:
            values[k] = np.nan
    return values


def computation_time(
    filter_factory,
    measurements: np.ndarray,
    n_runs: int = 5,
) -> dict:
    """Measure filter execution time.

    Parameters
    ----------
    filter_factory : callable() -> filter
        Zero-argument callable that creates a fresh filter instance.
    measurements : np.ndarray
        Measurement sequence to run through.
    n_runs : int
        Number of timing runs (returns mean and std).

    Returns
    -------
    dict with keys: mean_s, std_s, n_runs.
    """
    times = []
    for _ in range(n_runs):
        filt = filter_factory()
        t0 = time.perf_counter()
        filt.run(measurements)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    return {
        "mean_s": float(times.mean()),
        "std_s": float(times.std()),
        "n_runs": n_runs,
    }
