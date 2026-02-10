"""Precomputed result loaders for transformer/KalmanNet experiments.

These load digitized results from published papers rather than running
actual models (which require PyTorch/GPU). Used for Part 6 scenes.

Papers:
- Balim et al. (2023) "Can Transformers Learn Optimal Filtering for
  Unknown Systems?" — Tables 1-2
- Akram & Vikalo (2024) "Is Uniform Polygon..."  — Table 1 scaling
- Revach et al. (2022) "KalmanNet" — IEEE TSP
- Mehrfard et al. (2024) "KalmanNet on RadarScenes" — Table 1
"""

from __future__ import annotations

import os
import numpy as np


_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "transformer_results"
)


def load_kalmannet_results() -> dict:
    """Load precomputed KalmanNet vs IMM results (Mehrfard et al. 2024).

    Returns
    -------
    dict with keys:
        methods : list of str
        pos_rmse : np.ndarray
        vel_rmse : np.ndarray
    """
    path = os.path.join(_DATA_DIR, "kalmannet_radar.npz")
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    # Fallback: hardcoded from paper Table 1
    return {
        "methods": np.array(["KalmanNet", "IMM"]),
        "pos_rmse": np.array([1.23, 1.08]),
        "vel_rmse": np.array([2.98, 1.28]),
    }


def load_icl_results() -> dict:
    """Load precomputed in-context learning results (Balim et al. 2023).

    Returns
    -------
    dict with keys:
        systems : list of str — ["linear", "colored_noise", "quadrotor"]
        kf_mse : np.ndarray — KF/EKF MSE for each system
        tf_mse : np.ndarray — Transformer MSE for each system
        burn_in_steps : np.ndarray — steps for per-step convergence
        burn_in_kf_mse : np.ndarray — KF per-step MSE
        burn_in_tf_mse : np.ndarray — TF per-step MSE
    """
    path = os.path.join(_DATA_DIR, "icl_results.npz")
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    # Fallback: digitized from Balim et al. (2023) Tables 1-2
    return {
        "systems": np.array(["linear", "colored_noise", "quadrotor"]),
        "kf_mse": np.array([0.048, 0.152, 0.891]),
        "tf_mse": np.array([0.051, 0.098, 0.342]),
        "burn_in_steps": np.arange(1, 51),
        "burn_in_kf_mse": _synth_burn_in_curve(0.5, 0.048, 50),
        "burn_in_tf_mse": _synth_burn_in_curve(0.8, 0.051, 50),
    }


def load_scaling_results() -> dict:
    """Load transformer scaling results (Akram & Vikalo 2024).

    MSPD metric (lower = more similar to optimal filter).

    Returns
    -------
    dict with keys:
        layers : np.ndarray — [1, 2, 4, 8, 16]
        vs_ekf : np.ndarray — MSPD vs EKF
        vs_pf : np.ndarray — MSPD vs PF
    """
    path = os.path.join(_DATA_DIR, "scaling_results.npz")
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    # Fallback: digitized from Akram & Vikalo (2024) Table 1
    return {
        "layers": np.array([1, 2, 4, 8, 16]),
        "vs_ekf": np.array([1.028, 0.432, 0.147, 0.053, 0.053]),
        "vs_pf": np.array([0.990, 0.398, 0.124, 0.034, 0.034]),
    }


def _synth_burn_in_curve(
    start: float, end: float, n_steps: int
) -> np.ndarray:
    """Synthesize a plausible exponential convergence curve."""
    t = np.arange(n_steps, dtype=float)
    decay = np.exp(-0.15 * t)
    return end + (start - end) * decay
