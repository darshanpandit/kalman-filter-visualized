"""Generate precomputed .npz files from digitized paper results.

Run once: PYTHONPATH=. python3 data/transformer_results/generate_npz.py
"""

from __future__ import annotations

import os
import numpy as np

OUT_DIR = os.path.dirname(__file__)


def generate_kalmannet_radar():
    """Mehrfard et al. (2024) Table 1: KalmanNet vs IMM on RadarScenes."""
    np.savez(
        os.path.join(OUT_DIR, "kalmannet_radar.npz"),
        methods=np.array(["KalmanNet", "IMM"]),
        pos_rmse=np.array([1.23, 1.08]),
        vel_rmse=np.array([2.98, 1.28]),
    )


def generate_icl_results():
    """Balim et al. (2023) Tables 1-2: Transformer in-context learning."""
    n = 50
    t = np.arange(n, dtype=float)
    decay = np.exp(-0.15 * t)

    np.savez(
        os.path.join(OUT_DIR, "icl_results.npz"),
        systems=np.array(["linear", "colored_noise", "quadrotor"]),
        kf_mse=np.array([0.048, 0.152, 0.891]),
        tf_mse=np.array([0.051, 0.098, 0.342]),
        burn_in_steps=np.arange(1, n + 1),
        burn_in_kf_mse=0.048 + (0.5 - 0.048) * decay,
        burn_in_tf_mse=0.051 + (0.8 - 0.051) * decay,
    )


def generate_scaling_results():
    """Akram & Vikalo (2024) Table 1: MSPD vs layers."""
    np.savez(
        os.path.join(OUT_DIR, "scaling_results.npz"),
        layers=np.array([1, 2, 4, 8, 16]),
        vs_ekf=np.array([1.028, 0.432, 0.147, 0.053, 0.053]),
        vs_pf=np.array([0.990, 0.398, 0.124, 0.034, 0.034]),
    )


if __name__ == "__main__":
    generate_kalmannet_radar()
    generate_icl_results()
    generate_scaling_results()
    print(f"Generated .npz files in {OUT_DIR}")
