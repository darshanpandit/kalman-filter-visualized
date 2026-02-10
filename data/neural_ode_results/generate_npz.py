"""Generate precomputed .npz files for Neural ODE / HNN results.

Run once: PYTHONPATH=. python3 data/neural_ode_results/generate_npz.py
"""

from __future__ import annotations

import os
import numpy as np

OUT_DIR = os.path.dirname(__file__)


def generate_pets_results():
    """PETS sample efficiency (Chua et al. 2018)."""
    np.savez(
        os.path.join(OUT_DIR, "pets_efficiency.npz"),
        methods=np.array(["PPO", "SAC", "PETS"]),
        types=np.array(["Model-free", "Model-free", "Model-based"]),
        halfcheetah_samples_1k=np.array([3000, 600, 24]),
    )


def generate_hnn_results():
    """HNN energy conservation (Greydanus 2019)."""
    np.savez(
        os.path.join(OUT_DIR, "hnn_energy.npz"),
        methods=np.array(["Baseline NN", "HNN"]),
        energy_drift_1000_steps=np.array([3.2, 0.03]),
    )


if __name__ == "__main__":
    generate_pets_results()
    generate_hnn_results()
    print(f"Generated .npz files in {OUT_DIR}")
