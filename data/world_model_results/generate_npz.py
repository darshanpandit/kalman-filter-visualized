"""Generate precomputed .npz files for world model results.

Run once: PYTHONPATH=. python3 data/world_model_results/generate_npz.py
"""

from __future__ import annotations

import os
import numpy as np

OUT_DIR = os.path.dirname(__file__)


def generate_dmcontrol_results():
    """DMControl-100k scores from Hafner et al. papers."""
    np.savez(
        os.path.join(OUT_DIR, "dmcontrol_scores.npz"),
        methods=np.array(["D4PG", "SAC", "PlaNet", "DreamerV1", "DreamerV3"]),
        types=np.array(["Model-free", "Model-free", "Model-based",
                         "Model-based", "Model-based"]),
        avg_scores=np.array([274, 437, 650, 823, 853]),
    )


def generate_muzero_results():
    """MuZero results: matches AlphaZero without knowing game rules."""
    np.savez(
        os.path.join(OUT_DIR, "muzero_scores.npz"),
        games=np.array(["Go", "Chess", "Shogi", "Atari (57 games)"]),
        alphazero_elo=np.array([5185, 4722, 4509, 0]),
        muzero_elo=np.array([5205, 4738, 4520, 0]),
        muzero_atari_median=np.array([0, 0, 0, 1052]),
    )


if __name__ == "__main__":
    generate_dmcontrol_results()
    generate_muzero_results()
    print(f"Generated .npz files in {OUT_DIR}")
