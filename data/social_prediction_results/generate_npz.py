"""Generate precomputed .npz files for social prediction results.

Digitized from published ETH/UCY benchmark results.
Run once: PYTHONPATH=. python3 data/social_prediction_results/generate_npz.py
"""

from __future__ import annotations

import os
import numpy as np

OUT_DIR = os.path.dirname(__file__)


def generate_ethucyu_results():
    """Published trajectory prediction results on ETH/UCY (ADE/FDE, best-of-20)."""
    np.savez(
        os.path.join(OUT_DIR, "social_prediction.npz"),
        methods=np.array(["Linear", "S-LSTM", "S-GAN", "Trajectron++", "AgentFormer"]),
        years=np.array([0, 2016, 2018, 2020, 2021]),
        venues=np.array(["--", "CVPR", "CVPR", "ECCV", "ICCV"]),
        avg_ade=np.array([1.33, 0.72, 0.58, 0.43, 0.23]),
        avg_fde=np.array([2.94, 1.48, 1.18, 0.86, 0.39]),
    )


if __name__ == "__main__":
    generate_ethucyu_results()
    print(f"Generated .npz files in {OUT_DIR}")
