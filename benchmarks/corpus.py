"""Generate mixed trajectory corpus for comprehensive benchmarking."""

from __future__ import annotations

import numpy as np

from kalman_manim.data.generators import (
    generate_linear_trajectory,
    generate_pedestrian_trajectory,
    generate_nonlinear_trajectory,
    generate_sharp_turn_trajectory,
)
from kalman_manim.data.loader import load_trajectory


def generate_synthetic_corpus(
    n_per_regime: int = 30,
    base_seed: int = 100,
) -> list[dict]:
    """Generate synthetic trajectories across 4 nonlinearity regimes.

    Regimes:
        linear       — straight-line constant velocity
        pedestrian   — gentle random turns
        nonlinear    — coordinated turn (turn_rate ~0.15)
        sharp_turn   — abrupt 90-degree turns

    Each trajectory dict has extra keys: regime, seed.

    Returns
    -------
    list of trajectory dicts (4 × n_per_regime entries).
    """
    corpus = []
    seed = base_seed

    # Linear regime
    for i in range(n_per_regime):
        data = generate_linear_trajectory(
            n_steps=60, dt=0.5, measurement_noise_std=0.5, seed=seed,
        )
        data["regime"] = "linear"
        data["seed"] = seed
        corpus.append(data)
        seed += 1

    # Pedestrian regime (gentle turns)
    for i in range(n_per_regime):
        data = generate_pedestrian_trajectory(
            n_steps=60, dt=0.5, measurement_noise_std=0.5,
            turn_probability=0.1, turn_angle_std=np.pi / 6,
            process_noise_std=0.15, seed=seed,
        )
        data["regime"] = "pedestrian"
        data["seed"] = seed
        corpus.append(data)
        seed += 1

    # Nonlinear regime (coordinated turn)
    for i in range(n_per_regime):
        # Vary turn rate across trials
        turn_rate = 0.1 + 0.2 * (i / max(n_per_regime - 1, 1))
        data = generate_nonlinear_trajectory(
            n_steps=60, dt=0.5, turn_rate=turn_rate,
            measurement_noise_std=0.5, seed=seed,
        )
        data["regime"] = "nonlinear"
        data["seed"] = seed
        data["turn_rate"] = turn_rate
        corpus.append(data)
        seed += 1

    # Sharp turn regime
    for i in range(n_per_regime):
        data = generate_sharp_turn_trajectory(
            n_steps=60, dt=0.5, measurement_noise_std=0.5, seed=seed,
        )
        data["regime"] = "sharp_turn"
        data["seed"] = seed
        corpus.append(data)
        seed += 1

    return corpus


def load_real_corpus(
    min_steps: int = 20,
    noise_std: float = 0.5,
    base_seed: int = 200,
) -> list[dict]:
    """Load all real trajectories with sufficient length.

    Loads from ETH (eth, hotel) and UCY (univ, zara1, zara2) datasets.

    Returns
    -------
    list of trajectory dicts with extra keys: regime, dataset, pedestrian_id.
    """
    datasets = ["eth", "hotel", "univ", "zara1", "zara2"]
    corpus = []
    seed = base_seed

    for dataset in datasets:
        try:
            from kalman_manim.data.loader import list_available_trajectories
            available = list_available_trajectories(
                sequence=dataset, min_steps=min_steps,
            )
        except (FileNotFoundError, Exception):
            continue

        for info in available:
            try:
                data = load_trajectory(
                    dataset=dataset,
                    pedestrian_id=info["pedestrian_id"],
                    measurement_noise_std=noise_std,
                    seed=seed,
                )
                data["regime"] = "real"
                data["dataset"] = dataset
                data["pedestrian_id"] = info["pedestrian_id"]
                corpus.append(data)
                seed += 1
            except (ValueError, Exception):
                continue

    return corpus
