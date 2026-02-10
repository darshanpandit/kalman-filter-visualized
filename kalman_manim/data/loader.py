"""Load real-world pedestrian trajectory data (ETH Pedestrian Dataset).

Returns data in the same dict format as generators.py, so scenes can swap
one import line to switch between synthetic and real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DATASETS_DIR = Path(__file__).parent / "datasets"

# Frame interval in the ETH dataset (sampled every 10 frames at 25 FPS)
_ETH_FRAME_INTERVAL = 10
_ETH_FPS = 25
_ETH_DT = _ETH_FRAME_INTERVAL / _ETH_FPS  # 0.4 seconds between observations


def _load_eth_raw(sequence: str) -> np.ndarray:
    """Load raw ETH data file. Returns array of (frame, ped_id, x, y)."""
    path = _DATASETS_DIR / "eth" / f"{sequence}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"ETH dataset '{sequence}' not found at {path}. "
            f"Available: {[p.stem for p in (_DATASETS_DIR / 'eth').glob('*.txt')]}"
        )
    return np.loadtxt(path)


def list_available_trajectories(
    sequence: str = "hotel",
    min_steps: int = 40,
) -> list[dict]:
    """List pedestrian IDs with sufficiently long trajectories.

    Parameters
    ----------
    sequence : str
        Dataset sequence name: "hotel" or "eth".
    min_steps : int
        Minimum number of time steps (observations) required.

    Returns
    -------
    List of dicts with keys: pedestrian_id, n_steps, duration_s.
    """
    raw = _load_eth_raw(sequence)
    ped_ids = np.unique(raw[:, 1])
    results = []
    for pid in ped_ids:
        mask = raw[:, 1] == pid
        n_steps = mask.sum()
        if n_steps >= min_steps:
            frames = raw[mask, 0]
            duration = (frames[-1] - frames[0]) / _ETH_FPS
            results.append({
                "pedestrian_id": int(pid),
                "n_steps": int(n_steps),
                "duration_s": float(duration),
            })
    results.sort(key=lambda d: d["n_steps"], reverse=True)
    return results


def load_eth_trajectory(
    sequence: str = "hotel",
    pedestrian_id: int | None = None,
    measurement_noise_std: float = 0.5,
    max_steps: int | None = None,
    seed: int | None = None,
) -> dict:
    """Load a real pedestrian trajectory with synthetic measurement noise.

    Returns the same dict format as generators: {true_states, measurements, dt}.

    Parameters
    ----------
    sequence : str
        Dataset sequence: "hotel" or "eth".
    pedestrian_id : int or None
        Specific pedestrian to load. If None, picks the longest trajectory.
    measurement_noise_std : float
        Std dev of synthetic Gaussian noise added to positions to simulate
        GPS/LBS measurement error. Set to 0 for raw positions.
    max_steps : int or None
        Truncate trajectory to this many steps. None for full trajectory.
    seed : int or None
        Random seed for measurement noise reproducibility.

    Returns
    -------
    dict with keys:
        true_states  : np.ndarray (n_steps+1, 4) — [x, y, vx, vy]
        measurements : np.ndarray (n_steps, 2)   — noisy [x, y]
        dt           : float — time between observations
        metadata     : dict — {sequence, pedestrian_id, source}
    """
    raw = _load_eth_raw(sequence)

    # Select pedestrian
    if pedestrian_id is None:
        # Pick the longest trajectory
        ped_ids, counts = np.unique(raw[:, 1], return_counts=True)
        pedestrian_id = int(ped_ids[np.argmax(counts)])

    mask = raw[:, 1] == pedestrian_id
    if not mask.any():
        available = sorted(np.unique(raw[:, 1]).astype(int).tolist())
        raise ValueError(
            f"Pedestrian {pedestrian_id} not found in '{sequence}'. "
            f"Available IDs: {available[:20]}..."
        )

    ped_data = raw[mask]
    # Sort by frame
    ped_data = ped_data[ped_data[:, 0].argsort()]

    positions = ped_data[:, 2:4]  # (x, y)
    frames = ped_data[:, 0]

    # Compute dt from frame intervals
    dt = _ETH_DT

    if max_steps is not None:
        positions = positions[: max_steps + 1]
        frames = frames[: max_steps + 1]

    n_points = len(positions)
    n_steps = n_points - 1

    if n_steps < 2:
        raise ValueError(
            f"Pedestrian {pedestrian_id} in '{sequence}' has only "
            f"{n_points} points (need at least 3)."
        )

    # Build true_states: [x, y, vx, vy]
    # Velocities estimated via finite differences
    true_states = np.zeros((n_points, 4))
    true_states[:, :2] = positions

    for k in range(n_points - 1):
        # Use actual frame gap for velocity estimation
        frame_gap = frames[k + 1] - frames[k]
        actual_dt = frame_gap / _ETH_FPS
        if actual_dt > 0:
            true_states[k, 2:] = (positions[k + 1] - positions[k]) / actual_dt
    # Last velocity = same as second-to-last
    true_states[-1, 2:] = true_states[-2, 2:]

    # Generate noisy measurements (from positions[1:], matching generator convention)
    rng = np.random.default_rng(seed)
    measurements = positions[1:].copy()
    if measurement_noise_std > 0:
        measurements += rng.normal(0, measurement_noise_std, size=measurements.shape)

    return {
        "true_states": true_states,
        "measurements": measurements,
        "dt": dt,
        "metadata": {
            "sequence": sequence,
            "pedestrian_id": int(pedestrian_id),
            "source": "ETH Pedestrian Dataset (Pellegrini et al. 2009)",
        },
    }
