"""Parameter sweep: vary turn_rate and measure filter performance."""

from __future__ import annotations

import numpy as np

from kalman_manim.data.generators import generate_nonlinear_trajectory
from benchmarks.configs import make_all_filters, FILTER_NAMES
from benchmarks.metrics import position_rmse


def sweep_turn_rate(
    turn_rates: np.ndarray | None = None,
    n_trials_per_rate: int = 50,
    n_steps: int = 60,
    dt: float = 0.5,
    measurement_noise_std: float = 0.5,
    base_seed: int = 42,
    filter_names: list[str] | None = None,
    pf_particles: int = 300,
) -> dict:
    """Sweep turn_rate from linear to highly nonlinear and collect RMSE.

    Parameters
    ----------
    turn_rates : array-like or None
        Turn rate values to sweep. Default: 25 values from 0 to 0.5.
    n_trials_per_rate : int
        Number of random trials at each turn rate.
    n_steps : int
        Steps per trajectory.
    dt : float
        Time step.
    measurement_noise_std : float
        Measurement noise.
    base_seed : int
        Base seed for deterministic reproducibility.
    filter_names : list or None
        Filters to evaluate. None = all 4.
    pf_particles : int
        Particles for PF.

    Returns
    -------
    dict with keys:
        turn_rates   : np.ndarray (R,)
        filter_names : list of str
        rmse         : np.ndarray (R, F, T) — RMSE per rate, filter, trial
        mean_rmse    : np.ndarray (R, F)
        std_rmse     : np.ndarray (R, F)
    """
    if turn_rates is None:
        turn_rates = np.linspace(0, 0.5, 25)
    turn_rates = np.asarray(turn_rates)

    if filter_names is None:
        filter_names = FILTER_NAMES

    n_rates = len(turn_rates)
    n_filters = len(filter_names)

    rmse = np.zeros((n_rates, n_filters, n_trials_per_rate))

    for ri, rate in enumerate(turn_rates):
        for ti in range(n_trials_per_rate):
            seed = base_seed + ri * n_trials_per_rate + ti

            # Generate trajectory with this turn rate
            data = generate_nonlinear_trajectory(
                n_steps=n_steps,
                dt=dt,
                turn_rate=rate,
                measurement_noise_std=measurement_noise_std,
                seed=seed,
            )

            true_states = data["true_states"]
            measurements = data["measurements"]
            x0 = true_states[0, :4]

            filters = make_all_filters(
                dt=dt, x0=x0,
                pf_particles=pf_particles,
                pf_seed=seed,
            )

            for fi, fname in enumerate(filter_names):
                filt = filters[fname]
                res = filt.run(measurements)
                # true_states may have 5 cols (CT model), truncate to 4
                ts = true_states[:, :4] if true_states.shape[1] > 4 else true_states
                rmse[ri, fi, ti] = position_rmse(res["x_estimates"], ts)

    return {
        "turn_rates": turn_rates,
        "filter_names": filter_names,
        "rmse": rmse,
        "mean_rmse": rmse.mean(axis=2),
        "std_rmse": rmse.std(axis=2),
    }
