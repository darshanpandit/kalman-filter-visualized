"""Pedestrian trajectory generators for Kalman filter demonstrations."""

from __future__ import annotations

import numpy as np


def generate_pedestrian_trajectory(
    n_steps: int = 100,
    dt: float = 1.0,
    initial_pos: np.ndarray | None = None,
    initial_vel: np.ndarray | None = None,
    process_noise_std: float = 0.3,
    measurement_noise_std: float = 1.0,
    turn_probability: float = 0.05,
    turn_angle_std: float = np.pi / 4,
    speed: float = 1.0,
    seed: int | None = None,
):
    """Simulate a 2D pedestrian trajectory with noisy measurements.

    The pedestrian walks with approximately constant velocity, occasionally
    making random turns. Measurements are noisy position observations
    (simulating GPS/LBS pings).

    State vector: [x, y, vx, vy] (position + velocity in 2D)

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    dt : float
        Time step duration.
    initial_pos : np.ndarray
        Starting [x, y] position. Default [0, 0].
    initial_vel : np.ndarray
        Starting [vx, vy] velocity. Default [speed, 0].
    process_noise_std : float
        Std dev of acceleration noise (applied to velocity).
    measurement_noise_std : float
        Std dev of position measurement noise.
    turn_probability : float
        Probability of a random heading change at each step.
    turn_angle_std : float
        Std dev of heading change angle (radians) when a turn occurs.
    speed : float
        Base walking speed (used for initial velocity if not provided).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        true_states    : np.ndarray (n_steps+1, 4) — [x, y, vx, vy] at each step
        measurements   : np.ndarray (n_steps, 2) — noisy [x, y] observations
        dt             : float
    """
    rng = np.random.default_rng(seed)

    if initial_pos is None:
        initial_pos = np.array([0.0, 0.0])
    if initial_vel is None:
        initial_vel = np.array([speed, 0.0])

    true_states = np.zeros((n_steps + 1, 4))
    true_states[0, :2] = initial_pos
    true_states[0, 2:] = initial_vel

    measurements = np.zeros((n_steps, 2))

    for k in range(n_steps):
        pos = true_states[k, :2]
        vel = true_states[k, 2:]

        # Random turn
        if rng.random() < turn_probability:
            angle = rng.normal(0, turn_angle_std)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            vel = rot @ vel

        # Process noise (acceleration perturbation)
        accel_noise = rng.normal(0, process_noise_std, size=2)
        new_vel = vel + accel_noise * dt
        new_pos = pos + vel * dt + 0.5 * accel_noise * dt**2

        true_states[k + 1, :2] = new_pos
        true_states[k + 1, 2:] = new_vel

        # Noisy measurement of position
        meas_noise = rng.normal(0, measurement_noise_std, size=2)
        measurements[k] = new_pos + meas_noise

    return {
        "true_states": true_states,
        "measurements": measurements,
        "dt": dt,
    }


def generate_linear_trajectory(
    n_steps: int = 50,
    dt: float = 1.0,
    velocity: np.ndarray | None = None,
    process_noise_std: float = 0.1,
    measurement_noise_std: float = 0.5,
    seed: int | None = None,
):
    """Simpler trajectory: constant velocity with noise (no turns).

    Useful for early scenes demonstrating basic KF behavior.
    """
    rng = np.random.default_rng(seed)

    if velocity is None:
        velocity = np.array([0.5, 0.3])

    true_states = np.zeros((n_steps + 1, 4))
    true_states[0, 2:] = velocity

    measurements = np.zeros((n_steps, 2))

    for k in range(n_steps):
        pos = true_states[k, :2]
        vel = true_states[k, 2:]
        accel_noise = rng.normal(0, process_noise_std, size=2)

        new_pos = pos + vel * dt
        new_vel = vel + accel_noise * dt

        true_states[k + 1, :2] = new_pos
        true_states[k + 1, 2:] = new_vel

        measurements[k] = new_pos + rng.normal(0, measurement_noise_std, size=2)

    return {
        "true_states": true_states,
        "measurements": measurements,
        "dt": dt,
    }


def generate_nonlinear_trajectory(
    n_steps: int = 60,
    dt: float = 0.5,
    turn_rate: float = 0.15,
    speed: float = 1.0,
    process_noise_std: float = 0.1,
    measurement_noise_std: float = 0.5,
    seed: int | None = None,
):
    """Generate a trajectory with coordinated turns (nonlinear dynamics).

    State: [x, y, vx, vy, omega] where omega is turn rate.
    The pedestrian follows a curved path — suitable for EKF/UKF demos
    where a linear KF would fail.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    dt : float
        Time step duration.
    turn_rate : float
        Base turn rate (rad/s). Positive = left turn.
    speed : float
        Walking speed.
    process_noise_std : float
        Std dev of process noise on acceleration and turn rate.
    measurement_noise_std : float
        Std dev of position measurement noise.
    seed : int or None
        Random seed.

    Returns
    -------
    dict with keys:
        true_states  : np.ndarray (n_steps+1, 5) — [x, y, vx, vy, omega]
        measurements : np.ndarray (n_steps, 2) — noisy [x, y]
        dt           : float
    """
    rng = np.random.default_rng(seed)

    true_states = np.zeros((n_steps + 1, 5))
    true_states[0] = [0.0, 0.0, speed, 0.0, turn_rate]

    measurements = np.zeros((n_steps, 2))

    for k in range(n_steps):
        x, y, vx, vy, omega = true_states[k]

        # Coordinated turn model (exact integration)
        if abs(omega) > 1e-6:
            sin_wdt = np.sin(omega * dt)
            cos_wdt = np.cos(omega * dt)
            new_x = x + (vx * sin_wdt - vy * (1 - cos_wdt)) / omega
            new_y = y + (vx * (1 - cos_wdt) + vy * sin_wdt) / omega
            new_vx = vx * cos_wdt - vy * sin_wdt
            new_vy = vx * sin_wdt + vy * cos_wdt
        else:
            new_x = x + vx * dt
            new_y = y + vy * dt
            new_vx = vx
            new_vy = vy

        # Process noise
        new_vx += rng.normal(0, process_noise_std) * dt
        new_vy += rng.normal(0, process_noise_std) * dt
        new_omega = omega + rng.normal(0, process_noise_std * 0.5) * dt

        true_states[k + 1] = [new_x, new_y, new_vx, new_vy, new_omega]

        # Noisy measurement
        measurements[k] = [new_x, new_y] + rng.normal(0, measurement_noise_std, size=2)

    return {
        "true_states": true_states,
        "measurements": measurements,
        "dt": dt,
    }


def generate_sharp_turn_trajectory(
    n_steps: int = 60,
    dt: float = 0.5,
    speed: float = 0.8,
    process_noise_std: float = 0.05,
    measurement_noise_std: float = 0.5,
    seed: int | None = None,
):
    """Generate a trajectory with abrupt 90-degree turns.

    Curated for EKF failure demos: the linear KF will visibly fail
    at each sharp turn, while the EKF (with proper nonlinear model)
    can adapt.

    State: [x, y, vx, vy]. Turns happen at fixed intervals with
    exactly 90-degree heading changes.

    Returns same dict format as other generators.
    """
    rng = np.random.default_rng(seed)

    true_states = np.zeros((n_steps + 1, 4))
    true_states[0] = [0.0, 0.0, speed, 0.0]

    measurements = np.zeros((n_steps, 2))

    # Turn every ~15 steps, alternating left/right
    turn_interval = 15
    turn_direction = 1  # +1 = left (CCW), -1 = right (CW)

    for k in range(n_steps):
        pos = true_states[k, :2]
        vel = true_states[k, 2:]

        # Sharp 90-degree turn at intervals
        if k > 0 and k % turn_interval == 0:
            angle = turn_direction * np.pi / 2
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            vel = rot @ vel
            turn_direction *= -1  # alternate direction

        # Process noise
        accel_noise = rng.normal(0, process_noise_std, size=2)
        new_vel = vel + accel_noise * dt
        new_pos = pos + vel * dt

        true_states[k + 1, :2] = new_pos
        true_states[k + 1, 2:] = new_vel

        # Noisy measurement
        measurements[k] = new_pos + rng.normal(0, measurement_noise_std, size=2)

    return {
        "true_states": true_states,
        "measurements": measurements,
        "dt": dt,
    }


def generate_multimodal_scenario(
    n_steps: int = 50,
    dt: float = 0.5,
    speed: float = 0.6,
    process_noise_std: float = 0.05,
    measurement_noise_std: float = 0.5,
    fork_step: int = 25,
    seed: int | None = None,
):
    """Generate a corridor-fork scenario for particle filter demos.

    The pedestrian walks straight, then at fork_step takes a sharp turn
    into one of two corridors. This creates a scenario where a Gaussian
    filter would place its estimate between the two corridors (wrong),
    while particles can split and follow the correct branch.

    State: [x, y, vx, vy].

    Returns same dict format as other generators.
    """
    rng = np.random.default_rng(seed)

    true_states = np.zeros((n_steps + 1, 4))
    true_states[0] = [0.0, 0.0, speed, 0.0]

    measurements = np.zeros((n_steps, 2))

    for k in range(n_steps):
        pos = true_states[k, :2]
        vel = true_states[k, 2:]

        # At fork point, turn sharply
        if k == fork_step:
            # Turn 60 degrees (choose direction based on seed parity)
            angle = np.pi / 3 if rng.random() > 0.5 else -np.pi / 3
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            vel = rot @ vel

        # Gentle process noise
        accel_noise = rng.normal(0, process_noise_std, size=2)
        new_vel = vel + accel_noise * dt
        new_pos = pos + vel * dt

        true_states[k + 1, :2] = new_pos
        true_states[k + 1, 2:] = new_vel

        # Noisy measurement
        measurements[k] = new_pos + rng.normal(0, measurement_noise_std, size=2)

    return {
        "true_states": true_states,
        "measurements": measurements,
        "dt": dt,
    }
