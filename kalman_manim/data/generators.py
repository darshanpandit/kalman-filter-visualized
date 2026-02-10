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


def generate_multi_target_scenario(
    n_steps: int = 60,
    dt: float = 0.5,
    n_targets_init: int = 3,
    birth_step: int = 20,
    death_step: int = 40,
    process_noise_std: float = 0.1,
    measurement_noise_std: float = 0.5,
    clutter_rate: float = 0.5,
    seed: int | None = None,
):
    """Generate a multi-target tracking scenario.

    Multiple targets move independently. One target is born at
    birth_step and one dies at death_step, creating a varying
    cardinality scenario for PHD filter demos.

    State per target: [x, y, vx, vy].

    Returns
    -------
    dict with keys:
        true_tracks : list of dict, each with 'states' array, 'start_step', 'end_step'
        measurement_sets : list of list of np.ndarray (variable per step)
        true_cardinality : np.ndarray (n_steps,)
        dt : float
    """
    rng = np.random.default_rng(seed)

    # Initialize targets
    tracks = []
    for i in range(n_targets_init):
        angle = 2 * np.pi * i / n_targets_init
        pos = np.array([np.cos(angle) * 3, np.sin(angle) * 3])
        vel = np.array([-np.sin(angle) * 0.5, np.cos(angle) * 0.5])
        tracks.append({
            "states": [np.concatenate([pos, vel])],
            "start_step": 0,
            "end_step": n_steps,
            "alive": True,
        })

    # Born target
    born_track = {
        "states": [],
        "start_step": birth_step,
        "end_step": n_steps,
        "alive": False,
    }
    tracks.append(born_track)

    measurement_sets = []
    true_cardinality = np.zeros(n_steps, dtype=int)

    for k in range(n_steps):
        # Birth event
        if k == birth_step:
            born_track["alive"] = True
            born_track["states"].append(np.array([5.0, 0.0, -0.3, 0.4]))

        # Death event
        if k == death_step and len(tracks) > 1:
            tracks[0]["alive"] = False
            tracks[0]["end_step"] = death_step

        # Propagate alive targets
        measurements = []
        n_alive = 0
        for track in tracks:
            if not track["alive"]:
                continue
            n_alive += 1

            state = track["states"][-1]
            pos_t, vel_t = state[:2], state[2:]

            accel_noise = rng.normal(0, process_noise_std, size=2)
            new_vel_t = vel_t + accel_noise * dt
            new_pos_t = pos_t + vel_t * dt

            new_state = np.concatenate([new_pos_t, new_vel_t])
            track["states"].append(new_state)

            # Measurement (detection probability ~0.95)
            if rng.random() < 0.95:
                meas = new_pos_t + rng.normal(0, measurement_noise_std, size=2)
                measurements.append(meas)

        # Clutter
        n_clutter = rng.poisson(clutter_rate)
        for _ in range(n_clutter):
            measurements.append(rng.uniform(-8, 8, size=2))

        measurement_sets.append(measurements)
        true_cardinality[k] = n_alive

    # Convert track states to arrays
    for track in tracks:
        track["states"] = np.array(track["states"])
        del track["alive"]

    return {
        "true_tracks": tracks,
        "measurement_sets": measurement_sets,
        "true_cardinality": true_cardinality,
        "dt": dt,
    }


def generate_mode_switching_trajectory(
    n_steps: int = 80,
    dt: float = 0.5,
    speed: float = 0.8,
    process_noise_std: float = 0.05,
    measurement_noise_std: float = 0.5,
    seed: int | None = None,
):
    """Generate a trajectory that switches between constant velocity and
    coordinated turn models.

    Designed for IMM filter demos: the target moves straight, then turns,
    then goes straight again. A single-model KF will fail during turns.

    State: [x, y, vx, vy]. Mode changes at fixed intervals.

    Returns same dict format as other generators, plus:
        true_modes : np.ndarray (n_steps,) -- 0=CV, 1=CT
    """
    rng = np.random.default_rng(seed)

    true_states = np.zeros((n_steps + 1, 4))
    true_states[0] = [0.0, 0.0, speed, 0.0]
    measurements = np.zeros((n_steps, 2))
    true_modes = np.zeros(n_steps, dtype=int)

    # Mode schedule: CV for 20, CT for 20, CV for 20, CT for 20
    turn_rate = 0.15
    for k in range(n_steps):
        pos_k = true_states[k, :2]
        vel_k = true_states[k, 2:]

        segment = (k // 20) % 2
        true_modes[k] = segment

        if segment == 1:
            # Coordinated turn
            omega = turn_rate
            cos_w = np.cos(omega * dt)
            sin_w = np.sin(omega * dt)
            new_vx_k = vel_k[0] * cos_w - vel_k[1] * sin_w
            new_vy_k = vel_k[0] * sin_w + vel_k[1] * cos_w
            new_vel_k = np.array([new_vx_k, new_vy_k])
        else:
            new_vel_k = vel_k.copy()

        accel_noise = rng.normal(0, process_noise_std, size=2)
        new_vel_k = new_vel_k + accel_noise * dt
        new_pos_k = pos_k + vel_k * dt

        true_states[k + 1, :2] = new_pos_k
        true_states[k + 1, 2:] = new_vel_k

        measurements[k] = new_pos_k + rng.normal(0, measurement_noise_std, size=2)

    return {
        "true_states": true_states,
        "measurements": measurements,
        "true_modes": true_modes,
        "dt": dt,
    }


def generate_lorenz_trajectory(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
    n_steps: int = 3000,
    initial_state: np.ndarray | None = None,
    seed: int | None = None,
):
    """Generate a Lorenz attractor trajectory via RK4 integration.

    Parameters
    ----------
    sigma, rho, beta : float
        Lorenz system parameters. Defaults produce chaotic behavior.
    dt : float
        Integration time step.
    n_steps : int
        Number of integration steps.
    initial_state : np.ndarray (3,) or None
        Initial [x, y, z]. Default [1, 1, 1].

    Returns
    -------
    dict with keys:
        states : np.ndarray (n_steps+1, 3) -- [x, y, z]
        dt : float
    """
    if initial_state is None:
        initial_state = np.array([1.0, 1.0, 1.0])

    def lorenz(state):
        x, y, z = state
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ])

    states = np.zeros((n_steps + 1, 3))
    states[0] = initial_state

    for i in range(n_steps):
        s = states[i]
        k1 = dt * lorenz(s)
        k2 = dt * lorenz(s + 0.5 * k1)
        k3 = dt * lorenz(s + 0.5 * k2)
        k4 = dt * lorenz(s + k3)
        states[i + 1] = s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return {"states": states, "dt": dt}


def generate_pendulum_trajectory(
    length: float = 1.0,
    gravity: float = 9.81,
    theta0: float = 1.0,
    omega0: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 1000,
):
    """Generate a simple pendulum trajectory via RK4 integration.

    State: [theta, omega] where theta = angle, omega = angular velocity.
    Dynamics: d(theta)/dt = omega, d(omega)/dt = -(g/L) sin(theta).

    Also computes energy: E = 0.5 * L^2 * omega^2 + g * L * (1 - cos(theta)).

    Returns
    -------
    dict with keys:
        states : np.ndarray (n_steps+1, 2) -- [theta, omega]
        energy : np.ndarray (n_steps+1,)
        dt : float
    """
    def pendulum(state):
        theta, omega = state
        return np.array([omega, -(gravity / length) * np.sin(theta)])

    states = np.zeros((n_steps + 1, 2))
    states[0] = [theta0, omega0]

    for i in range(n_steps):
        s = states[i]
        k1 = dt * pendulum(s)
        k2 = dt * pendulum(s + 0.5 * k1)
        k3 = dt * pendulum(s + 0.5 * k2)
        k4 = dt * pendulum(s + k3)
        states[i + 1] = s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Energy
    energy = (0.5 * length ** 2 * states[:, 1] ** 2
              + gravity * length * (1 - np.cos(states[:, 0])))

    return {"states": states, "energy": energy, "dt": dt}
