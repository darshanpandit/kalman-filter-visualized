"""Filter factory functions for benchmark comparisons.

All filters use the same constant-velocity (CV) model with 4-state [x, y, vx, vy].
This measures robustness to model mismatch when the true dynamics have curvature.
"""

from __future__ import annotations

import numpy as np

from filters.kalman import KalmanFilter
from filters.ekf import ExtendedKalmanFilter
from filters.ukf import UnscentedKalmanFilter
from filters.particle import ParticleFilter


# ── Shared model functions ─────────────────────────────────────────────────


def make_cv_transition(dt: float):
    """Constant-velocity transition: f(x, u) -> x_next."""
    def f(x, u):
        return np.array([
            x[0] + x[2] * dt,
            x[1] + x[3] * dt,
            x[2],
            x[3],
        ])
    return f


def make_cv_jacobian(dt: float):
    """Jacobian of the CV transition."""
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    def F_jac(x, u):
        return F
    return F_jac


def _h(x):
    """Position-only measurement function."""
    return x[:2]


def _H_jac(x):
    """Jacobian of h."""
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])


def _pf_transition_factory(dt: float):
    """PF transition: f(x, u, noise) -> x_next."""
    def f(x, u, noise):
        return np.array([
            x[0] + x[2] * dt + noise[0],
            x[1] + x[3] * dt + noise[1],
            x[2] + noise[2],
            x[3] + noise[3],
        ])
    return f


# ── Default noise parameters ──────────────────────────────────────────────


def default_Q(dt: float) -> np.ndarray:
    """Default process noise covariance."""
    return 0.08 * np.eye(4)


def default_R() -> np.ndarray:
    """Default measurement noise covariance."""
    return 0.25 * np.eye(2)


def default_P0() -> np.ndarray:
    """Default initial covariance."""
    return np.eye(4)


# ── Filter factories ──────────────────────────────────────────────────────


def make_kf(
    dt: float,
    x0: np.ndarray,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> KalmanFilter:
    """Create a linear Kalman Filter with CV model."""
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    return KalmanFilter(
        F=F, H=H,
        Q=Q if Q is not None else default_Q(dt),
        R=R if R is not None else default_R(),
        x0=x0.copy(),
        P0=P0 if P0 is not None else default_P0(),
    )


def make_ekf(
    dt: float,
    x0: np.ndarray,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> ExtendedKalmanFilter:
    """Create an EKF with CV model."""
    return ExtendedKalmanFilter(
        f=make_cv_transition(dt),
        h=_h,
        F_jacobian=make_cv_jacobian(dt),
        H_jacobian=_H_jac,
        Q=Q if Q is not None else default_Q(dt),
        R=R if R is not None else default_R(),
        x0=x0.copy(),
        P0=P0 if P0 is not None else default_P0(),
    )


def make_ukf(
    dt: float,
    x0: np.ndarray,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> UnscentedKalmanFilter:
    """Create a UKF with CV model."""
    return UnscentedKalmanFilter(
        f=make_cv_transition(dt),
        h=_h,
        Q=Q if Q is not None else default_Q(dt),
        R=R if R is not None else default_R(),
        x0=x0.copy(),
        P0=P0 if P0 is not None else default_P0(),
    )


def make_pf(
    dt: float,
    x0: np.ndarray,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    P0: np.ndarray | None = None,
    n_particles: int = 300,
    seed: int = 42,
) -> ParticleFilter:
    """Create a Particle Filter with CV model."""
    Q_pf = Q if Q is not None else np.diag([0.02, 0.02, 0.04, 0.04])
    return ParticleFilter(
        f=_pf_transition_factory(dt),
        h=_h,
        Q=Q_pf,
        R=R if R is not None else default_R(),
        n_particles=n_particles,
        x0=x0.copy(),
        P0=P0 if P0 is not None else default_P0(),
        seed=seed,
    )


# ── Convenience ───────────────────────────────────────────────────────────

FILTER_NAMES = ["KF", "EKF", "UKF", "PF"]


def make_all_filters(
    dt: float,
    x0: np.ndarray,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    P0: np.ndarray | None = None,
    pf_particles: int = 300,
    pf_seed: int = 42,
) -> dict:
    """Create all 4 filters with matching parameters.

    Returns dict: {"KF": kf, "EKF": ekf, "UKF": ukf, "PF": pf}.
    """
    return {
        "KF": make_kf(dt, x0, Q, R, P0),
        "EKF": make_ekf(dt, x0, Q, R, P0),
        "UKF": make_ukf(dt, x0, Q, R, P0),
        "PF": make_pf(dt, x0, Q, R, P0, pf_particles, pf_seed),
    }
