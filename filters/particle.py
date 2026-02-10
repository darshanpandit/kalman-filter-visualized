"""Particle Filter (Sequential Monte Carlo) — pure numpy implementation."""

from __future__ import annotations

import numpy as np


class ParticleFilter:
    """Discrete-time Particle Filter using Sequential Importance Resampling (SIR).

    Represents the posterior as a set of weighted particles.
    Can handle non-Gaussian, multimodal distributions.

    Parameters
    ----------
    f : callable(x, u, noise) -> np.ndarray
        State transition function. Takes a single particle, control input,
        and process noise sample. Returns the new particle state.
    h : callable(x) -> np.ndarray
        Measurement function. Maps state to expected measurement.
    Q : np.ndarray
        Process noise covariance (used for sampling noise).
    R : np.ndarray
        Measurement noise covariance (used for weighting).
    n_particles : int
        Number of particles.
    x0 : np.ndarray
        Initial state estimate (center of initial particle cloud).
    P0 : np.ndarray
        Initial covariance (spread of initial particle cloud).
    resample_threshold : float
        Fraction of N_eff/N below which resampling occurs (0 to 1).
    seed : int or None
        Random seed.
    """

    def __init__(self, f, h, Q, R, n_particles: int = 500,
                 x0=None, P0=None, resample_threshold: float = 0.5,
                 seed: int | None = None):
        self.f = f
        self.h = h
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.n_particles = n_particles
        self.n = self.Q.shape[0]
        self.m = self.R.shape[0]
        self.resample_threshold = resample_threshold
        self.rng = np.random.default_rng(seed)

        # Initialize particles
        x0 = np.zeros(self.n) if x0 is None else np.array(x0, dtype=float)
        P0 = np.eye(self.n) if P0 is None else np.array(P0, dtype=float)

        self.particles = self.rng.multivariate_normal(x0, P0, size=n_particles)
        self.weights = np.ones(n_particles) / n_particles

    @property
    def x(self):
        """Weighted mean estimate."""
        return np.average(self.particles, weights=self.weights, axis=0)

    @property
    def P(self):
        """Weighted covariance estimate."""
        mean = self.x
        diff = self.particles - mean
        return np.average(
            np.array([np.outer(d, d) for d in diff]),
            weights=self.weights, axis=0,
        )

    def predict(self, u=None):
        """Prediction step: propagate each particle through f with noise."""
        for i in range(self.n_particles):
            noise = self.rng.multivariate_normal(np.zeros(self.n), self.Q)
            self.particles[i] = self.f(self.particles[i], u, noise)
        return self.x.copy(), self.P.copy()

    def update(self, z):
        """Measurement update: weight particles by likelihood, then resample."""
        z = np.array(z, dtype=float)
        R_inv = np.linalg.inv(self.R)
        R_det = np.linalg.det(self.R)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** self.m * R_det)

        # Compute likelihood for each particle
        for i in range(self.n_particles):
            z_pred = self.h(self.particles[i])
            diff = z - z_pred
            log_likelihood = -0.5 * diff @ R_inv @ diff
            self.weights[i] *= norm_const * np.exp(log_likelihood)

        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum < 1e-300:
            # All weights collapsed — reinitialize uniformly
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            self.weights /= weight_sum

        # Innovation (for compatibility with KF/EKF/UKF interface)
        y = z - self.h(self.x)

        # Resample if effective particle count is too low
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.resample_threshold * self.n_particles:
            self._systematic_resample()

        return self.x.copy(), self.P.copy(), self.weights.copy(), y.copy()

    def _systematic_resample(self):
        """Systematic resampling (low-variance resampling)."""
        N = self.n_particles
        positions = (self.rng.random() + np.arange(N)) / N
        cumsum = np.cumsum(self.weights)

        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, N - 1)

        self.particles = self.particles[indices].copy()
        self.weights = np.ones(N) / N

    def run(self, measurements, controls=None):
        """Run the particle filter over a sequence of measurements."""
        results = {
            "x_predictions": [],
            "P_predictions": [],
            "x_estimates": [],
            "P_estimates": [],
            "weights_history": [],
            "particles_history": [],
            "innovations": [],
        }
        for k, z in enumerate(measurements):
            u = controls[k] if controls is not None else None
            x_pred, P_pred = self.predict(u)
            results["x_predictions"].append(x_pred)
            results["P_predictions"].append(P_pred)

            x_est, P_est, w, innov = self.update(z)
            results["x_estimates"].append(x_est)
            results["P_estimates"].append(P_est)
            results["weights_history"].append(w)
            results["particles_history"].append(self.particles.copy())
            results["innovations"].append(innov)

        return results
