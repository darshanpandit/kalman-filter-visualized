"""Unscented Kalman Filter — pure numpy implementation."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky


class UnscentedKalmanFilter:
    """Discrete-time Unscented Kalman Filter.

    Uses sigma points to propagate the mean and covariance through
    nonlinear functions without requiring Jacobians.

    Nonlinear state model:
        x_k = f(x_{k-1}, u_k) + w_k
    Nonlinear measurement model:
        z_k = h(x_k) + v_k

    Parameters
    ----------
    f : callable(x, u) -> np.ndarray
        State transition function.
    h : callable(x) -> np.ndarray
        Measurement function.
    Q : np.ndarray
        Process noise covariance (n x n).
    R : np.ndarray
        Measurement noise covariance (m x m).
    x0 : np.ndarray
        Initial state estimate.
    P0 : np.ndarray
        Initial covariance estimate.
    alpha : float
        Spread of sigma points around mean (typically 1e-3 to 1).
    beta : float
        Prior knowledge about distribution (2.0 is optimal for Gaussian).
    kappa : float
        Secondary scaling parameter (typically 0 or 3 - n).
    """

    def __init__(self, f, h, Q, R, x0=None, P0=None,
                 alpha: float = 1e-1, beta: float = 2.0, kappa: float = 0.0):
        self.f = f
        self.h = h
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.n = self.Q.shape[0]
        self.m = self.R.shape[0]

        self.x = np.zeros(self.n) if x0 is None else np.array(x0, dtype=float)
        self.P = np.eye(self.n) if P0 is None else np.array(P0, dtype=float)

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Compute weights
        self._compute_weights()

    def _compute_weights(self):
        """Compute Van der Merwe sigma point weights."""
        n = self.n
        lam = self.alpha**2 * (n + self.kappa) - n

        self.n_sigma = 2 * n + 1
        self.lam = lam

        # Mean weights
        self.Wm = np.full(self.n_sigma, 1.0 / (2 * (n + lam)))
        self.Wm[0] = lam / (n + lam)

        # Covariance weights
        self.Wc = np.full(self.n_sigma, 1.0 / (2 * (n + lam)))
        self.Wc[0] = lam / (n + lam) + (1 - self.alpha**2 + self.beta)

    def _generate_sigma_points(self, x, P):
        """Generate sigma points around mean x with covariance P.

        Returns (2n+1, n) array of sigma points.
        """
        n = self.n
        sigma_points = np.zeros((self.n_sigma, n))
        sigma_points[0] = x

        # Square root of (n + lambda) * P
        try:
            S = cholesky((n + self.lam) * P, lower=True)
        except np.linalg.LinAlgError:
            # Fallback: add jitter for numerical stability
            S = cholesky((n + self.lam) * P + 1e-6 * np.eye(n), lower=True)

        for i in range(n):
            sigma_points[i + 1] = x + S[:, i]
            sigma_points[n + i + 1] = x - S[:, i]

        return sigma_points

    def predict(self, u=None):
        """Prediction step using sigma points through f."""
        # Generate sigma points
        sigmas = self._generate_sigma_points(self.x, self.P)

        # Transform sigma points through f
        sigmas_pred = np.array([self.f(s, u) for s in sigmas])

        # Recover mean
        self.x = np.sum(self.Wm[:, None] * sigmas_pred, axis=0)

        # Recover covariance
        self.P = np.zeros((self.n, self.n))
        for i in range(self.n_sigma):
            diff = sigmas_pred[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)
        self.P += self.Q

        self._sigmas_pred = sigmas_pred  # save for update
        return self.x.copy(), self.P.copy()

    def update(self, z):
        """Measurement update using sigma points through h."""
        z = np.array(z, dtype=float)

        # Re-generate sigma points from predicted state
        sigmas = self._generate_sigma_points(self.x, self.P)

        # Transform through measurement function
        sigmas_meas = np.array([self.h(s) for s in sigmas])

        # Predicted measurement mean
        z_pred = np.sum(self.Wm[:, None] * sigmas_meas, axis=0)

        # Innovation covariance S
        S = np.zeros((self.m, self.m))
        for i in range(self.n_sigma):
            dz = sigmas_meas[i] - z_pred
            S += self.Wc[i] * np.outer(dz, dz)
        S += self.R

        # Cross-covariance Pxz
        Pxz = np.zeros((self.n, self.m))
        for i in range(self.n_sigma):
            dx = sigmas[i] - self.x
            dz = sigmas_meas[i] - z_pred
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)

        # Innovation
        y = z - z_pred

        # Update
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T

        return self.x.copy(), self.P.copy(), K.copy(), y.copy()

    def run(self, measurements, controls=None):
        """Run the UKF over a sequence of measurements."""
        results = {
            "x_predictions": [],
            "P_predictions": [],
            "x_estimates": [],
            "P_estimates": [],
            "kalman_gains": [],
            "innovations": [],
        }
        for k, z in enumerate(measurements):
            u = controls[k] if controls is not None else None
            x_pred, P_pred = self.predict(u)
            results["x_predictions"].append(x_pred)
            results["P_predictions"].append(P_pred)

            x_est, P_est, K, innov = self.update(z)
            results["x_estimates"].append(x_est)
            results["P_estimates"].append(P_est)
            results["kalman_gains"].append(K)
            results["innovations"].append(innov)

        return results

    def get_sigma_points(self):
        """Return current sigma points (useful for visualization)."""
        return self._generate_sigma_points(self.x, self.P)
