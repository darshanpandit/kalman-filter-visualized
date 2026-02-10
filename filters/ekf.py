"""Extended Kalman Filter — pure numpy implementation."""

from __future__ import annotations

import numpy as np


class ExtendedKalmanFilter:
    """Discrete-time Extended Kalman Filter.

    Nonlinear state model:
        x_k = f(x_{k-1}, u_k) + w_k,   w_k ~ N(0, Q)
    Nonlinear measurement model:
        z_k = h(x_k) + v_k,             v_k ~ N(0, R)

    The EKF linearizes f and h around the current estimate using Jacobians.

    Parameters
    ----------
    f : callable(x, u) -> np.ndarray
        State transition function.
    h : callable(x) -> np.ndarray
        Measurement function.
    F_jacobian : callable(x, u) -> np.ndarray
        Jacobian of f w.r.t. state x (n x n matrix).
    H_jacobian : callable(x) -> np.ndarray
        Jacobian of h w.r.t. state x (m x n matrix).
    Q : np.ndarray
        Process noise covariance (n x n).
    R : np.ndarray
        Measurement noise covariance (m x m).
    x0 : np.ndarray
        Initial state estimate.
    P0 : np.ndarray
        Initial covariance estimate.
    """

    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R, x0=None, P0=None):
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.n = self.Q.shape[0]
        self.m = self.R.shape[0]

        self.x = np.zeros(self.n) if x0 is None else np.array(x0, dtype=float)
        self.P = np.eye(self.n) if P0 is None else np.array(P0, dtype=float)

    def predict(self, u=None):
        """Prediction step using nonlinear f and Jacobian F."""
        F = self.F_jacobian(self.x, u)
        self.x = self.f(self.x, u)
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z):
        """Measurement update using nonlinear h and Jacobian H."""
        z = np.array(z, dtype=float)
        H = self.H_jacobian(self.x)

        # Innovation
        y = z - self.h(self.x)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form)
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x.copy(), self.P.copy(), K.copy(), y.copy()

    def run(self, measurements, controls=None):
        """Run the EKF over a sequence of measurements."""
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
