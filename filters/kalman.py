"""Standard (linear) Kalman Filter — pure numpy implementation."""

import numpy as np


class KalmanFilter:
    """Discrete-time linear Kalman Filter.

    State model:
        x_k = F @ x_{k-1} + B @ u_k + w_k,   w_k ~ N(0, Q)
    Measurement model:
        z_k = H @ x_k + v_k,                   v_k ~ N(0, R)

    Parameters
    ----------
    F : np.ndarray  – State transition matrix (n x n)
    H : np.ndarray  – Measurement matrix (m x n)
    Q : np.ndarray  – Process noise covariance (n x n)
    R : np.ndarray  – Measurement noise covariance (m x m)
    B : np.ndarray  – Control input matrix (n x l), default zeros
    x0 : np.ndarray – Initial state estimate (n,)
    P0 : np.ndarray – Initial covariance estimate (n x n)
    """

    def __init__(self, F, H, Q, R, B=None, x0=None, P0=None):
        self.F = np.array(F, dtype=float)
        self.H = np.array(H, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.n = self.F.shape[0]  # state dimension
        self.m = self.H.shape[0]  # measurement dimension

        self.B = np.zeros((self.n, 1)) if B is None else np.array(B, dtype=float)
        self.x = np.zeros(self.n) if x0 is None else np.array(x0, dtype=float)
        self.P = np.eye(self.n) if P0 is None else np.array(P0, dtype=float)

    def predict(self, u=None):
        """Prediction step: propagate state and covariance forward.

        Returns (x_predicted, P_predicted).
        """
        if u is None:
            u = np.zeros(self.B.shape[1])
        u = np.array(u, dtype=float)

        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z):
        """Measurement update step: incorporate a new measurement.

        Returns (x_updated, P_updated, K, innovation).
        """
        z = np.array(z, dtype=float)

        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x.copy(), self.P.copy(), K.copy(), y.copy()

    def run(self, measurements, controls=None):
        """Run the filter over a sequence of measurements.

        Parameters
        ----------
        measurements : list of np.ndarray
            Sequence of measurement vectors.
        controls : list of np.ndarray or None
            Sequence of control inputs (same length as measurements).

        Returns
        -------
        dict with keys:
            x_predictions  : list of predicted state means
            P_predictions  : list of predicted covariances
            x_estimates    : list of updated state means
            P_estimates    : list of updated covariances
            kalman_gains   : list of Kalman gain matrices
            innovations    : list of innovation vectors
        """
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
