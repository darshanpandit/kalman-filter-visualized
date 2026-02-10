"""Interacting Multiple Model (IMM) filter — pure numpy implementation.

Based on Blom & Bar-Shalom (1988). Runs N sub-filters (KF or EKF)
in parallel, mixing their outputs via a Markov transition probability
matrix. The mode probabilities evolve over time.

Reference:
    Blom, H.A.P. and Bar-Shalom, Y. (1988) "The Interacting Multiple
    Model Algorithm for Systems with Markovian Switching Coefficients."
    IEEE Trans. Automatic Control, 33(8), pp. 780-783.
"""

from __future__ import annotations

import numpy as np
from copy import deepcopy


class IMMFilter:
    """Interacting Multiple Model filter.

    Parameters
    ----------
    filters : list
        List of KalmanFilter or ExtendedKalmanFilter instances.
        Each must have .x, .P, .predict(u), .update(z) interface.
    transition_matrix : np.ndarray, shape (N, N)
        Markov transition probability matrix. transition_matrix[i, j]
        = P(model j at k | model i at k-1).
    mode_probabilities : np.ndarray, shape (N,) or None
        Initial mode probabilities. Default: uniform.
    """

    def __init__(
        self,
        filters: list,
        transition_matrix: np.ndarray,
        mode_probabilities: np.ndarray | None = None,
    ):
        self.filters = [deepcopy(f) for f in filters]
        self.N = len(filters)
        self.Pi = np.array(transition_matrix, dtype=float)
        assert self.Pi.shape == (self.N, self.N)

        if mode_probabilities is None:
            self.mu = np.ones(self.N) / self.N
        else:
            self.mu = np.array(mode_probabilities, dtype=float)
            self.mu /= self.mu.sum()

        self.n = self.filters[0].x.shape[0]
        # Combined state estimate
        self.x = self._combined_state()
        self.P = self._combined_covariance()

    def _combined_state(self) -> np.ndarray:
        """Compute combined state: weighted sum of filter states."""
        x = np.zeros(self.n)
        for j in range(self.N):
            x += self.mu[j] * self.filters[j].x
        return x

    def _combined_covariance(self) -> np.ndarray:
        """Compute combined covariance including spread of means."""
        x_bar = self._combined_state()
        P = np.zeros((self.n, self.n))
        for j in range(self.N):
            diff = self.filters[j].x - x_bar
            P += self.mu[j] * (self.filters[j].P + np.outer(diff, diff))
        return P

    def predict(self, u=None):
        """IMM prediction step: mix, then predict each sub-filter.

        Returns (x_predicted, P_predicted).
        """
        # Step 1: Compute mixing probabilities
        # c_bar[j] = sum_i Pi[i,j] * mu[i]
        c_bar = self.Pi.T @ self.mu  # (N,)
        c_bar = np.maximum(c_bar, 1e-20)  # avoid division by zero

        # mu_ij[i,j] = Pi[i,j] * mu[i] / c_bar[j]
        mu_ij = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                mu_ij[i, j] = self.Pi[i, j] * self.mu[i] / c_bar[j]

        # Step 2: Mix states and covariances for each sub-filter
        mixed_x = []
        mixed_P = []
        for j in range(self.N):
            x_mix = np.zeros(self.n)
            for i in range(self.N):
                x_mix += mu_ij[i, j] * self.filters[i].x
            mixed_x.append(x_mix)

            P_mix = np.zeros((self.n, self.n))
            for i in range(self.N):
                diff = self.filters[i].x - x_mix
                P_mix += mu_ij[i, j] * (self.filters[i].P + np.outer(diff, diff))
            mixed_P.append(P_mix)

        # Step 3: Set mixed state into each filter and predict
        for j in range(self.N):
            self.filters[j].x = mixed_x[j].copy()
            self.filters[j].P = mixed_P[j].copy()
            self.filters[j].predict(u)

        # Update mode priors (will be refined in update step)
        self.mu = c_bar.copy()

        self.x = self._combined_state()
        self.P = self._combined_covariance()
        return self.x.copy(), self.P.copy()

    def update(self, z):
        """IMM update step: update each sub-filter, reweight modes.

        Returns (x_updated, P_updated, model_probabilities).
        """
        z = np.array(z, dtype=float)

        # Update each sub-filter and compute likelihoods
        likelihoods = np.zeros(self.N)
        for j in range(self.N):
            x_pred = self.filters[j].x.copy()
            P_pred = self.filters[j].P.copy()

            self.filters[j].update(z)

            # Compute innovation likelihood
            # Get the measurement function
            if hasattr(self.filters[j], 'h'):
                z_pred = self.filters[j].h(x_pred)
                H = self.filters[j].H_jacobian(x_pred)
            elif hasattr(self.filters[j], 'H'):
                z_pred = self.filters[j].H @ x_pred
                H = self.filters[j].H
            else:
                raise ValueError("Filter must have h() or H attribute")

            innov = z - z_pred
            S = H @ P_pred @ H.T + self.filters[j].R
            # Gaussian likelihood
            d = len(innov)
            det_S = np.linalg.det(S)
            if det_S < 1e-30:
                det_S = 1e-30
            exponent = -0.5 * innov @ np.linalg.inv(S) @ innov
            likelihoods[j] = np.exp(exponent) / np.sqrt(
                (2 * np.pi) ** d * det_S
            )

        # Update mode probabilities
        self.mu = self.mu * likelihoods
        total = self.mu.sum()
        if total < 1e-30:
            self.mu = np.ones(self.N) / self.N
        else:
            self.mu /= total

        self.x = self._combined_state()
        self.P = self._combined_covariance()
        return self.x.copy(), self.P.copy(), self.mu.copy()

    def run(self, measurements, controls=None):
        """Run the IMM filter over a sequence of measurements.

        Returns
        -------
        dict with keys:
            x_estimates : list of combined state estimates
            P_estimates : list of combined covariances
            model_probabilities : list of mode probability vectors
        """
        results = {
            "x_predictions": [],
            "P_predictions": [],
            "x_estimates": [],
            "P_estimates": [],
            "model_probabilities": [],
        }

        for k, z in enumerate(measurements):
            u = controls[k] if controls is not None else None
            x_pred, P_pred = self.predict(u)
            results["x_predictions"].append(x_pred)
            results["P_predictions"].append(P_pred)

            x_est, P_est, mu = self.update(z)
            results["x_estimates"].append(x_est)
            results["P_estimates"].append(P_est)
            results["model_probabilities"].append(mu)

        return results
