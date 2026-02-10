"""Nadaraya-Watson estimator demonstrating how attention approximates Kalman filtering.

Based on Goel & Bartlett (2024, L4DC) "Can Transformers Learn to Solve
Problems Algorithmically?", Theorem 1: a single softmax-attention layer
can implement Nadaraya-Watson kernel regression, which recovers the
Kalman filter estimate for linear-Gaussian systems.

This is NOT a real transformer. It computes:
    x_hat[t] = softmax(similarity_matrix[t]) @ past_states

to show that attention over past observations approximates KF.
"""

from __future__ import annotations

import numpy as np


class NWKalmanEstimator:
    """Nadaraya-Watson kernel estimator mimicking transformer attention.

    For a 1D linear-Gaussian state-space model:
        x[t] = F * x[t-1] + w,   w ~ N(0, Q)
        z[t] = H * x[t-1] + v,   v ~ N(0, R)

    The estimator uses a Gaussian kernel over observation similarities
    to produce a weighted average of past state estimates, analogous to
    how a single attention head computes softmax(Q K^T / sqrt(d)) V.

    Parameters
    ----------
    bandwidth : float
        Kernel bandwidth (analogous to 1/sqrt(d_k) temperature).
        Smaller = sharper attention. Default 1.0.
    """

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth

    def estimate(self, observations: np.ndarray) -> np.ndarray:
        """Estimate hidden states from a sequence of observations.

        Parameters
        ----------
        observations : np.ndarray, shape (T,) or (T, d_obs)
            Sequence of noisy observations.

        Returns
        -------
        estimates : np.ndarray, shape (T,) or (T, d_obs)
            Estimated states at each time step.
        """
        obs = np.atleast_2d(observations)
        if obs.shape[0] == 1 and observations.ndim == 1:
            obs = obs.T  # (T, 1)
        T, d = obs.shape

        estimates = np.zeros_like(obs)
        # First observation: trivial estimate
        estimates[0] = obs[0]

        for t in range(1, T):
            # Keys: past observations z[0..t-1]
            keys = obs[:t]  # (t, d)
            # Query: current observation z[t]
            query = obs[t]  # (d,)

            # Similarity = -||query - key||^2 / (2 * bandwidth^2)
            diff = keys - query[np.newaxis, :]  # (t, d)
            sq_dist = np.sum(diff ** 2, axis=1)  # (t,)
            logits = -sq_dist / (2 * self.bandwidth ** 2)

            # Softmax (numerically stable)
            logits -= np.max(logits)
            weights = np.exp(logits)
            weights /= np.sum(weights)

            # Values: past observations (in this simple model, best
            # estimate of x[i] given z[i] alone is just z[i])
            values = obs[:t]  # (t, d)

            # Weighted sum = attention output
            estimates[t] = weights @ values

        if observations.ndim == 1:
            return estimates.ravel()
        return estimates

    def attention_weights(self, observations: np.ndarray) -> np.ndarray:
        """Compute the full attention weight matrix for visualization.

        Parameters
        ----------
        observations : np.ndarray, shape (T,) or (T, d_obs)

        Returns
        -------
        weights : np.ndarray, shape (T, T)
            Lower-triangular attention matrix. weights[t, j] is the
            attention from step t to past step j.
        """
        obs = np.atleast_2d(observations)
        if obs.shape[0] == 1 and observations.ndim == 1:
            obs = obs.T
        T, d = obs.shape

        W = np.zeros((T, T))
        W[0, 0] = 1.0

        for t in range(1, T):
            keys = obs[:t]
            query = obs[t]
            diff = keys - query[np.newaxis, :]
            sq_dist = np.sum(diff ** 2, axis=1)
            logits = -sq_dist / (2 * self.bandwidth ** 2)
            logits -= np.max(logits)
            weights = np.exp(logits)
            weights /= np.sum(weights)
            W[t, :t] = weights

        return W
