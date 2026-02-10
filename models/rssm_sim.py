"""Teaching RSSM (Recurrent State-Space Model) simulator.

A simplified numpy simulation of the RSSM architecture from:
    Hafner et al. (2019) "Learning Latent Dynamics for Planning from Pixels"

This is NOT a real RSSM. It demonstrates the information flow:
deterministic path (GRU-like) + stochastic path (prior/posterior),
using linear dynamics to show predict/update parallels with KF.

The teaching model:
    Deterministic: h_t = A_h * h_{t-1} + B_h * s_{t-1} + C_h * a_{t-1}
    Prior:         s_t ~ N(W_prior * h_t, sigma_prior^2)
    Posterior:     s_t ~ N(W_post * [h_t, o_t], sigma_post^2)
    Observation:   o_t = D * [h_t, s_t] + noise
"""

from __future__ import annotations

import numpy as np


class RSSMSim:
    """Teaching RSSM simulator with linear latent dynamics.

    Parameters
    ----------
    h_dim : int
        Deterministic state dimension.
    s_dim : int
        Stochastic state dimension.
    a_dim : int
        Action dimension.
    o_dim : int
        Observation dimension.
    sigma_prior : float
        Prior std dev.
    sigma_post : float
        Posterior std dev (< sigma_prior for information gain).
    seed : int or None
    """

    def __init__(
        self,
        h_dim: int = 4,
        s_dim: int = 2,
        a_dim: int = 2,
        o_dim: int = 2,
        sigma_prior: float = 0.5,
        sigma_post: float = 0.1,
        seed: int | None = None,
    ):
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.o_dim = o_dim
        self.sigma_prior = sigma_prior
        self.sigma_post = sigma_post
        self.rng = np.random.default_rng(seed)

        # Initialize random (but stable) linear dynamics
        self.A_h = 0.9 * np.eye(h_dim) + 0.05 * self.rng.normal(size=(h_dim, h_dim))
        self.B_h = 0.3 * self.rng.normal(size=(h_dim, s_dim))
        self.C_h = 0.3 * self.rng.normal(size=(h_dim, a_dim))
        self.W_prior = 0.5 * self.rng.normal(size=(s_dim, h_dim))
        self.W_post = 0.5 * self.rng.normal(size=(s_dim, h_dim + o_dim))
        self.D = 0.5 * self.rng.normal(size=(o_dim, h_dim + s_dim))

        # State
        self.h = np.zeros(h_dim)
        self.s = np.zeros(s_dim)

    def predict(self, action: np.ndarray | None = None) -> dict:
        """Prediction step (prior): deterministic transition + stochastic prior.

        Returns dict with h_pred, s_prior_mean, s_prior_sample.
        """
        if action is None:
            action = np.zeros(self.a_dim)

        # Deterministic transition
        h_new = self.A_h @ self.h + self.B_h @ self.s + self.C_h @ action
        self.h = h_new

        # Stochastic prior
        s_prior_mean = self.W_prior @ self.h
        s_prior = s_prior_mean + self.sigma_prior * self.rng.normal(size=self.s_dim)
        self.s = s_prior

        return {
            "h": self.h.copy(),
            "s_prior_mean": s_prior_mean.copy(),
            "s_prior_sample": s_prior.copy(),
        }

    def update(self, observation: np.ndarray) -> dict:
        """Update step (posterior): incorporate observation.

        Returns dict with s_post_mean, s_post_sample, kl_divergence.
        """
        # Posterior
        concat = np.concatenate([self.h, observation])
        s_post_mean = self.W_post @ concat
        s_post = s_post_mean + self.sigma_post * self.rng.normal(size=self.s_dim)

        # Approximate KL between posterior and prior
        s_prior_mean = self.W_prior @ self.h
        kl = 0.5 * (
            np.sum((s_post_mean - s_prior_mean) ** 2)
            / self.sigma_prior ** 2
            + self.s_dim * (self.sigma_post ** 2 / self.sigma_prior ** 2 - 1)
            + self.s_dim * np.log(self.sigma_prior ** 2 / self.sigma_post ** 2)
        )

        self.s = s_post

        return {
            "s_post_mean": s_post_mean.copy(),
            "s_post_sample": s_post.copy(),
            "kl_divergence": float(kl),
        }

    def observe(self) -> np.ndarray:
        """Generate an observation from current state."""
        concat = np.concatenate([self.h, self.s])
        return self.D @ concat + 0.1 * self.rng.normal(size=self.o_dim)

    def run(
        self,
        observations: list[np.ndarray],
        actions: list[np.ndarray] | None = None,
    ) -> dict:
        """Run predict/update loop.

        Returns dict with lists of predictions and posteriors.
        """
        T = len(observations)
        if actions is None:
            actions = [np.zeros(self.a_dim)] * T

        results = {
            "h_history": [],
            "s_prior_means": [],
            "s_post_means": [],
            "kl_history": [],
        }

        for t in range(T):
            pred = self.predict(actions[t])
            results["h_history"].append(pred["h"])
            results["s_prior_means"].append(pred["s_prior_mean"])

            upd = self.update(observations[t])
            results["s_post_means"].append(upd["s_post_mean"])
            results["kl_history"].append(upd["kl_divergence"])

        return results
