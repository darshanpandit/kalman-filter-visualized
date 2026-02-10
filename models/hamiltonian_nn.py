"""Simple Hamiltonian Neural Network for pendulum dynamics.

Based on Greydanus et al. (2019, NeurIPS):
    "Hamiltonian Neural Networks"

A 2-layer MLP that outputs a scalar H(q, p), with dynamics derived
from Hamilton's equations via finite-difference gradients:
    dq/dt = dH/dp,  dp/dt = -dH/dq

Pre-trained weights loaded from data/hnn_weights/pendulum.npz.
"""

from __future__ import annotations

import os
import numpy as np


class SimpleHNN:
    """Simple Hamiltonian Neural Network for 1D pendulum.

    State: [q, p] where q = angle, p = angular momentum.
    Network: input(2) -> hidden(32) -> hidden(32) -> output(1) = H(q,p).

    Parameters
    ----------
    weights_path : str or None
        Path to .npz with W1, b1, W2, b2, W3, b3.
        If None, uses random initialization.
    """

    def __init__(self, weights_path: str | None = None):
        if weights_path and os.path.exists(weights_path):
            data = np.load(weights_path)
            self.W1 = data["W1"]
            self.b1 = data["b1"]
            self.W2 = data["W2"]
            self.b2 = data["b2"]
            self.W3 = data["W3"]
            self.b3 = data["b3"]
        else:
            # Random init (untrained — for testing)
            rng = np.random.default_rng(42)
            self.W1 = rng.normal(0, 0.5, (2, 32))
            self.b1 = np.zeros(32)
            self.W2 = rng.normal(0, 0.5, (32, 32))
            self.b2 = np.zeros(32)
            self.W3 = rng.normal(0, 0.5, (32, 1))
            self.b3 = np.zeros(1)

    def _forward(self, x: np.ndarray) -> float:
        """Compute H(q, p)."""
        h = np.tanh(x @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)
        return float((h @ self.W3 + self.b3)[0])

    def hamiltonian(self, q: float, p: float) -> float:
        """Evaluate the learned Hamiltonian."""
        return self._forward(np.array([q, p]))

    def derivatives(self, q: float, p: float, eps: float = 1e-4) -> tuple:
        """Compute dH/dq and dH/dp via finite differences.

        Returns (dH_dq, dH_dp).
        """
        dH_dq = (self._forward(np.array([q + eps, p]))
                  - self._forward(np.array([q - eps, p]))) / (2 * eps)
        dH_dp = (self._forward(np.array([q, p + eps]))
                  - self._forward(np.array([q, p - eps]))) / (2 * eps)
        return dH_dq, dH_dp

    def dynamics(self, state: np.ndarray) -> np.ndarray:
        """Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq."""
        q, p = state
        dH_dq, dH_dp = self.derivatives(q, p)
        return np.array([dH_dp, -dH_dq])

    def integrate(
        self, q0: float, p0: float, dt: float = 0.01, n_steps: int = 100,
    ) -> np.ndarray:
        """Integrate dynamics using RK4.

        Returns states array of shape (n_steps+1, 2).
        """
        states = np.zeros((n_steps + 1, 2))
        states[0] = [q0, p0]

        for i in range(n_steps):
            s = states[i]
            k1 = dt * self.dynamics(s)
            k2 = dt * self.dynamics(s + 0.5 * k1)
            k3 = dt * self.dynamics(s + 0.5 * k2)
            k4 = dt * self.dynamics(s + k3)
            states[i + 1] = s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return states


def train_pendulum_hnn(
    n_samples: int = 5000,
    n_epochs: int = 2000,
    lr: float = 0.001,
    seed: int = 42,
) -> SimpleHNN:
    """Train a SimpleHNN on pendulum data.

    True Hamiltonian: H = 0.5 * p^2 + (1 - cos(q))
    """
    rng = np.random.default_rng(seed)

    # Generate training data
    q_data = rng.uniform(-np.pi, np.pi, n_samples)
    p_data = rng.uniform(-2, 2, n_samples)

    # True derivatives: dH/dq = sin(q), dH/dp = p
    dHdq_true = np.sin(q_data)
    dHdp_true = p_data

    hnn = SimpleHNN()

    for epoch in range(n_epochs):
        total_loss = 0.0
        # Mini-batch SGD
        perm = rng.permutation(n_samples)
        batch_size = 64

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = perm[start:end]

            for idx in batch_idx:
                q, p = q_data[idx], p_data[idx]
                dHdq_pred, dHdp_pred = hnn.derivatives(q, p)

                # Loss = (dH/dq - sin(q))^2 + (dH/dp - p)^2
                loss_dq = dHdq_pred - dHdq_true[idx]
                loss_dp = dHdp_pred - dHdp_true[idx]

                # Finite-diff gradient w.r.t. weights (simplified)
                eps = 1e-4
                x = np.array([q, p])

                # Update via numerical gradient on W3 (output layer only
                # for speed — sufficient for demo quality)
                for j in range(hnn.W3.shape[0]):
                    for k in range(hnn.W3.shape[1]):
                        hnn.W3[j, k] += eps
                        dq_plus, dp_plus = hnn.derivatives(q, p)
                        hnn.W3[j, k] -= 2 * eps
                        dq_minus, dp_minus = hnn.derivatives(q, p)
                        hnn.W3[j, k] += eps  # restore

                        grad = ((dq_plus - dq_minus) * loss_dq
                                + (dp_plus - dp_minus) * loss_dp) / (2 * eps)
                        hnn.W3[j, k] -= lr * grad

                total_loss += loss_dq ** 2 + loss_dp ** 2

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: loss = {total_loss / n_samples:.6f}")

    return hnn


def save_hnn_weights(hnn: SimpleHNN, path: str):
    """Save HNN weights to .npz."""
    np.savez(
        path,
        W1=hnn.W1, b1=hnn.b1,
        W2=hnn.W2, b2=hnn.b2,
        W3=hnn.W3, b3=hnn.b3,
    )
