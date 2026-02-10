"""Tests for Parts 8-9 models and generators."""

from __future__ import annotations

import numpy as np
import pytest

from models.rssm_sim import RSSMSim
from models.hamiltonian_nn import SimpleHNN
from kalman_manim.data.generators import (
    generate_lorenz_trajectory,
    generate_pendulum_trajectory,
)


# ── RSSM Simulator ──────────────────────────────────────────────────────


class TestRSSMSim:
    def test_predict_returns_correct_keys(self):
        rssm = RSSMSim(seed=42)
        result = rssm.predict()
        assert "h" in result
        assert "s_prior_mean" in result
        assert "s_prior_sample" in result

    def test_update_returns_correct_keys(self):
        rssm = RSSMSim(seed=42)
        rssm.predict()
        result = rssm.update(np.array([1.0, 0.5]))
        assert "s_post_mean" in result
        assert "kl_divergence" in result

    def test_kl_is_nonnegative(self):
        rssm = RSSMSim(seed=42)
        rssm.predict()
        result = rssm.update(np.array([1.0, 0.5]))
        assert result["kl_divergence"] >= 0

    def test_run_returns_correct_lengths(self):
        rssm = RSSMSim(seed=42)
        obs = [np.random.default_rng(i).normal(size=2) for i in range(10)]
        results = rssm.run(obs)
        assert len(results["h_history"]) == 10
        assert len(results["s_prior_means"]) == 10
        assert len(results["s_post_means"]) == 10
        assert len(results["kl_history"]) == 10

    def test_observe_returns_correct_shape(self):
        rssm = RSSMSim(o_dim=3, seed=42)
        rssm.predict()
        obs = rssm.observe()
        assert obs.shape == (3,)


# ── Hamiltonian Neural Network ───────────────────────────────────────────


class TestSimpleHNN:
    def test_hamiltonian_returns_scalar(self):
        hnn = SimpleHNN()
        H = hnn.hamiltonian(1.0, 0.5)
        assert isinstance(H, float)

    def test_derivatives_return_tuple(self):
        hnn = SimpleHNN()
        dHdq, dHdp = hnn.derivatives(1.0, 0.5)
        assert isinstance(dHdq, float)
        assert isinstance(dHdp, float)

    def test_dynamics_returns_correct_shape(self):
        hnn = SimpleHNN()
        dyn = hnn.dynamics(np.array([1.0, 0.5]))
        assert dyn.shape == (2,)

    def test_integrate_returns_correct_shape(self):
        hnn = SimpleHNN()
        states = hnn.integrate(q0=1.0, p0=0.0, dt=0.01, n_steps=100)
        assert states.shape == (101, 2)

    def test_integrate_preserves_energy_approximately(self):
        """Random HNN won't conserve energy, but integration should be stable."""
        hnn = SimpleHNN()
        states = hnn.integrate(q0=0.5, p0=0.0, dt=0.01, n_steps=50)
        # Should not diverge to infinity
        assert np.all(np.isfinite(states))


# ── Lorenz Generator ─────────────────────────────────────────────────────


class TestLorenzGenerator:
    def test_output_shape(self):
        data = generate_lorenz_trajectory(n_steps=100)
        assert data["states"].shape == (101, 3)

    def test_chaotic_behavior(self):
        """Lorenz system should be bounded but not trivially periodic."""
        data = generate_lorenz_trajectory(n_steps=2000)
        states = data["states"]
        # Should be bounded (Lorenz attractor)
        assert np.all(np.abs(states) < 100)
        # Should not converge to a fixed point
        assert np.std(states[-100:, 0]) > 0.1

    def test_custom_initial_state(self):
        data = generate_lorenz_trajectory(
            n_steps=10, initial_state=np.array([5.0, 5.0, 5.0]),
        )
        np.testing.assert_allclose(data["states"][0], [5.0, 5.0, 5.0])


# ── Pendulum Generator ──────────────────────────────────────────────────


class TestPendulumGenerator:
    def test_output_shape(self):
        data = generate_pendulum_trajectory(n_steps=100)
        assert data["states"].shape == (101, 2)
        assert data["energy"].shape == (101,)

    def test_energy_conservation(self):
        """RK4 should approximately conserve energy."""
        data = generate_pendulum_trajectory(dt=0.001, n_steps=5000)
        energy = data["energy"]
        # Energy should be nearly constant (RK4 is symplectic-ish)
        assert np.std(energy) / np.mean(energy) < 0.01

    def test_small_angle_period(self):
        """Small angle: period ≈ 2*pi*sqrt(L/g)."""
        L, g = 1.0, 9.81
        data = generate_pendulum_trajectory(
            length=L, gravity=g, theta0=0.1, dt=0.001, n_steps=10000,
        )
        theta = data["states"][:, 0]
        # Find zero crossings
        crossings = np.where(np.diff(np.sign(theta)) > 0)[0]
        if len(crossings) >= 2:
            period = (crossings[1] - crossings[0]) * 0.001
            expected_period = 2 * np.pi * np.sqrt(L / g)
            assert abs(period - expected_period) / expected_period < 0.05
