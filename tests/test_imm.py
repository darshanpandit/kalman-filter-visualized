"""Tests for IMM filter implementation."""

from __future__ import annotations

import numpy as np
import pytest

from filters.kalman import KalmanFilter
from filters.imm import IMMFilter
from kalman_manim.data.generators import generate_mode_switching_trajectory


class TestIMMFilter:
    def _make_cv_filter(self, dt=0.5, x0=None):
        """Constant velocity KF sub-model."""
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                       [0, 0, 1, 0], [0, 0, 0, 1]])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        Q = 0.01 * np.eye(4)
        R = 0.25 * np.eye(2)
        x0 = x0 if x0 is not None else np.zeros(4)
        return KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=np.eye(4))

    def _make_ct_filter(self, dt=0.5, omega=0.15, x0=None):
        """Coordinated turn KF sub-model (linearized)."""
        cos_w = np.cos(omega * dt)
        sin_w = np.sin(omega * dt)
        F = np.array([
            [1, 0, sin_w / omega, -(1 - cos_w) / omega],
            [0, 1, (1 - cos_w) / omega, sin_w / omega],
            [0, 0, cos_w, -sin_w],
            [0, 0, sin_w, cos_w],
        ])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        Q = 0.05 * np.eye(4)
        R = 0.25 * np.eye(2)
        x0 = x0 if x0 is not None else np.zeros(4)
        return KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=np.eye(4))

    def test_interface_consistency(self):
        """IMM has predict(), update(), run() like other filters."""
        cv = self._make_cv_filter()
        ct = self._make_ct_filter()
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMFilter(filters=[cv, ct], transition_matrix=Pi)

        assert hasattr(imm, "predict")
        assert hasattr(imm, "update")
        assert hasattr(imm, "run")

    def test_initial_probabilities_uniform(self):
        cv = self._make_cv_filter()
        ct = self._make_ct_filter()
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMFilter(filters=[cv, ct], transition_matrix=Pi)

        np.testing.assert_allclose(imm.mu, [0.5, 0.5])

    def test_custom_initial_probabilities(self):
        cv = self._make_cv_filter()
        ct = self._make_ct_filter()
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMFilter(
            filters=[cv, ct], transition_matrix=Pi,
            mode_probabilities=np.array([0.8, 0.2]),
        )

        np.testing.assert_allclose(imm.mu, [0.8, 0.2])

    def test_predict_returns_correct_shape(self):
        cv = self._make_cv_filter()
        ct = self._make_ct_filter()
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMFilter(filters=[cv, ct], transition_matrix=Pi)

        x, P = imm.predict()
        assert x.shape == (4,)
        assert P.shape == (4, 4)

    def test_update_returns_probabilities(self):
        cv = self._make_cv_filter()
        ct = self._make_ct_filter()
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMFilter(filters=[cv, ct], transition_matrix=Pi)

        imm.predict()
        x, P, mu = imm.update(np.array([0.5, 0.0]))
        assert x.shape == (4,)
        assert P.shape == (4, 4)
        assert mu.shape == (2,)
        assert pytest.approx(mu.sum(), abs=1e-10) == 1.0

    def test_run_returns_correct_lengths(self):
        cv = self._make_cv_filter()
        ct = self._make_ct_filter()
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMFilter(filters=[cv, ct], transition_matrix=Pi)

        measurements = [np.array([i * 0.1, 0.0]) for i in range(15)]
        results = imm.run(measurements)
        assert len(results["x_estimates"]) == 15
        assert len(results["model_probabilities"]) == 15

    def test_mode_probability_converges(self):
        """On a straight-line trajectory, CV model should dominate."""
        x0 = np.array([0.0, 0.0, 0.5, 0.0])
        cv = self._make_cv_filter(x0=x0)
        ct = self._make_ct_filter(x0=x0)
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMFilter(filters=[cv, ct], transition_matrix=Pi)

        # Straight line: x increases at 0.5/step
        rng = np.random.default_rng(42)
        for t in range(1, 31):
            imm.predict()
            z = np.array([t * 0.25, 0.0]) + rng.normal(0, 0.1, size=2)
            imm.update(z)

        # CV model (index 0) should have higher probability
        assert imm.mu[0] > 0.5

    def test_single_model_matches_kf(self):
        """IMM with one model should match standalone KF."""
        x0 = np.array([0.0, 0.0, 1.0, 0.0])
        cv1 = self._make_cv_filter(x0=x0.copy())
        cv_standalone = self._make_cv_filter(x0=x0.copy())
        Pi = np.array([[1.0]])
        imm = IMMFilter(filters=[cv1], transition_matrix=Pi)

        rng = np.random.default_rng(42)
        for t in range(10):
            imm.predict()
            cv_standalone.predict()
            z = np.array([t * 0.5, 0.0]) + rng.normal(0, 0.1, size=2)
            imm.update(z)
            cv_standalone.update(z)

        np.testing.assert_allclose(imm.x, cv_standalone.x, atol=1e-6)
