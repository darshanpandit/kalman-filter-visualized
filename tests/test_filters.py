"""Tests for all filter implementations (KF, EKF, UKF, PF)."""

import numpy as np
import pytest

from filters.kalman import KalmanFilter
from filters.ekf import ExtendedKalmanFilter
from filters.ukf import UnscentedKalmanFilter
from filters.particle import ParticleFilter
from kalman_manim.utils import (
    cov_to_ellipse_params,
    gaussian_product_1d,
    gaussian_product_2d,
)
from kalman_manim.data.generators import (
    generate_pedestrian_trajectory,
    generate_nonlinear_trajectory,
)


# ── cov_to_ellipse_params ──────────────────────────────────────────────────


class TestCovToEllipseParams:
    def test_identity_covariance(self):
        """Identity matrix → circle (equal width and height)."""
        params = cov_to_ellipse_params(np.eye(2), n_sigma=1.0)
        assert pytest.approx(params["width"], abs=1e-10) == 2.0
        assert pytest.approx(params["height"], abs=1e-10) == 2.0

    def test_diagonal_covariance(self):
        """Diagonal matrix → axis-aligned ellipse."""
        cov = np.diag([4.0, 1.0])
        params = cov_to_ellipse_params(cov, n_sigma=1.0)
        # Larger eigenvalue (4) gives major axis width = 2*sqrt(4) = 4
        assert pytest.approx(params["width"], abs=1e-10) == 4.0
        assert pytest.approx(params["height"], abs=1e-10) == 2.0

    def test_n_sigma_scaling(self):
        """n_sigma scales the axes linearly."""
        cov = np.eye(2)
        params_1 = cov_to_ellipse_params(cov, n_sigma=1.0)
        params_2 = cov_to_ellipse_params(cov, n_sigma=2.0)
        assert pytest.approx(params_2["width"], abs=1e-10) == 2 * params_1["width"]

    def test_rotated_covariance(self):
        """Off-diagonal elements → rotated ellipse."""
        cov = np.array([[2.0, 1.0], [1.0, 2.0]])
        params = cov_to_ellipse_params(cov, n_sigma=1.0)
        # Eigenvalues: 3 and 1 → axes: 2*sqrt(3) and 2*sqrt(1)
        assert pytest.approx(params["width"], abs=1e-6) == 2 * np.sqrt(3)
        assert pytest.approx(params["height"], abs=1e-6) == 2.0
        # 45 degree rotation
        assert pytest.approx(abs(params["angle"]), abs=0.01) == np.pi / 4


# ── Gaussian products ──────────────────────────────────────────────────────


class TestGaussianProduct1D:
    def test_equal_variances(self):
        """Equal variances → result is the midpoint with half variance."""
        mu, var = gaussian_product_1d(0.0, 1.0, 2.0, 1.0)
        assert pytest.approx(mu) == 1.0
        assert pytest.approx(var) == 0.5

    def test_one_very_certain(self):
        """One very tight Gaussian dominates."""
        mu, var = gaussian_product_1d(0.0, 100.0, 5.0, 0.01)
        assert pytest.approx(mu, abs=0.1) == 5.0
        assert var < 0.01


class TestGaussianProduct2D:
    def test_identity_covariances(self):
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([2.0, 2.0])
        cov = np.eye(2)
        mu_new, cov_new = gaussian_product_2d(mu1, cov, mu2, cov)
        np.testing.assert_allclose(mu_new, [1.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(cov_new, 0.5 * np.eye(2), atol=1e-10)


# ── Kalman Filter ──────────────────────────────────────────────────────────


class TestKalmanFilter:
    def _make_1d_position_filter(self):
        """Simple 1D position-only tracking filter."""
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[1.0]])
        return KalmanFilter(F=F, H=H, Q=Q, R=R, x0=np.array([0.0]), P0=np.eye(1))

    def test_predict_grows_uncertainty(self):
        kf = self._make_1d_position_filter()
        P_before = kf.P.copy()
        kf.predict()
        assert kf.P[0, 0] > P_before[0, 0]

    def test_update_shrinks_uncertainty(self):
        kf = self._make_1d_position_filter()
        kf.predict()
        P_after_pred = kf.P.copy()
        kf.update(np.array([1.0]))
        assert kf.P[0, 0] < P_after_pred[0, 0]

    def test_converges_to_true_value(self):
        """After many measurements at the same value, state should converge."""
        kf = self._make_1d_position_filter()
        true_value = 5.0
        for _ in range(100):
            kf.predict()
            kf.update(np.array([true_value]))
        assert pytest.approx(kf.x[0], abs=0.2) == true_value

    def test_2d_constant_velocity(self):
        """2D position+velocity filter tracks a linear trajectory."""
        dt = 1.0
        F = np.array([[1, dt], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.array([[0.01, 0], [0, 0.01]])
        R = np.array([[0.5]])

        kf = KalmanFilter(
            F=F, H=H, Q=Q, R=R,
            x0=np.array([0.0, 1.0]),  # start at 0, velocity=1
            P0=np.eye(2),
        )

        # True trajectory: position = t, velocity = 1
        rng = np.random.default_rng(42)
        for t in range(1, 51):
            kf.predict()
            z = np.array([t + rng.normal(0, 0.5)])
            kf.update(z)

        # After 50 steps, should be near position=50, velocity=1
        assert pytest.approx(kf.x[0], abs=2.0) == 50.0
        assert pytest.approx(kf.x[1], abs=0.5) == 1.0

    def test_run_returns_correct_lengths(self):
        kf = self._make_1d_position_filter()
        measurements = [np.array([i * 0.1]) for i in range(20)]
        results = kf.run(measurements)
        assert len(results["x_estimates"]) == 20
        assert len(results["P_estimates"]) == 20
        assert len(results["kalman_gains"]) == 20
        assert len(results["innovations"]) == 20


# ── Trajectory Generator ──────────────────────────────────────────────────


class TestTrajectoryGenerator:
    def test_output_shapes(self):
        data = generate_pedestrian_trajectory(n_steps=50, seed=42)
        assert data["true_states"].shape == (51, 4)
        assert data["measurements"].shape == (50, 2)

    def test_reproducibility(self):
        d1 = generate_pedestrian_trajectory(n_steps=10, seed=123)
        d2 = generate_pedestrian_trajectory(n_steps=10, seed=123)
        np.testing.assert_array_equal(d1["true_states"], d2["true_states"])
        np.testing.assert_array_equal(d1["measurements"], d2["measurements"])

    def test_measurements_are_noisy(self):
        data = generate_pedestrian_trajectory(
            n_steps=50, measurement_noise_std=1.0, seed=42
        )
        # Measurements should differ from true positions
        true_pos = data["true_states"][1:, :2]
        diff = np.linalg.norm(data["measurements"] - true_pos, axis=1)
        assert np.mean(diff) > 0.3  # should be noticeably noisy

    def test_nonlinear_trajectory_shapes(self):
        data = generate_nonlinear_trajectory(n_steps=30, seed=99)
        assert data["true_states"].shape == (31, 5)  # [x, y, vx, vy, omega]
        assert data["measurements"].shape == (30, 2)

    def test_nonlinear_trajectory_is_curved(self):
        data = generate_nonlinear_trajectory(
            n_steps=40, turn_rate=0.3, seed=99,
        )
        # With a positive turn rate, the path should curve
        positions = data["true_states"][:, :2]
        # Check that final heading differs from initial (path is not straight)
        initial_dir = positions[5] - positions[0]
        final_dir = positions[-1] - positions[-6]
        cos_angle = np.dot(initial_dir, final_dir) / (
            np.linalg.norm(initial_dir) * np.linalg.norm(final_dir) + 1e-10
        )
        # Angle should be noticeably different from perfectly straight
        assert cos_angle < 0.99


# ── Extended Kalman Filter ─────────────────────────────────────────────────


class TestExtendedKalmanFilter:
    """EKF tests using a linear system (should match KF exactly)."""

    def _linear_f(self, x, u):
        F = np.array([[1, 1], [0, 1]])
        return F @ x

    def _linear_F(self, x, u):
        return np.array([[1, 1], [0, 1]])

    def _linear_h(self, x):
        return np.array([x[0]])

    def _linear_H(self, x):
        return np.array([[1, 0]])

    def test_ekf_matches_kf_on_linear_system(self):
        """On a linear system, EKF should produce same results as KF."""
        Q = np.diag([0.01, 0.01])
        R = np.array([[0.5]])
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2)

        ekf = ExtendedKalmanFilter(
            f=self._linear_f, h=self._linear_h,
            F_jacobian=self._linear_F, H_jacobian=self._linear_H,
            Q=Q, R=R, x0=x0.copy(), P0=P0.copy(),
        )

        kf = KalmanFilter(
            F=np.array([[1, 1], [0, 1]]),
            H=np.array([[1, 0]]),
            Q=Q, R=R, x0=x0.copy(), P0=P0.copy(),
        )

        rng = np.random.default_rng(42)
        for t in range(1, 21):
            ekf.predict()
            kf.predict()
            z = np.array([t + rng.normal(0, 0.5)])
            ekf.update(z)
            kf.update(z)

        np.testing.assert_allclose(ekf.x, kf.x, atol=1e-10)
        np.testing.assert_allclose(ekf.P, kf.P, atol=1e-10)

    def test_ekf_converges(self):
        ekf = ExtendedKalmanFilter(
            f=self._linear_f, h=self._linear_h,
            F_jacobian=self._linear_F, H_jacobian=self._linear_H,
            Q=np.diag([0.01, 0.01]), R=np.array([[0.5]]),
            x0=np.array([0.0, 1.0]), P0=np.eye(2),
        )
        for t in range(1, 51):
            ekf.predict()
            ekf.update(np.array([float(t)]))
        assert pytest.approx(ekf.x[0], abs=2.0) == 50.0

    def test_ekf_run_returns_correct_lengths(self):
        ekf = ExtendedKalmanFilter(
            f=self._linear_f, h=self._linear_h,
            F_jacobian=self._linear_F, H_jacobian=self._linear_H,
            Q=np.diag([0.01, 0.01]), R=np.array([[0.5]]),
            x0=np.array([0.0, 1.0]), P0=np.eye(2),
        )
        measurements = [np.array([float(i)]) for i in range(15)]
        results = ekf.run(measurements)
        assert len(results["x_estimates"]) == 15


# ── Unscented Kalman Filter ───────────────────────────────────────────────


class TestUnscentedKalmanFilter:
    def _linear_f(self, x, u):
        return np.array([x[0] + x[1], x[1]])

    def _linear_h(self, x):
        return np.array([x[0]])

    def test_ukf_converges_on_linear(self):
        """UKF on a linear system should converge to the true state."""
        ukf = UnscentedKalmanFilter(
            f=self._linear_f, h=self._linear_h,
            Q=np.diag([0.01, 0.01]), R=np.array([[0.5]]),
            x0=np.array([0.0, 1.0]), P0=np.eye(2),
        )
        for t in range(1, 51):
            ukf.predict()
            ukf.update(np.array([float(t)]))
        assert pytest.approx(ukf.x[0], abs=2.0) == 50.0

    def test_ukf_sigma_points_count(self):
        """For n=2, should generate 2*2+1=5 sigma points."""
        ukf = UnscentedKalmanFilter(
            f=self._linear_f, h=self._linear_h,
            Q=np.diag([0.01, 0.01]), R=np.array([[0.5]]),
            x0=np.array([0.0, 0.0]), P0=np.eye(2),
        )
        sigmas = ukf.get_sigma_points()
        assert sigmas.shape == (5, 2)

    def test_ukf_predict_grows_uncertainty(self):
        ukf = UnscentedKalmanFilter(
            f=self._linear_f, h=self._linear_h,
            Q=np.diag([0.1, 0.1]), R=np.array([[0.5]]),
            x0=np.array([0.0, 0.0]), P0=np.eye(2),
        )
        P_before = ukf.P.copy()
        ukf.predict()
        # Trace of P should increase (uncertainty grows)
        assert np.trace(ukf.P) > np.trace(P_before)

    def test_ukf_run_returns_correct_lengths(self):
        ukf = UnscentedKalmanFilter(
            f=self._linear_f, h=self._linear_h,
            Q=np.diag([0.01, 0.01]), R=np.array([[0.5]]),
            x0=np.array([0.0, 1.0]), P0=np.eye(2),
        )
        measurements = [np.array([float(i)]) for i in range(15)]
        results = ukf.run(measurements)
        assert len(results["x_estimates"]) == 15


# ── Particle Filter ───────────────────────────────────────────────────────


class TestParticleFilter:
    def _pf_f(self, x, u, noise):
        return np.array([x[0] + noise[0]])

    def _pf_h(self, x):
        return np.array([x[0]])

    def test_pf_converges_to_measurement(self):
        """PF with no dynamics noise should converge to measurements."""
        pf = ParticleFilter(
            f=self._pf_f, h=self._pf_h,
            Q=np.array([[0.01]]), R=np.array([[0.1]]),
            n_particles=500,
            x0=np.array([0.0]), P0=np.array([[1.0]]),
            seed=42,
        )
        true_val = 3.0
        for _ in range(50):
            pf.predict()
            pf.update(np.array([true_val]))
        assert pytest.approx(pf.x[0], abs=0.5) == true_val

    def test_pf_weighted_mean(self):
        pf = ParticleFilter(
            f=self._pf_f, h=self._pf_h,
            Q=np.array([[0.1]]), R=np.array([[0.5]]),
            n_particles=100,
            x0=np.array([5.0]), P0=np.array([[0.01]]),
            seed=42,
        )
        # Initially, mean should be near x0
        assert pytest.approx(pf.x[0], abs=0.5) == 5.0

    def test_pf_run_returns_particles(self):
        pf = ParticleFilter(
            f=self._pf_f, h=self._pf_h,
            Q=np.array([[0.1]]), R=np.array([[0.5]]),
            n_particles=50,
            x0=np.array([0.0]), P0=np.array([[1.0]]),
            seed=42,
        )
        measurements = [np.array([float(i) * 0.1]) for i in range(10)]
        results = pf.run(measurements)
        assert len(results["x_estimates"]) == 10
        assert len(results["particles_history"]) == 10
        assert results["particles_history"][0].shape == (50, 1)

    def test_pf_resampling_happens(self):
        """After many updates, weights should be uniform (resampling occurred)."""
        pf = ParticleFilter(
            f=self._pf_f, h=self._pf_h,
            Q=np.array([[0.1]]), R=np.array([[0.5]]),
            n_particles=100,
            x0=np.array([0.0]), P0=np.array([[1.0]]),
            resample_threshold=0.5,
            seed=42,
        )
        for _ in range(20):
            pf.predict()
            pf.update(np.array([1.0]))
        # After resampling, weights should be approximately uniform
        assert np.std(pf.weights) < 0.05
