"""Tests for models/ layer (NW estimator, precomputed loaders)."""

from __future__ import annotations

import numpy as np
import pytest

from models.transformer_kf import NWKalmanEstimator
from models.kalmannet_stub import (
    load_kalmannet_results,
    load_icl_results,
    load_scaling_results,
)


# ── NWKalmanEstimator ────────────────────────────────────────────────────


class TestNWKalmanEstimator:
    def test_output_shape_1d(self):
        """1D observations produce 1D estimates of same length."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        nw = NWKalmanEstimator(bandwidth=1.0)
        est = nw.estimate(obs)
        assert est.shape == obs.shape

    def test_output_shape_2d(self):
        """2D observations produce 2D estimates of same shape."""
        obs = np.random.default_rng(42).normal(size=(20, 2))
        nw = NWKalmanEstimator(bandwidth=1.0)
        est = nw.estimate(obs)
        assert est.shape == obs.shape

    def test_first_estimate_equals_first_observation(self):
        """With no history, estimate[0] = observation[0]."""
        obs = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        nw = NWKalmanEstimator(bandwidth=1.0)
        est = nw.estimate(obs)
        assert est[0] == obs[0]

    def test_constant_observations_converge(self):
        """Constant observations → estimate converges to that constant."""
        obs = np.full(50, 7.0)
        nw = NWKalmanEstimator(bandwidth=1.0)
        est = nw.estimate(obs)
        np.testing.assert_allclose(est[-1], 7.0, atol=1e-6)

    def test_attention_weights_shape(self):
        """Attention matrix is (T, T) lower-triangular."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        nw = NWKalmanEstimator(bandwidth=1.0)
        W = nw.attention_weights(obs)
        assert W.shape == (4, 4)
        # Upper triangle (above diagonal) should be zero
        for i in range(4):
            for j in range(i + 1, 4):
                assert W[i, j] == 0.0

    def test_attention_weights_sum_to_one(self):
        """Each row of attention weights sums to 1 (or 0 for future)."""
        obs = np.random.default_rng(99).normal(size=10)
        nw = NWKalmanEstimator(bandwidth=0.5)
        W = nw.attention_weights(obs)
        for t in range(len(obs)):
            row_sum = np.sum(W[t, :])
            assert pytest.approx(row_sum, abs=1e-10) == 1.0

    def test_narrow_bandwidth_focuses_attention(self):
        """Small bandwidth → attention concentrates on nearest neighbor."""
        obs = np.array([0.0, 10.0, 0.1, 0.2, 0.15])
        nw = NWKalmanEstimator(bandwidth=0.01)
        W = nw.attention_weights(obs)
        # At t=4 (obs=0.15), nearest past obs is t=3 (obs=0.2) or t=2 (obs=0.1)
        # One of them should dominate
        assert np.max(W[4, :4]) > 0.5

    def test_tracks_linear_trend(self):
        """NW estimator should roughly follow a linear trend."""
        obs = np.arange(20, dtype=float) + np.random.default_rng(42).normal(0, 0.3, 20)
        nw = NWKalmanEstimator(bandwidth=2.0)
        est = nw.estimate(obs)
        # End estimate should be near the trend value
        assert abs(est[-1] - obs[-1]) < 5.0


# ── Precomputed Loaders ──────────────────────────────────────────────────


class TestKalmanNetResults:
    def test_load_returns_required_keys(self):
        results = load_kalmannet_results()
        assert "methods" in results
        assert "pos_rmse" in results
        assert "vel_rmse" in results

    def test_values_match_paper(self):
        results = load_kalmannet_results()
        methods = list(results["methods"])
        assert "KalmanNet" in methods
        assert "IMM" in methods
        # KalmanNet has higher pos RMSE than IMM (Mehrfard et al. finding)
        kn_idx = methods.index("KalmanNet")
        imm_idx = methods.index("IMM")
        assert results["pos_rmse"][kn_idx] > results["pos_rmse"][imm_idx]


class TestICLResults:
    def test_load_returns_required_keys(self):
        results = load_icl_results()
        assert "systems" in results
        assert "kf_mse" in results
        assert "tf_mse" in results

    def test_transformer_beats_kf_on_nonlinear(self):
        """Balim et al.: TF outperforms KF/EKF on quadrotor."""
        results = load_icl_results()
        systems = list(results["systems"])
        quad_idx = systems.index("quadrotor")
        assert results["tf_mse"][quad_idx] < results["kf_mse"][quad_idx]

    def test_burn_in_curves_decrease(self):
        """Per-step MSE curves should decrease over time."""
        results = load_icl_results()
        kf_curve = results["burn_in_kf_mse"]
        assert kf_curve[0] > kf_curve[-1]


class TestScalingResults:
    def test_load_returns_required_keys(self):
        results = load_scaling_results()
        assert "layers" in results
        assert "vs_ekf" in results
        assert "vs_pf" in results

    def test_more_layers_improves(self):
        """More transformer layers → lower MSPD (closer to optimal)."""
        results = load_scaling_results()
        assert results["vs_ekf"][0] > results["vs_ekf"][-1]
        assert results["vs_pf"][0] > results["vs_pf"][-1]
