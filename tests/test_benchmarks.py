"""Tests for the benchmark engine."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.metrics import position_rmse, position_mae, per_step_errors, nees
from benchmarks.configs import (
    make_kf, make_ekf, make_ukf, make_pf, make_all_filters,
    make_cv_transition, make_cv_jacobian, FILTER_NAMES,
)
from benchmarks.runner import run_single_trajectory, run_corpus
from benchmarks.sweep import sweep_turn_rate
from benchmarks.corpus import generate_synthetic_corpus


# ── Helpers ────────────────────────────────────────────────────────────────

def _perfect_trajectory(n_steps=20, dt=0.5):
    """Create a simple trajectory where perfect filter gets zero error."""
    true_states = np.zeros((n_steps + 1, 4))
    true_states[0] = [0, 0, 1, 0.5]
    for k in range(n_steps):
        true_states[k + 1, 0] = true_states[k, 0] + true_states[k, 2] * dt
        true_states[k + 1, 1] = true_states[k, 1] + true_states[k, 3] * dt
        true_states[k + 1, 2] = true_states[k, 2]
        true_states[k + 1, 3] = true_states[k, 3]
    measurements = true_states[1:, :2].copy()
    return {"true_states": true_states, "measurements": measurements, "dt": dt}


# ── TestMetrics ────────────────────────────────────────────────────────────

class TestMetrics:
    def test_rmse_zero_for_perfect(self):
        traj = _perfect_trajectory()
        # x_estimates = true positions
        x_estimates = [traj["true_states"][k + 1] for k in range(len(traj["measurements"]))]
        assert position_rmse(x_estimates, traj["true_states"]) == pytest.approx(0.0)

    def test_mae_zero_for_perfect(self):
        traj = _perfect_trajectory()
        x_estimates = [traj["true_states"][k + 1] for k in range(len(traj["measurements"]))]
        assert position_mae(x_estimates, traj["true_states"]) == pytest.approx(0.0)

    def test_mae_leq_rmse(self):
        traj = _perfect_trajectory()
        # Add some noise to estimates
        rng = np.random.default_rng(42)
        x_estimates = [
            traj["true_states"][k + 1] + rng.normal(0, 0.5, size=4)
            for k in range(len(traj["measurements"]))
        ]
        rmse = position_rmse(x_estimates, traj["true_states"])
        mae = position_mae(x_estimates, traj["true_states"])
        assert mae <= rmse + 1e-10

    def test_per_step_errors_shape(self):
        traj = _perfect_trajectory(n_steps=30)
        x_estimates = [traj["true_states"][k + 1] for k in range(30)]
        errors = per_step_errors(x_estimates, traj["true_states"])
        assert errors.shape == (30,)

    def test_nees_shape(self):
        traj = _perfect_trajectory(n_steps=10)
        x_estimates = [traj["true_states"][k + 1] for k in range(10)]
        P_estimates = [0.1 * np.eye(4) for _ in range(10)]
        vals = nees(x_estimates, P_estimates, traj["true_states"])
        assert vals.shape == (10,)


# ── TestConfigs ────────────────────────────────────────────────────────────

class TestConfigs:
    def test_make_kf_type(self):
        from filters.kalman import KalmanFilter
        kf = make_kf(dt=0.5, x0=np.zeros(4))
        assert isinstance(kf, KalmanFilter)

    def test_make_ekf_type(self):
        from filters.ekf import ExtendedKalmanFilter
        ekf = make_ekf(dt=0.5, x0=np.zeros(4))
        assert isinstance(ekf, ExtendedKalmanFilter)

    def test_make_ukf_type(self):
        from filters.ukf import UnscentedKalmanFilter
        ukf = make_ukf(dt=0.5, x0=np.zeros(4))
        assert isinstance(ukf, UnscentedKalmanFilter)

    def test_make_pf_type(self):
        from filters.particle import ParticleFilter
        pf = make_pf(dt=0.5, x0=np.zeros(4))
        assert isinstance(pf, ParticleFilter)

    def test_make_all_filters_keys(self):
        filters = make_all_filters(dt=0.5, x0=np.zeros(4))
        assert set(filters.keys()) == set(FILTER_NAMES)

    def test_cv_transition_matches_matrix(self):
        """CV callable should give same result as F @ x."""
        dt = 0.5
        f = make_cv_transition(dt)
        F_jac = make_cv_jacobian(dt)
        x = np.array([1.0, 2.0, 0.5, -0.3])
        F = F_jac(x, None)
        np.testing.assert_array_almost_equal(f(x, None), F @ x)

    def test_cv_reduces_to_identity_at_zero_dt(self):
        f = make_cv_transition(0.0)
        x = np.array([1.0, 2.0, 0.5, -0.3])
        np.testing.assert_array_almost_equal(f(x, None), x)

    def test_jacobian_finite_diff(self):
        """Finite difference check on Jacobian."""
        dt = 0.5
        f = make_cv_transition(dt)
        F_jac = make_cv_jacobian(dt)
        x = np.array([1.0, 2.0, 0.5, -0.3])
        F_analytic = F_jac(x, None)
        eps = 1e-6
        F_numeric = np.zeros((4, 4))
        for i in range(4):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            F_numeric[:, i] = (f(x_plus, None) - f(x_minus, None)) / (2 * eps)
        np.testing.assert_array_almost_equal(F_analytic, F_numeric, decimal=5)


# ── TestRunner ─────────────────────────────────────────────────────────────

class TestRunner:
    def test_single_trajectory(self):
        traj = _perfect_trajectory(n_steps=30)
        results = run_single_trajectory(traj)
        assert set(results.keys()) == set(FILTER_NAMES)
        for name in FILTER_NAMES:
            assert "rmse" in results[name]
            assert "mae" in results[name]
            assert results[name]["rmse"] >= 0

    def test_corpus_summary(self):
        corpus = [_perfect_trajectory(n_steps=20) for _ in range(5)]
        results = run_corpus(corpus)
        assert "per_trajectory" in results
        assert "summary" in results
        assert len(results["per_trajectory"]) == 5
        for name in FILTER_NAMES:
            assert name in results["summary"]
            assert results["summary"][name]["n_trajectories"] == 5


# ── TestSweep ──────────────────────────────────────────────────────────────

class TestSweep:
    def test_sweep_shapes(self):
        result = sweep_turn_rate(
            turn_rates=np.array([0.0, 0.1, 0.2]),
            n_trials_per_rate=3,
            n_steps=20,
        )
        assert result["rmse"].shape == (3, 4, 3)  # 3 rates, 4 filters, 3 trials
        assert result["mean_rmse"].shape == (3, 4)
        assert result["std_rmse"].shape == (3, 4)

    def test_sweep_reproducible(self):
        r1 = sweep_turn_rate(
            turn_rates=np.array([0.0, 0.2]),
            n_trials_per_rate=2, n_steps=15, base_seed=42,
        )
        r2 = sweep_turn_rate(
            turn_rates=np.array([0.0, 0.2]),
            n_trials_per_rate=2, n_steps=15, base_seed=42,
        )
        np.testing.assert_array_almost_equal(r1["mean_rmse"], r2["mean_rmse"])

    def test_kf_degrades_with_turn_rate(self):
        result = sweep_turn_rate(
            turn_rates=np.array([0.0, 0.3, 0.5]),
            n_trials_per_rate=10,
            n_steps=40,
        )
        kf_idx = list(result["filter_names"]).index("KF")
        kf_rmse = result["mean_rmse"][:, kf_idx]
        # KF should get worse as turn rate increases
        assert kf_rmse[-1] > kf_rmse[0]


# ── TestCorpus ─────────────────────────────────────────────────────────────

class TestCorpus:
    def test_synthetic_corpus_length(self):
        corpus = generate_synthetic_corpus(n_per_regime=5, base_seed=42)
        assert len(corpus) == 20  # 4 regimes × 5

    def test_synthetic_corpus_regimes(self):
        corpus = generate_synthetic_corpus(n_per_regime=3, base_seed=42)
        regimes = {t["regime"] for t in corpus}
        assert regimes == {"linear", "pedestrian", "nonlinear", "sharp_turn"}

    def test_synthetic_corpus_has_required_keys(self):
        corpus = generate_synthetic_corpus(n_per_regime=2, base_seed=42)
        for traj in corpus:
            assert "true_states" in traj
            assert "measurements" in traj
            assert "dt" in traj
            assert "regime" in traj
