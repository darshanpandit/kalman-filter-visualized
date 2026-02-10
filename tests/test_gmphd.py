"""Tests for GM-PHD filter implementation."""

from __future__ import annotations

import numpy as np
import pytest

from filters.gmphd import GMPHDFilter, GaussianComponent
from kalman_manim.data.generators import generate_multi_target_scenario


class TestGMPHDFilter:
    def _make_phd(self, birth_pos=None):
        """Create a GM-PHD filter with constant velocity model."""
        dt = 0.5
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                       [0, 0, 1, 0], [0, 0, 0, 1]])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        Q = 0.1 * np.eye(4)
        R = 0.5 * np.eye(2)

        birth = []
        if birth_pos is not None:
            for pos in birth_pos:
                birth.append(GaussianComponent(
                    weight=0.1,
                    mean=np.array([pos[0], pos[1], 0, 0]),
                    cov=np.eye(4),
                ))

        return GMPHDFilter(
            F=F, H=H, Q=Q, R=R,
            ps=0.99, pd=0.95,
            clutter_intensity=1e-4,
            birth_components=birth,
            merge_threshold=4.0,
            prune_threshold=1e-5,
        )

    def test_initial_empty(self):
        """PHD starts with no components."""
        phd = self._make_phd()
        assert len(phd.components) == 0
        assert phd.estimated_cardinality == 0.0

    def test_predict_adds_birth_components(self):
        """After prediction, birth components are added."""
        phd = self._make_phd(birth_pos=[[0, 0], [5, 5]])
        phd.predict()
        assert len(phd.components) == 2

    def test_cardinality_grows_with_measurements(self):
        """Cardinality estimate increases when targets are detected."""
        phd = self._make_phd(birth_pos=[[0, 0]])

        for _ in range(5):
            phd.predict()
            phd.update([np.array([0.1, 0.1])])

        assert phd.estimated_cardinality > 0.3

    def test_extract_states(self):
        """extract_states returns positions for high-weight components."""
        phd = self._make_phd(birth_pos=[[0, 0]])

        # Run several cycles to build up weight
        for t in range(10):
            phd.predict()
            phd.update([np.array([t * 0.2, 0.0])])

        states = phd.extract_states(threshold=0.3)
        # Should have at least one extracted state
        assert len(states) >= 1
        assert states[0].shape == (4,)

    def test_prune_removes_low_weight(self):
        """Pruning removes components below threshold."""
        phd = self._make_phd()
        # Manually add low-weight components
        phd.components = [
            GaussianComponent(1e-6, np.zeros(4), np.eye(4)),
            GaussianComponent(0.5, np.ones(4), np.eye(4)),
        ]
        phd.prune_and_merge()
        assert len(phd.components) == 1
        assert phd.components[0].weight == pytest.approx(0.5)

    def test_merge_nearby_components(self):
        """Merging combines nearby components."""
        phd = self._make_phd()
        # Two very close components
        phd.components = [
            GaussianComponent(0.5, np.array([1.0, 1.0, 0, 0]),
                              0.1 * np.eye(4)),
            GaussianComponent(0.3, np.array([1.01, 1.01, 0, 0]),
                              0.1 * np.eye(4)),
        ]
        phd.prune_and_merge()
        # Should merge into one
        assert len(phd.components) == 1
        assert phd.components[0].weight == pytest.approx(0.8, abs=0.01)

    def test_run_returns_correct_format(self):
        """run() returns cardinality estimates and extracted states."""
        phd = self._make_phd(birth_pos=[[0, 0]])
        measurement_sets = [
            [np.array([t * 0.2, 0.0])] for t in range(10)
        ]
        results = phd.run(measurement_sets)
        assert len(results["cardinality_estimates"]) == 10
        assert len(results["extracted_states"]) == 10


class TestMultiTargetGenerator:
    def test_output_format(self):
        data = generate_multi_target_scenario(n_steps=30, seed=42)
        assert "true_tracks" in data
        assert "measurement_sets" in data
        assert "true_cardinality" in data
        assert data["true_cardinality"].shape == (30,)

    def test_initial_cardinality(self):
        data = generate_multi_target_scenario(
            n_steps=10, n_targets_init=3, birth_step=50, death_step=50,
            seed=42,
        )
        assert data["true_cardinality"][0] == 3

    def test_birth_increases_cardinality(self):
        data = generate_multi_target_scenario(
            n_steps=30, n_targets_init=2, birth_step=10, death_step=50,
            seed=42,
        )
        assert data["true_cardinality"][5] == 2
        assert data["true_cardinality"][15] == 3

    def test_death_decreases_cardinality(self):
        data = generate_multi_target_scenario(
            n_steps=50, n_targets_init=3, birth_step=5, death_step=30,
            seed=42,
        )
        assert data["true_cardinality"][35] < data["true_cardinality"][25]

    def test_measurements_vary_in_length(self):
        """Each step may have different number of measurements (clutter)."""
        data = generate_multi_target_scenario(
            n_steps=20, clutter_rate=2.0, seed=42,
        )
        lengths = [len(ms) for ms in data["measurement_sets"]]
        # With clutter, lengths should vary
        assert max(lengths) > min(lengths)

    def test_reproducibility(self):
        d1 = generate_multi_target_scenario(n_steps=10, seed=123)
        d2 = generate_multi_target_scenario(n_steps=10, seed=123)
        np.testing.assert_array_equal(
            d1["true_cardinality"], d2["true_cardinality"],
        )
