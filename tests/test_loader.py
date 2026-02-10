"""Tests for ETH pedestrian data loader."""

from __future__ import annotations

import numpy as np
import pytest

from kalman_manim.data.loader import list_available_trajectories, load_eth_trajectory


class TestListAvailable:
    def test_returns_list_of_dicts(self):
        result = list_available_trajectories("hotel")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all("pedestrian_id" in d for d in result)
        assert all("n_steps" in d for d in result)

    def test_min_steps_filter(self):
        long = list_available_trajectories("hotel", min_steps=40)
        short = list_available_trajectories("hotel", min_steps=10)
        assert len(short) >= len(long)

    def test_eth_sequence(self):
        result = list_available_trajectories("eth")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_sorted_by_length(self):
        result = list_available_trajectories("hotel", min_steps=10)
        for i in range(len(result) - 1):
            assert result[i]["n_steps"] >= result[i + 1]["n_steps"]


class TestLoadTrajectory:
    def test_output_format_matches_generators(self):
        data = load_eth_trajectory("hotel", measurement_noise_std=0.5, seed=42)
        assert "true_states" in data
        assert "measurements" in data
        assert "dt" in data
        assert isinstance(data["dt"], float)
        assert data["dt"] > 0

    def test_true_states_shape(self):
        data = load_eth_trajectory("hotel", seed=42)
        ts = data["true_states"]
        m = data["measurements"]
        assert ts.ndim == 2
        assert ts.shape[1] == 4  # [x, y, vx, vy]
        assert m.ndim == 2
        assert m.shape[1] == 2  # [x, y]
        assert ts.shape[0] == m.shape[0] + 1  # n+1 states, n measurements

    def test_specific_pedestrian(self):
        available = list_available_trajectories("hotel", min_steps=10)
        pid = available[0]["pedestrian_id"]
        data = load_eth_trajectory("hotel", pedestrian_id=pid, seed=42)
        assert data["metadata"]["pedestrian_id"] == pid

    def test_max_steps(self):
        data = load_eth_trajectory("hotel", max_steps=20, seed=42)
        assert data["measurements"].shape[0] == 20

    def test_reproducibility(self):
        d1 = load_eth_trajectory("hotel", measurement_noise_std=0.5, seed=99)
        d2 = load_eth_trajectory("hotel", measurement_noise_std=0.5, seed=99)
        np.testing.assert_array_equal(d1["measurements"], d2["measurements"])

    def test_different_seeds_give_different_noise(self):
        d1 = load_eth_trajectory("hotel", measurement_noise_std=0.5, seed=1)
        d2 = load_eth_trajectory("hotel", measurement_noise_std=0.5, seed=2)
        assert not np.allclose(d1["measurements"], d2["measurements"])

    def test_zero_noise(self):
        data = load_eth_trajectory("hotel", measurement_noise_std=0.0, seed=42)
        ts = data["true_states"]
        m = data["measurements"]
        # Measurements should exactly equal true positions[1:]
        np.testing.assert_array_almost_equal(m, ts[1:, :2])

    def test_metadata_present(self):
        data = load_eth_trajectory("hotel", seed=42)
        assert "metadata" in data
        assert data["metadata"]["sequence"] == "hotel"
        assert "source" in data["metadata"]

    def test_velocities_reasonable(self):
        data = load_eth_trajectory("hotel", seed=42)
        vels = data["true_states"][:, 2:]
        speeds = np.linalg.norm(vels, axis=1)
        # Pedestrian speeds should be < 10 m/s
        assert np.all(speeds < 10.0)

    def test_invalid_sequence(self):
        with pytest.raises(FileNotFoundError):
            load_eth_trajectory("nonexistent")

    def test_invalid_pedestrian(self):
        with pytest.raises(ValueError):
            load_eth_trajectory("hotel", pedestrian_id=999999)


class TestCuratedGenerators:
    def test_sharp_turn_trajectory(self):
        from kalman_manim.data.generators import generate_sharp_turn_trajectory
        data = generate_sharp_turn_trajectory(seed=42)
        assert "true_states" in data
        assert "measurements" in data
        assert data["true_states"].shape[1] == 4
        # Should have clear directional changes
        vels = data["true_states"][:, 2:]
        headings = np.arctan2(vels[:, 1], vels[:, 0])
        heading_changes = np.abs(np.diff(headings))
        # Wrap to [-pi, pi]
        heading_changes = np.minimum(heading_changes, 2 * np.pi - heading_changes)
        # Should have at least one sharp turn (>45 degrees)
        assert np.any(heading_changes > np.pi / 4)

    def test_multimodal_scenario(self):
        from kalman_manim.data.generators import generate_multimodal_scenario
        data = generate_multimodal_scenario(seed=42)
        assert "true_states" in data
        assert "measurements" in data
        assert data["true_states"].shape[1] == 4
