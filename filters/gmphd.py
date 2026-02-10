"""Gaussian-Mixture PHD (Probability Hypothesis Density) filter.

Pure numpy implementation based on Vo & Ma (2006):
    Vo, B.-N. and Ma, W.-K. (2006) "The Gaussian Mixture Probability
    Hypothesis Density Filter." IEEE Trans. Signal Processing, 54(11).

Tracks a variable number of targets without explicit data association.
The PHD (intensity function) is represented as a mixture of Gaussians.
"""

from __future__ import annotations

import numpy as np


class GaussianComponent:
    """Single Gaussian component in the PHD mixture."""

    __slots__ = ("weight", "mean", "cov")

    def __init__(self, weight: float, mean: np.ndarray, cov: np.ndarray):
        self.weight = weight
        self.mean = np.array(mean, dtype=float)
        self.cov = np.array(cov, dtype=float)


class GMPHDFilter:
    """Gaussian-Mixture PHD filter for multi-target tracking.

    Parameters
    ----------
    F : np.ndarray
        State transition matrix (n x n).
    H : np.ndarray
        Measurement matrix (m x n).
    Q : np.ndarray
        Process noise covariance (n x n).
    R : np.ndarray
        Measurement noise covariance (m x m).
    ps : float
        Survival probability. Default 0.99.
    pd : float
        Detection probability. Default 0.98.
    clutter_intensity : float
        Expected clutter intensity (false alarms per unit volume).
    birth_components : list of GaussianComponent
        Components added at each prediction step (target birth model).
    merge_threshold : float
        Mahalanobis distance threshold for merging. Default 4.0.
    prune_threshold : float
        Minimum weight to keep a component. Default 1e-5.
    max_components : int
        Maximum number of components after pruning. Default 100.
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        ps: float = 0.99,
        pd: float = 0.98,
        clutter_intensity: float = 1e-5,
        birth_components: list | None = None,
        merge_threshold: float = 4.0,
        prune_threshold: float = 1e-5,
        max_components: int = 100,
    ):
        self.F = np.array(F, dtype=float)
        self.H = np.array(H, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.n = self.F.shape[0]
        self.m = self.H.shape[0]
        self.ps = ps
        self.pd = pd
        self.clutter_intensity = clutter_intensity
        self.birth_components = birth_components or []
        self.merge_threshold = merge_threshold
        self.prune_threshold = prune_threshold
        self.max_components = max_components

        self.components: list[GaussianComponent] = []

    @property
    def estimated_cardinality(self) -> float:
        """Estimated number of targets (sum of weights)."""
        return sum(c.weight for c in self.components)

    def extract_states(self, threshold: float = 0.5) -> list[np.ndarray]:
        """Extract target states (components with weight > threshold)."""
        states = []
        for c in self.components:
            n_targets = int(round(c.weight))
            if c.weight >= threshold:
                for _ in range(max(1, n_targets)):
                    states.append(c.mean.copy())
        return states

    def predict(self, u=None):
        """PHD prediction: propagate existing + add birth components.

        Returns estimated cardinality after prediction.
        """
        predicted = []

        # Surviving targets
        for c in self.components:
            new_mean = self.F @ c.mean
            new_cov = self.F @ c.cov @ self.F.T + self.Q
            predicted.append(GaussianComponent(
                weight=self.ps * c.weight,
                mean=new_mean,
                cov=new_cov,
            ))

        # Birth targets
        for bc in self.birth_components:
            predicted.append(GaussianComponent(
                weight=bc.weight,
                mean=bc.mean.copy(),
                cov=bc.cov.copy(),
            ))

        self.components = predicted
        return self.estimated_cardinality

    def update(self, measurements: list[np.ndarray]):
        """PHD update: reweight components using measurements.

        Parameters
        ----------
        measurements : list of np.ndarray
            Each element is a measurement vector of dimension m.

        Returns estimated cardinality after update.
        """
        if not self.components:
            return 0.0

        # Misdetection term: components that were not detected
        updated = []
        for c in self.components:
            updated.append(GaussianComponent(
                weight=(1 - self.pd) * c.weight,
                mean=c.mean.copy(),
                cov=c.cov.copy(),
            ))

        # Detection + measurement update
        for z in measurements:
            z = np.array(z, dtype=float)
            new_components = []
            weight_sum = 0.0

            for c in self.components:
                # Predicted measurement
                z_pred = self.H @ c.mean
                S = self.H @ c.cov @ self.H.T + self.R

                # Kalman gain
                K = c.cov @ self.H.T @ np.linalg.inv(S)

                # Updated state
                innov = z - z_pred
                new_mean = c.mean + K @ innov
                new_cov = (np.eye(self.n) - K @ self.H) @ c.cov

                # Gaussian likelihood
                d = len(innov)
                det_S = max(np.linalg.det(S), 1e-30)
                exponent = -0.5 * innov @ np.linalg.inv(S) @ innov
                q_z = np.exp(exponent) / np.sqrt((2 * np.pi) ** d * det_S)

                new_weight = self.pd * c.weight * q_z
                weight_sum += new_weight

                new_components.append(GaussianComponent(
                    weight=new_weight,
                    mean=new_mean,
                    cov=new_cov,
                ))

            # Normalize by clutter + total detection weight
            denominator = self.clutter_intensity + weight_sum
            if denominator > 0:
                for nc in new_components:
                    nc.weight /= denominator

            updated.extend(new_components)

        self.components = updated
        self.prune_and_merge()
        return self.estimated_cardinality

    def prune_and_merge(self):
        """Prune low-weight components and merge nearby ones."""
        # Prune
        self.components = [
            c for c in self.components if c.weight >= self.prune_threshold
        ]

        if not self.components:
            return

        # Merge nearby components
        merged = []
        used = [False] * len(self.components)

        # Sort by weight (descending) for greedy merging
        indices = sorted(
            range(len(self.components)),
            key=lambda i: self.components[i].weight,
            reverse=True,
        )

        for i in indices:
            if used[i]:
                continue

            # Find all components within merge_threshold of this one
            merge_set = [i]
            for j in indices:
                if j == i or used[j]:
                    continue
                diff = self.components[j].mean - self.components[i].mean
                cov_inv = np.linalg.inv(self.components[i].cov)
                mahal = np.sqrt(diff @ cov_inv @ diff)
                if mahal < self.merge_threshold:
                    merge_set.append(j)

            # Merge
            total_weight = sum(self.components[k].weight for k in merge_set)
            merged_mean = np.zeros(self.n)
            for k in merge_set:
                merged_mean += self.components[k].weight * self.components[k].mean
            merged_mean /= total_weight

            merged_cov = np.zeros((self.n, self.n))
            for k in merge_set:
                diff = self.components[k].mean - merged_mean
                merged_cov += self.components[k].weight * (
                    self.components[k].cov + np.outer(diff, diff)
                )
            merged_cov /= total_weight

            merged.append(GaussianComponent(
                weight=total_weight,
                mean=merged_mean,
                cov=merged_cov,
            ))

            for k in merge_set:
                used[k] = True

        # Cap maximum components (keep highest weight)
        if len(merged) > self.max_components:
            merged.sort(key=lambda c: c.weight, reverse=True)
            merged = merged[:self.max_components]

        self.components = merged

    def run(self, measurement_sets: list[list[np.ndarray]]):
        """Run the GM-PHD filter over a sequence of measurement sets.

        Parameters
        ----------
        measurement_sets : list of list of np.ndarray
            Each element is a set of measurements at one time step.

        Returns
        -------
        dict with keys:
            cardinality_estimates : list of float
            extracted_states : list of list of np.ndarray
        """
        results = {
            "cardinality_estimates": [],
            "extracted_states": [],
        }

        for meas_set in measurement_sets:
            self.predict()
            self.update(meas_set)
            results["cardinality_estimates"].append(self.estimated_cardinality)
            results["extracted_states"].append(self.extract_states())

        return results
