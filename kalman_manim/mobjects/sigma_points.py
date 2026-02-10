"""SigmaPointCloud — visualizes UKF sigma points around an ellipse."""

from __future__ import annotations

from manim import *
import numpy as np
from scipy.linalg import cholesky

from kalman_manim.style import COLOR_HIGHLIGHT, COLOR_PROCESS_NOISE, DOT_RADIUS_MEDIUM


class SigmaPointCloud(VGroup):
    """Renders sigma points for the Unscented Kalman Filter.

    Places 2n+1 weighted dots symmetrically around the mean,
    positioned on the covariance ellipse boundary.

    Parameters
    ----------
    mean : np.ndarray
        2D mean [x, y].
    cov : np.ndarray
        2x2 covariance matrix.
    alpha : float
        UKF spread parameter.
    kappa : float
        UKF secondary scaling parameter.
    color : str
        Color for sigma point dots.
    center_color : str
        Color for the center (mean) sigma point.
    axes : Axes or None
        Coordinate system for conversion.
    dot_radius : float
        Radius of sigma point dots.
    """

    def __init__(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        alpha: float = 0.1,
        kappa: float = 0.0,
        color: str = COLOR_HIGHLIGHT,
        center_color: str = COLOR_PROCESS_NOISE,
        axes=None,
        dot_radius: float = DOT_RADIUS_MEDIUM,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mean = np.array(mean, dtype=float)
        self._cov = np.array(cov, dtype=float)
        self._axes = axes
        n = len(mean)

        lam = alpha**2 * (n + kappa) - n

        # Compute sigma points
        try:
            S = cholesky((n + lam) * cov, lower=True)
        except np.linalg.LinAlgError:
            S = cholesky((n + lam) * cov + 1e-6 * np.eye(n), lower=True)

        sigma_points = [mean.copy()]
        for i in range(n):
            sigma_points.append(mean + S[:, i])
            sigma_points.append(mean - S[:, i])

        self.sigma_points = np.array(sigma_points)

        # Compute weights for sizing dots
        Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        Wm[0] = lam / (n + lam)
        self.weights = Wm

        # Create dots
        self.dots = VGroup()
        for i, sp in enumerate(self.sigma_points):
            pos = self._to_scene(sp)
            is_center = (i == 0)
            # Scale dot size by weight (center point gets special treatment)
            r = dot_radius * (1.5 if is_center else 1.0)
            c = center_color if is_center else color
            dot = Dot(pos, radius=r, color=c, fill_opacity=0.9)
            self.dots.add(dot)

        self.add(self.dots)

    def _to_scene(self, xy):
        if self._axes is not None:
            return self._axes.c2p(xy[0], xy[1])
        return np.array([xy[0], xy[1], 0])

    def get_transformed_cloud(self, func, color: str = COLOR_HIGHLIGHT,
                               axes=None):
        """Transform sigma points through a nonlinear function.

        Returns a new SigmaPointCloud-like VGroup at the transformed locations.
        """
        transformed = VGroup()
        for i, sp in enumerate(self.sigma_points):
            new_pos = func(sp)
            pos = axes.c2p(new_pos[0], new_pos[1]) if axes else np.array([new_pos[0], new_pos[1], 0])
            is_center = (i == 0)
            r = DOT_RADIUS_MEDIUM * (1.5 if is_center else 1.0)
            dot = Dot(pos, radius=r, color=color, fill_opacity=0.9)
            transformed.add(dot)
        return transformed
