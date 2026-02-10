"""GaussianEllipse — a VGroup that renders a 2D Gaussian as a filled ellipse."""

from __future__ import annotations

from manim import *
import numpy as np

from kalman_manim.utils import cov_to_ellipse_params
from kalman_manim.style import (
    ELLIPSE_FILL_OPACITY,
    ELLIPSE_STROKE_WIDTH,
    ELLIPSE_N_SIGMA,
    DOT_RADIUS_SMALL,
)


class GaussianEllipse(VGroup):
    """Visualizes a 2D Gaussian distribution as a semi-transparent ellipse.

    Parameters
    ----------
    mean : np.ndarray
        2D mean vector [x, y].
    cov : np.ndarray
        2x2 covariance matrix.
    color : str
        Color for the ellipse stroke and fill.
    fill_opacity : float
        Opacity of the ellipse fill.
    stroke_width : float
        Width of the ellipse border.
    n_sigma : float
        Number of standard deviations for the boundary.
    show_center : bool
        Whether to draw a dot at the center.
    show_axes : bool
        Whether to draw eigenvector axis lines.
    axes : Axes | None
        If provided, use axes.c2p() for coordinate conversion.
        If None, mean values are used directly as scene coordinates.
    label : str | None
        Optional label text placed next to the ellipse.
    """

    def __init__(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        color: str = WHITE,
        fill_opacity: float = ELLIPSE_FILL_OPACITY,
        stroke_width: float = ELLIPSE_STROKE_WIDTH,
        n_sigma: float = ELLIPSE_N_SIGMA,
        show_center: bool = True,
        show_axes: bool = False,
        axes: Axes | None = None,
        label: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mean = np.array(mean, dtype=float)
        self._cov = np.array(cov, dtype=float)
        self._color = color
        self._fill_opacity = fill_opacity
        self._stroke_width = stroke_width
        self._n_sigma = n_sigma
        self._axes = axes
        self._show_center = show_center
        self._show_axes = show_axes

        # Build the visual components
        self._build(label)

    def _build(self, label: str | None):
        params = cov_to_ellipse_params(self._cov, self._n_sigma)

        # Scale dimensions if using axes (convert data units to scene units)
        if self._axes is not None:
            # Approximate scale: difference in scene coords for a unit data step
            origin = self._axes.c2p(0, 0)
            unit_x = self._axes.c2p(1, 0)
            unit_y = self._axes.c2p(0, 1)
            sx = np.linalg.norm(np.array(unit_x) - np.array(origin))
            sy = np.linalg.norm(np.array(unit_y) - np.array(origin))
            width = params["width"] * sx
            height = params["height"] * sy
            center = self._axes.c2p(self._mean[0], self._mean[1])
        else:
            width = params["width"]
            height = params["height"]
            center = np.array([self._mean[0], self._mean[1], 0])

        # Ellipse
        self.ellipse = Ellipse(
            width=width,
            height=height,
            color=self._color,
            fill_color=self._color,
            fill_opacity=self._fill_opacity,
            stroke_width=self._stroke_width,
        )
        self.ellipse.rotate(params["angle"])
        self.ellipse.move_to(center)
        self.add(self.ellipse)

        # Center dot
        if self._show_center:
            self.center_dot = Dot(
                center,
                radius=DOT_RADIUS_SMALL,
                color=self._color,
            )
            self.add(self.center_dot)

        # Eigenvector axes
        if self._show_axes:
            eigenvalues, eigenvectors = np.linalg.eigh(self._cov)
            eigenvalues = np.maximum(eigenvalues, 0)
            for i in range(2):
                half_len = self._n_sigma * np.sqrt(eigenvalues[i])
                if self._axes is not None:
                    half_len *= sx if i == 0 else sy  # rough scaling
                direction = np.array([eigenvectors[0, i], eigenvectors[1, i], 0])
                direction = direction / np.linalg.norm(direction) * half_len
                axis_line = DashedLine(
                    center - direction,
                    center + direction,
                    color=self._color,
                    stroke_width=1.5,
                    stroke_opacity=0.6,
                )
                self.add(axis_line)

        # Label
        if label is not None:
            self.label = MathTex(label, color=self._color, font_size=24)
            self.label.next_to(self.ellipse, UR, buff=0.15)
            self.add(self.label)

    def animate_to(self, new_mean: np.ndarray, new_cov: np.ndarray, **kwargs):
        """Return an animation group that morphs this ellipse to a new Gaussian.

        Usage: self.play(gaussian.animate_to(new_mu, new_P))
        """
        target = GaussianEllipse(
            mean=new_mean,
            cov=new_cov,
            color=self._color,
            fill_opacity=self._fill_opacity,
            stroke_width=self._stroke_width,
            n_sigma=self._n_sigma,
            show_center=self._show_center,
            show_axes=self._show_axes,
            axes=self._axes,
        )
        self._mean = np.array(new_mean, dtype=float)
        self._cov = np.array(new_cov, dtype=float)
        return Transform(self, target, **kwargs)
