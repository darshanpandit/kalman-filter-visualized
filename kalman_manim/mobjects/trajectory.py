"""PedestrianPath — animated trajectory visualization on a minimalist map."""

from __future__ import annotations

from manim import *
import numpy as np

from kalman_manim.style import (
    COLOR_TRUE_PATH,
    COLOR_MEASUREMENT,
    COLOR_POSTERIOR,
    MEASUREMENT_DOT_RADIUS,
    DOT_RADIUS_SMALL,
)


class PedestrianPath(VGroup):
    """Renders a pedestrian trajectory with true path, measurements, and filtered estimate.

    Parameters
    ----------
    true_positions : np.ndarray (N, 2)
        Ground truth [x, y] positions.
    measurements : np.ndarray (M, 2) or None
        Noisy measurement positions.
    estimates : np.ndarray (M, 2) or None
        Kalman-filtered position estimates.
    axes : Axes or None
        Coordinate system for conversion. If None, uses raw coords.
    true_color : str
        Color for the true path.
    meas_color : str
        Color for measurement dots.
    est_color : str
        Color for the filtered estimate path.
    """

    def __init__(
        self,
        true_positions: np.ndarray,
        measurements: np.ndarray | None = None,
        estimates: np.ndarray | None = None,
        axes=None,
        true_color: str = COLOR_TRUE_PATH,
        meas_color: str = COLOR_MEASUREMENT,
        est_color: str = COLOR_POSTERIOR,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._axes = axes

        def to_point(xy):
            if axes is not None:
                return axes.c2p(xy[0], xy[1])
            return np.array([xy[0], xy[1], 0])

        # True path — dashed white line
        if len(true_positions) >= 2:
            true_points = [to_point(p) for p in true_positions]
            self.true_path = DashedVMobject(
                VMobject().set_points_smoothly(true_points),
                num_dashes=len(true_points) * 2,
            )
            self.true_path.set_color(true_color)
            self.true_path.set_stroke(width=1.5, opacity=0.7)
            self.add(self.true_path)

        # Measurement dots
        if measurements is not None:
            self.meas_dots = VGroup(*[
                Dot(to_point(m), radius=MEASUREMENT_DOT_RADIUS, color=meas_color,
                    fill_opacity=0.7)
                for m in measurements
            ])
            self.add(self.meas_dots)

        # Filtered estimate path — solid gold line
        if estimates is not None and len(estimates) >= 2:
            est_points = [to_point(p) for p in estimates]
            self.est_path = VMobject()
            self.est_path.set_points_smoothly(est_points)
            self.est_path.set_color(est_color)
            self.est_path.set_stroke(width=3)
            self.add(self.est_path)
