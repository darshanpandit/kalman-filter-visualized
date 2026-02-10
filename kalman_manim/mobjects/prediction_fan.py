"""Prediction fan mobject for trajectory prediction visualization.

Shows a fan of future trajectory samples diverging from a common point,
used in Part 7 Scene 4 for social prediction demos.
"""

from __future__ import annotations

import numpy as np
from manim import *

from kalman_manim.style import SLATE, COLOR_SOCIAL


class PredictionFan(VGroup):
    """Fan of future trajectory predictions from a branching point.

    Parameters
    ----------
    origin : np.ndarray, shape (2,)
        Branch point in scene coordinates.
    trajectories : list of np.ndarray, each (T, 2)
        Predicted future trajectories in scene coordinates.
    color : str
    opacity : float
        Line opacity for individual predictions.
    """

    def __init__(
        self,
        origin: np.ndarray,
        trajectories: list[np.ndarray],
        color: str = COLOR_SOCIAL,
        opacity: float = 0.3,
        stroke_width: float = 1.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.origin_pt = np.array(origin)

        self.fan_lines = VGroup()
        for traj in trajectories:
            points = [np.array([*self.origin_pt, 0])]
            for pt in traj:
                points.append(np.array([pt[0], pt[1], 0]))
            if len(points) >= 2:
                line = VMobject()
                line.set_points_smoothly(points)
                line.set_color(color).set_stroke(width=stroke_width, opacity=opacity)
                self.fan_lines.add(line)

        # Origin dot
        self.origin_dot = Dot(
            np.array([*self.origin_pt, 0]),
            radius=0.06, color=color,
        )

        self.add(self.fan_lines, self.origin_dot)

    def animate_fan(self, run_time: float = 1.5) -> list:
        """Return animations to draw the fan from the origin."""
        return [
            FadeIn(self.origin_dot, run_time=0.3),
            *[Create(line, run_time=run_time) for line in self.fan_lines],
        ]
