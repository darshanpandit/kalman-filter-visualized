"""Vector field and phase space plot mobjects for Part 9 dynamics scenes."""

from __future__ import annotations

import numpy as np
from manim import *

from kalman_manim.style import CREAM, SLATE, TEAL, COLOR_POSTERIOR


class VectorFieldPlot(VGroup):
    """2D quiver plot showing a vector field.

    Parameters
    ----------
    func : callable(x, y) -> (dx, dy)
        Vector field function.
    x_range, y_range : tuple (min, max)
    n_arrows : int
        Number of arrows per axis.
    width, height : float
    color : str
    """

    def __init__(
        self,
        func,
        x_range: tuple = (-3, 3),
        y_range: tuple = (-3, 3),
        n_arrows: int = 10,
        width: float = 5.0,
        height: float = 5.0,
        color: str = TEAL,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.axes = Axes(
            x_range=[x_range[0], x_range[1], (x_range[1] - x_range[0]) / 5],
            y_range=[y_range[0], y_range[1], (y_range[1] - y_range[0]) / 5],
            x_length=width, y_length=height,
            axis_config={"color": CREAM, "include_tip": False},
        )
        self.add(self.axes)

        xs = np.linspace(x_range[0], x_range[1], n_arrows)
        ys = np.linspace(y_range[0], y_range[1], n_arrows)

        max_mag = 0
        vectors = []
        for x in xs:
            for y in ys:
                dx, dy = func(x, y)
                mag = np.sqrt(dx ** 2 + dy ** 2)
                max_mag = max(max_mag, mag)
                vectors.append((x, y, dx, dy))

        self.arrows_group = VGroup()
        scale = 0.3 / max(max_mag, 1e-6) * (x_range[1] - x_range[0]) / n_arrows
        for x, y, dx, dy in vectors:
            start = self.axes.c2p(x, y)
            end = self.axes.c2p(x + dx * scale, y + dy * scale)
            arrow = Arrow(
                start, end, color=color,
                stroke_width=1.5, buff=0,
                max_tip_length_to_length_ratio=0.3,
            )
            self.arrows_group.add(arrow)

        self.add(self.arrows_group)


class PhaseSpacePlot(VGroup):
    """2D phase space with trajectory and optional energy contours.

    Parameters
    ----------
    trajectory : np.ndarray, shape (T, 2) — [q, p]
    energy_func : callable(q, p) -> E or None
    q_range, p_range : tuple
    width, height : float
    """

    def __init__(
        self,
        trajectory: np.ndarray,
        energy_func=None,
        q_range: tuple = (-4, 4),
        p_range: tuple = (-4, 4),
        width: float = 5.0,
        height: float = 5.0,
        traj_color: str = COLOR_POSTERIOR,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.axes = Axes(
            x_range=[q_range[0], q_range[1], (q_range[1] - q_range[0]) / 4],
            y_range=[p_range[0], p_range[1], (p_range[1] - p_range[0]) / 4],
            x_length=width, y_length=height,
            axis_config={"color": CREAM, "include_tip": False},
        )

        x_lab = Text("q (angle)", font_size=14, color=SLATE)
        x_lab.next_to(self.axes.x_axis, DOWN, buff=0.15)
        y_lab = Text("p (momentum)", font_size=14, color=SLATE)
        y_lab.next_to(self.axes.y_axis, LEFT, buff=0.15).rotate(PI / 2)

        self.add(self.axes, x_lab, y_lab)

        # Trajectory
        points = [self.axes.c2p(trajectory[i, 0], trajectory[i, 1])
                  for i in range(len(trajectory))]
        if len(points) >= 2:
            traj_line = VMobject()
            traj_line.set_points_smoothly(points[:500])  # cap for performance
            traj_line.set_color(traj_color).set_stroke(width=2)
            self.traj_line = traj_line
            self.add(traj_line)

        # Start dot
        start = Dot(self.axes.c2p(trajectory[0, 0], trajectory[0, 1]),
                     radius=0.06, color=traj_color)
        self.add(start)
