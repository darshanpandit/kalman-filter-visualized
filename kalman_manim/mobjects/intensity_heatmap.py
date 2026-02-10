"""Intensity heatmap for PHD filter visualization.

2D grid showing the PHD intensity function (expected number of targets
per unit area). Used in Part 7 Scene 3.
"""

from __future__ import annotations

import numpy as np
from manim import *

from kalman_manim.style import SLATE, DARK_SLATE, COLOR_FILTER_PHD


class IntensityHeatmap(VGroup):
    """2D heatmap of PHD intensity values.

    Parameters
    ----------
    intensity : np.ndarray, shape (ny, nx)
        Intensity values on a grid.
    x_range : tuple (x_min, x_max)
    y_range : tuple (y_min, y_max)
    high_color : str
    low_color : str
    width, height : float
    """

    def __init__(
        self,
        intensity: np.ndarray,
        x_range: tuple = (-5, 5),
        y_range: tuple = (-5, 5),
        high_color: str = COLOR_FILTER_PHD,
        low_color: str = DARK_SLATE,
        width: float = 5.0,
        height: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        ny, nx = intensity.shape
        cell_w = width / nx
        cell_h = height / ny

        high_rgb = color_to_rgb(high_color)
        low_rgb = color_to_rgb(low_color)

        # Normalize intensity to [0, 1]
        vmax = intensity.max()
        if vmax > 0:
            normed = intensity / vmax
        else:
            normed = intensity

        self.cells = VGroup()
        for i in range(ny):
            for j in range(nx):
                v = float(normed[i, j])
                rgb = low_rgb + v * (high_rgb - low_rgb)
                cell = Square(
                    side_length=min(cell_w, cell_h),
                    fill_color=rgb_to_color(rgb),
                    fill_opacity=0.8,
                    stroke_width=0.3,
                    stroke_color=SLATE,
                )
                cell.move_to(
                    RIGHT * (j * cell_w - width / 2 + cell_w / 2)
                    + UP * (i * cell_h - height / 2 + cell_h / 2)
                )
                self.cells.add(cell)

        self.add(self.cells)

        # Axis labels
        x_lab = Text(
            f"{x_range[0]:.0f} ← x → {x_range[1]:.0f}",
            color=SLATE, font_size=12,
        )
        x_lab.next_to(self.cells, DOWN, buff=0.2)
        y_lab = Text(
            f"{y_range[0]:.0f} ← y → {y_range[1]:.0f}",
            color=SLATE, font_size=12,
        )
        y_lab.next_to(self.cells, LEFT, buff=0.2).rotate(PI / 2)
        self.add(x_lab, y_lab)
