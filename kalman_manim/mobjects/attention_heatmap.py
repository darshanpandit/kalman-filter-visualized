"""Attention heatmap mobject for visualizing transformer attention patterns.

Shows a grid of colored cells representing attention weights, animatable
row-by-row to demonstrate causal attention in filtering contexts.
"""

from __future__ import annotations

import numpy as np
from manim import *

from kalman_manim.style import (
    COLOR_TEXT, COLOR_FILTER_TF, SLATE, DARK_SLATE,
    SMALL_FONT_SIZE,
)


class AttentionHeatmap(VGroup):
    """Grid visualization of an attention weight matrix.

    Parameters
    ----------
    weights : np.ndarray, shape (T, T)
        Attention weight matrix (lower-triangular for causal).
    cell_size : float
        Size of each cell in scene units.
    high_color : str
        Color for high attention weights.
    low_color : str
        Color for zero/low attention weights.
    show_labels : bool
        Whether to show row/column index labels.
    max_display : int
        Maximum grid dimension (truncate if T > max_display).
    """

    def __init__(
        self,
        weights: np.ndarray,
        cell_size: float = 0.35,
        high_color: str = COLOR_FILTER_TF,
        low_color: str = DARK_SLATE,
        show_labels: bool = True,
        max_display: int = 15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        T = min(weights.shape[0], max_display)
        self.T = T
        self.weights = weights[:T, :T]
        self.cell_size = cell_size

        # Color interpolation
        high_rgb = color_to_rgb(high_color)
        low_rgb = color_to_rgb(low_color)

        self.cells = VGroup()
        self.row_groups = []

        for i in range(T):
            row_cells = VGroup()
            for j in range(T):
                w = float(self.weights[i, j])
                # Interpolate color
                rgb = low_rgb + w * (high_rgb - low_rgb)
                cell_color = rgb_to_color(rgb)

                cell = Square(
                    side_length=cell_size,
                    fill_color=cell_color,
                    fill_opacity=0.9 if w > 0.01 else 0.3,
                    stroke_color=SLATE,
                    stroke_width=0.5,
                )
                cell.move_to(
                    RIGHT * (j * cell_size) + DOWN * (i * cell_size)
                )
                row_cells.add(cell)
            self.cells.add(row_cells)
            self.row_groups.append(row_cells)

        # Center the grid
        self.cells.center()
        self.add(self.cells)

        # Labels
        if show_labels:
            self.labels = VGroup()
            for i in range(T):
                # Row labels (left)
                row_label = Text(
                    str(i), color=SLATE, font_size=12,
                )
                row_label.next_to(self.row_groups[i][0], LEFT, buff=0.1)
                self.labels.add(row_label)

                # Column labels (top)
                if i == 0:
                    for j in range(T):
                        col_label = Text(
                            str(j), color=SLATE, font_size=12,
                        )
                        col_label.next_to(
                            self.row_groups[0][j], UP, buff=0.1,
                        )
                        self.labels.add(col_label)

            self.add(self.labels)

        # Axis titles
        query_label = Text("query (t)", color=SLATE, font_size=14)
        query_label.next_to(self.cells, LEFT, buff=0.4).rotate(PI / 2)
        key_label = Text("key (past)", color=SLATE, font_size=14)
        key_label.next_to(self.cells, UP, buff=0.4)
        self.add(query_label, key_label)

    def animate_rows(self, run_time_per_row: float = 0.3) -> list:
        """Return animations that reveal the heatmap row by row."""
        anims = []
        for row in self.row_groups:
            anims.append(FadeIn(row, run_time=run_time_per_row))
        return anims
