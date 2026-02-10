"""Reusable color-coded comparison table mobject for paper results.

Used across Parts 6-9 to display published results from papers.
"""

from __future__ import annotations

import numpy as np
from manim import *

from kalman_manim.style import (
    COLOR_TEXT, SLATE, DARK_SLATE,
    SMALL_FONT_SIZE, BODY_FONT_SIZE,
)


class ComparisonTable(VGroup):
    """Color-coded table for displaying benchmark results from papers.

    Parameters
    ----------
    headers : list of str
        Column headers (e.g., ["Method", "RMSE", "MAE"]).
    rows : list of list of str
        Each row is a list of cell values as strings.
    row_colors : list of str or None
        One color per row (applied to the method/first column).
        If None, all rows use COLOR_TEXT.
    title : str or None
        Table title displayed above.
    highlight_best : list of int or None
        Column indices where the lowest value should be highlighted.
        Only applies to numeric columns.
    width : float
        Total table width in scene units.
    font_size : int
        Font size for cell text.
    header_font_size : int
        Font size for column headers.
    """

    def __init__(
        self,
        headers: list[str],
        rows: list[list[str]],
        row_colors: list[str] | None = None,
        title: str | None = None,
        highlight_best: list[int] | None = None,
        width: float = 9.0,
        font_size: int = 20,
        header_font_size: int = 22,
        bg_color: str = DARK_SLATE,
        bg_opacity: float = 0.6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_cols = len(headers)
        self.n_rows = len(rows)
        col_width = width / self.n_cols
        row_height = 0.45

        if row_colors is None:
            row_colors = [COLOR_TEXT] * self.n_rows

        # Find best values for highlighting
        best_in_col = {}
        if highlight_best:
            for col_idx in highlight_best:
                vals = []
                for row in rows:
                    try:
                        vals.append(float(row[col_idx]))
                    except (ValueError, IndexError):
                        vals.append(float("inf"))
                best_in_col[col_idx] = min(vals)

        # Title
        if title:
            title_text = Text(
                title, color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
            )
            self.add(title_text)

        # Background
        total_height = (self.n_rows + 1) * row_height
        bg = RoundedRectangle(
            corner_radius=0.08,
            width=width + 0.4,
            height=total_height + 0.3,
            fill_color=bg_color,
            fill_opacity=bg_opacity,
            stroke_width=0.5,
            stroke_color=SLATE,
        )
        self.bg = bg

        # Header row
        header_group = VGroup()
        for j, h in enumerate(headers):
            cell = Text(h, color=SLATE, font_size=header_font_size)
            cell.move_to(
                bg.get_top()
                + DOWN * (row_height * 0.5 + 0.15)
                + RIGHT * (col_width * (j - self.n_cols / 2 + 0.5))
            )
            header_group.add(cell)

        # Divider line under headers
        divider = Line(
            bg.get_left() + DOWN * (row_height + 0.15) + RIGHT * 0.1,
            bg.get_right() + DOWN * (row_height + 0.15) + LEFT * 0.1,
            color=SLATE,
            stroke_width=0.8,
        )

        # Data rows
        self.row_groups = VGroup()
        for i, row_data in enumerate(rows):
            row_group = VGroup()
            y_offset = DOWN * (row_height * (i + 1.5) + 0.15)
            for j, val in enumerate(row_data):
                # Determine color
                if j == 0:
                    color = row_colors[i]
                elif j in best_in_col:
                    try:
                        if float(val) == best_in_col[j]:
                            color = "#2ecc71"  # green highlight for best
                        else:
                            color = COLOR_TEXT
                    except ValueError:
                        color = COLOR_TEXT
                else:
                    color = COLOR_TEXT

                cell = Text(str(val), color=color, font_size=font_size)
                cell.move_to(
                    bg.get_top()
                    + y_offset
                    + RIGHT * (col_width * (j - self.n_cols / 2 + 0.5))
                )
                row_group.add(cell)
            self.row_groups.add(row_group)

        if title:
            title_text.next_to(bg, UP, buff=0.2)

        self.add(bg, header_group, divider, self.row_groups)

    def animate_rows(self, run_time: float = 0.5) -> list:
        """Return animations that fade in rows one at a time."""
        return [FadeIn(row, shift=RIGHT * 0.2) for row in self.row_groups]
