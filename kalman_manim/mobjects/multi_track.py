"""Multi-track trajectory plot for multi-target tracking scenes.

Displays N colored trajectory paths simultaneously on shared axes.
"""

from __future__ import annotations

import numpy as np
from manim import *

from kalman_manim.style import COLOR_TEXT, SLATE, CREAM, SMALL_FONT_SIZE


# 8-color palette for simultaneous tracks
TRACK_COLORS = [
    "#e63946",  # red
    "#457b9d",  # blue
    "#f4a261",  # gold
    "#2a9d8f",  # teal
    "#9b59b6",  # violet
    "#e74c3c",  # bright red
    "#1abc9c",  # emerald
    "#e67e22",  # orange
]


class MultiTrackPlot(VGroup):
    """Plot multiple trajectories on shared axes.

    Parameters
    ----------
    tracks : list of np.ndarray, each (T_i, 2) — position sequences
    track_labels : list of str or None
    colors : list of str or None — one per track
    axes_range : tuple (x_min, x_max, y_min, y_max) or None (auto)
    width, height : float — axes dimensions in scene units
    """

    def __init__(
        self,
        tracks: list[np.ndarray],
        track_labels: list[str] | None = None,
        colors: list[str] | None = None,
        axes_range: tuple | None = None,
        width: float = 7.0,
        height: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_tracks = len(tracks)
        if colors is None:
            colors = [TRACK_COLORS[i % len(TRACK_COLORS)] for i in range(self.n_tracks)]
        if track_labels is None:
            track_labels = [f"Track {i}" for i in range(self.n_tracks)]

        # Determine axes range
        if axes_range is None:
            all_pts = np.vstack(tracks)
            margin = 1.0
            x_min, x_max = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
            y_min, y_max = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin
        else:
            x_min, x_max, y_min, y_max = axes_range

        self.axes = Axes(
            x_range=[x_min, x_max, (x_max - x_min) / 5],
            y_range=[y_min, y_max, (y_max - y_min) / 5],
            x_length=width,
            y_length=height,
            axis_config={"color": CREAM, "include_tip": False},
        )
        self.add(self.axes)

        # Draw each track
        self.track_lines = VGroup()
        self.track_dots = VGroup()
        for i, (track, color) in enumerate(zip(tracks, colors)):
            points = [self.axes.c2p(p[0], p[1]) for p in track]
            if len(points) >= 2:
                line = VMobject()
                line.set_points_smoothly(points)
                line.set_color(color).set_stroke(width=2.0)
                self.track_lines.add(line)

            # Start/end dots
            start_dot = Dot(points[0], radius=0.06, color=color)
            end_dot = Dot(points[-1], radius=0.06, color=color)
            self.track_dots.add(start_dot, end_dot)

        self.add(self.track_lines, self.track_dots)

        # Legend
        legend_items = VGroup()
        for i in range(min(self.n_tracks, 6)):
            item = VGroup(
                Line(ORIGIN, RIGHT * 0.3, color=colors[i], stroke_width=2.5),
                Text(track_labels[i], color=colors[i], font_size=12),
            ).arrange(RIGHT, buff=0.1)
            legend_items.add(item)
        legend_items.arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        legend_items.next_to(self.axes, RIGHT, buff=0.3)
        self.legend = legend_items
        self.add(legend_items)
