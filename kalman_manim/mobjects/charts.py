"""Chart mobjects for benchmark visualization scenes."""

from __future__ import annotations

import numpy as np
from manim import *


class RMSELineChart(VGroup):
    """Axes with colored lines and optional confidence bands for sweep results.

    Parameters
    ----------
    x_values : np.ndarray
        X-axis values (e.g. turn_rates).
    y_data : dict
        Mapping filter_name -> np.ndarray of y values.
    y_std : dict or None
        Mapping filter_name -> np.ndarray of std values for confidence bands.
    colors : dict
        Mapping filter_name -> color string.
    x_label : str
        X-axis label.
    y_label : str
        Y-axis label.
    width : float
        Chart width in scene units.
    height : float
        Chart height in scene units.
    """

    def __init__(
        self,
        x_values: np.ndarray,
        y_data: dict,
        y_std: dict | None = None,
        colors: dict | None = None,
        x_label: str = "Turn Rate",
        y_label: str = "RMSE",
        width: float = 8.0,
        height: float = 4.5,
        axis_color: str = "#f1faee",
        grid_color: str = "#4a4e69",
        label_font_size: int = 20,
        tick_font_size: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.x_values = np.asarray(x_values)
        self.y_data = y_data
        self.y_std = y_std or {}
        self.colors = colors or {}
        self.filter_names = list(y_data.keys())

        # Determine axis ranges
        x_min, x_max = float(self.x_values.min()), float(self.x_values.max())
        all_y = np.concatenate(list(y_data.values()))
        y_min = 0.0
        y_max = float(np.max(all_y)) * 1.15

        self.axes = Axes(
            x_range=[x_min, x_max, (x_max - x_min) / 5],
            y_range=[y_min, y_max, y_max / 5],
            x_length=width,
            y_length=height,
            axis_config={
                "color": axis_color,
                "include_tip": False,
                "tick_size": 0.05,
            },
        )

        # Labels
        x_lab = Text(x_label, font_size=label_font_size, color=axis_color)
        x_lab.next_to(self.axes.x_axis, DOWN, buff=0.3)
        y_lab = Text(y_label, font_size=label_font_size, color=axis_color)
        y_lab.next_to(self.axes.y_axis, LEFT, buff=0.3).rotate(PI / 2)

        self.add(self.axes, x_lab, y_lab)

        # Store line/band mobjects for animation
        self.lines = {}
        self.bands = {}

        for name in self.filter_names:
            color = self.colors.get(name, WHITE)
            y = y_data[name]
            points = [
                self.axes.c2p(self.x_values[i], y[i])
                for i in range(len(self.x_values))
            ]
            line = VMobject()
            line.set_points_smoothly(points)
            line.set_color(color).set_stroke(width=2.5)
            self.lines[name] = line

            # Confidence band
            if name in self.y_std:
                std = self.y_std[name]
                upper = [
                    self.axes.c2p(self.x_values[i], y[i] + std[i])
                    for i in range(len(self.x_values))
                ]
                lower = [
                    self.axes.c2p(self.x_values[i], y[i] - std[i])
                    for i in range(len(self.x_values))
                ][::-1]  # reversed for closed polygon
                band_points = upper + lower
                band = Polygon(
                    *band_points,
                    color=color,
                    fill_opacity=0.12,
                    stroke_width=0,
                )
                self.bands[name] = band

    def animate_line(self, filter_name: str, run_time: float = 1.5) -> list:
        """Return animations to draw one filter's line + band."""
        anims = []
        if filter_name in self.bands:
            anims.append(FadeIn(self.bands[filter_name]))
            self.add(self.bands[filter_name])
        if filter_name in self.lines:
            anims.append(Create(self.lines[filter_name], run_time=run_time))
            self.add(self.lines[filter_name])
        return anims


class FilterBarChart(VGroup):
    """Grouped bar chart with error bars for corpus results.

    Parameters
    ----------
    filter_names : list of str
    values : np.ndarray (n_filters,)
    errors : np.ndarray (n_filters,) or None
    colors : dict mapping filter_name -> color
    """

    def __init__(
        self,
        filter_names: list[str],
        values: np.ndarray,
        errors: np.ndarray | None = None,
        colors: dict | None = None,
        title: str = "Average RMSE",
        width: float = 6.0,
        height: float = 3.5,
        axis_color: str = "#f1faee",
        label_font_size: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filter_names = filter_names
        self.values = np.asarray(values)
        self.errors = np.asarray(errors) if errors is not None else None
        self.colors = colors or {}

        n = len(filter_names)
        bar_width = width / (n * 1.5)
        y_max = float(self.values.max()) * 1.3

        self.axes = Axes(
            x_range=[0, n + 1, 1],
            y_range=[0, y_max, y_max / 4],
            x_length=width,
            y_length=height,
            axis_config={"color": axis_color, "include_tip": False},
        )

        title_text = Text(title, font_size=label_font_size, color=axis_color)
        title_text.next_to(self.axes, UP, buff=0.2)
        self.add(self.axes, title_text)

        self.bars = VGroup()
        self.labels = VGroup()
        self.error_bars = VGroup()

        for i, name in enumerate(filter_names):
            color = self.colors.get(name, WHITE)
            x_pos = i + 1
            bar_height = self.values[i]

            bottom = self.axes.c2p(x_pos, 0)
            top = self.axes.c2p(x_pos, bar_height)
            bar_h = top[1] - bottom[1]

            bar = Rectangle(
                width=bar_width,
                height=bar_h,
                fill_color=color,
                fill_opacity=0.75,
                stroke_color=color,
                stroke_width=1,
            )
            bar.move_to(self.axes.c2p(x_pos, bar_height / 2))
            self.bars.add(bar)

            label = Text(str(name), font_size=16, color=color)
            label.next_to(bar, DOWN, buff=0.15)
            self.labels.add(label)

            # Value label on top
            val_label = Text(f"{float(bar_height):.3f}", font_size=14, color=axis_color)
            val_label.next_to(bar, UP, buff=0.08)
            self.labels.add(val_label)

            # Error bar
            if self.errors is not None:
                err = self.errors[i]
                err_top = self.axes.c2p(x_pos, bar_height + err)
                err_bot = self.axes.c2p(x_pos, bar_height - err)
                err_line = Line(err_bot, err_top, color=axis_color, stroke_width=1.5)
                cap_w = bar_width * 0.3
                top_cap = Line(
                    err_top + LEFT * cap_w / 2,
                    err_top + RIGHT * cap_w / 2,
                    color=axis_color, stroke_width=1.5,
                )
                bot_cap = Line(
                    err_bot + LEFT * cap_w / 2,
                    err_bot + RIGHT * cap_w / 2,
                    color=axis_color, stroke_width=1.5,
                )
                self.error_bars.add(err_line, top_cap, bot_cap)

    def animate_bars(self, run_time: float = 1.0) -> list:
        """Return animations to grow bars from bottom."""
        anims = []
        for bar in self.bars:
            anims.append(GrowFromEdge(bar, DOWN))
        anims.append(FadeIn(self.labels))
        if len(self.error_bars) > 0:
            anims.append(FadeIn(self.error_bars))
        self.add(self.bars, self.labels, self.error_bars)
        return anims


class ErrorHistogram(VGroup):
    """Overlaid semi-transparent histograms per filter.

    Parameters
    ----------
    data : dict mapping filter_name -> np.ndarray of error values
    colors : dict mapping filter_name -> color
    """

    def __init__(
        self,
        data: dict,
        colors: dict | None = None,
        n_bins: int = 20,
        width: float = 6.0,
        height: float = 3.0,
        axis_color: str = "#f1faee",
        label_font_size: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data
        self.colors = colors or {}
        self.filter_names = list(data.keys())

        # Compute global bin edges
        all_vals = np.concatenate(list(data.values()))
        self.bin_edges = np.linspace(
            float(np.min(all_vals)),
            float(np.percentile(all_vals, 98)),
            n_bins + 1,
        )
        bin_width_val = self.bin_edges[1] - self.bin_edges[0]

        # Find max count for y-axis
        max_count = 0
        for vals in data.values():
            counts, _ = np.histogram(vals, bins=self.bin_edges)
            max_count = max(max_count, counts.max())

        x_min = float(self.bin_edges[0])
        x_max = float(self.bin_edges[-1])
        y_max = float(max_count) * 1.15

        self.axes = Axes(
            x_range=[x_min, x_max, (x_max - x_min) / 5],
            y_range=[0, y_max, y_max / 4],
            x_length=width,
            y_length=height,
            axis_config={"color": axis_color, "include_tip": False},
        )

        x_lab = Text("RMSE", font_size=label_font_size, color=axis_color)
        x_lab.next_to(self.axes.x_axis, DOWN, buff=0.25)
        y_lab = Text("Count", font_size=label_font_size, color=axis_color)
        y_lab.next_to(self.axes.y_axis, LEFT, buff=0.25).rotate(PI / 2)

        self.add(self.axes, x_lab, y_lab)

        self.hist_groups = {}
        for name in self.filter_names:
            color = self.colors.get(name, WHITE)
            counts, _ = np.histogram(data[name], bins=self.bin_edges)

            bars = VGroup()
            for j in range(len(counts)):
                if counts[j] == 0:
                    continue
                left = self.bin_edges[j]
                right = self.bin_edges[j + 1]
                bl = self.axes.c2p(left, 0)
                tr = self.axes.c2p(right, counts[j])
                bar = Rectangle(
                    width=tr[0] - bl[0],
                    height=tr[1] - bl[1],
                    fill_color=color,
                    fill_opacity=0.3,
                    stroke_color=color,
                    stroke_width=1,
                )
                bar.move_to((np.array(bl) + np.array(tr)) / 2)
                bars.add(bar)

            self.hist_groups[name] = bars

    def animate_histogram(self, filter_name: str) -> list:
        """Return animations for one filter's histogram."""
        group = self.hist_groups.get(filter_name, VGroup())
        self.add(group)
        return [FadeIn(group)]
