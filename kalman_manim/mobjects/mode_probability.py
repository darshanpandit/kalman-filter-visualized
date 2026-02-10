"""Mode probability bar for IMM filter visualization.

Stacked horizontal bar showing model mixing weights over time.
"""

from __future__ import annotations

import numpy as np
from manim import *

from kalman_manim.style import COLOR_TEXT, SLATE, SMALL_FONT_SIZE


class ModeProbabilityBar(VGroup):
    """Animated stacked bar showing IMM mode probabilities.

    Parameters
    ----------
    model_names : list of str
    colors : list of str
    width : float
        Bar width in scene units.
    height : float
        Bar height.
    """

    def __init__(
        self,
        model_names: list[str],
        colors: list[str],
        width: float = 4.0,
        height: float = 0.5,
        font_size: int = 14,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_names = model_names
        self.colors = colors
        self.width = width
        self.height = height
        self.n_models = len(model_names)

        # Background bar
        self.bg_bar = Rectangle(
            width=width, height=height,
            fill_color="#1a1a2e", fill_opacity=0.5,
            stroke_color=SLATE, stroke_width=1,
        )
        self.add(self.bg_bar)

        # Probability segments (initially equal)
        self.segments = VGroup()
        probs = np.ones(self.n_models) / self.n_models
        self._create_segments(probs)

        # Labels
        self.labels = VGroup()
        for i, (name, color) in enumerate(zip(model_names, colors)):
            label = Text(name, color=color, font_size=font_size)
            self.labels.add(label)
        self.labels.arrange(RIGHT, buff=0.3)
        self.labels.next_to(self.bg_bar, UP, buff=0.15)
        self.add(self.labels)

    def _create_segments(self, probabilities: np.ndarray):
        """Create colored segments for given probabilities."""
        self.segments.submobjects.clear()
        x_start = self.bg_bar.get_left()[0]

        for i in range(self.n_models):
            seg_width = probabilities[i] * self.width
            if seg_width < 0.01:
                continue
            seg = Rectangle(
                width=seg_width, height=self.height,
                fill_color=self.colors[i], fill_opacity=0.7,
                stroke_width=0,
            )
            seg.move_to(
                self.bg_bar.get_center()
            ).align_to(
                self.bg_bar.get_left() + RIGHT * sum(
                    probabilities[j] * self.width for j in range(i)
                ),
                LEFT,
            )
            self.segments.add(seg)

        self.add(self.segments)

    def set_probabilities(self, probabilities: np.ndarray):
        """Update the bar to show new probabilities."""
        self.remove(self.segments)
        self.segments = VGroup()
        self._create_segments(probabilities)
        return self
