"""Subtle on-screen annotation for theory-observation gaps.

Use when the visualization doesn't fully support a theoretical claim,
to maintain honesty without being defensive.
"""

from __future__ import annotations

from manim import VGroup, Text, RoundedRectangle, DR
import numpy as np

from kalman_manim.style import SLATE, DARK_SLATE, SMALL_FONT_SIZE


def make_observation_note(
    text: str,
    position=DR,
    font_size: int = 18,
    text_color: str = SLATE,
    bg_color: str = DARK_SLATE,
    bg_opacity: float = 0.85,
    buff: float = 0.3,
    corner_radius: float = 0.1,
    max_width: float = 4.5,
) -> VGroup:
    """Create a subtle on-screen annotation for theory-observation gaps.

    Parameters
    ----------
    text : str
        The annotation text. Keep concise (1-2 lines).
    position
        Manim direction constant (DR, DL, UR, UL) for corner placement.
    font_size : int
        Font size. Default 18 (smaller than body text).
    text_color : str
        Text color. Default SLATE (muted).
    bg_color : str
        Background rectangle color.
    bg_opacity : float
        Background opacity.
    buff : float
        Distance from screen edge.
    corner_radius : float
        Rounded corner radius.
    max_width : float
        Maximum width before text wraps.

    Returns
    -------
    VGroup containing background rectangle and text, positioned at corner.
    """
    label = Text(
        text,
        color=text_color,
        font_size=font_size,
        line_spacing=1.1,
    )

    # Scale down if too wide
    if label.width > max_width:
        label.scale(max_width / label.width)

    bg = RoundedRectangle(
        corner_radius=corner_radius,
        width=label.width + 0.3,
        height=label.height + 0.2,
        color=bg_color,
        fill_color=bg_color,
        fill_opacity=bg_opacity,
        stroke_width=0.5,
    )
    bg.move_to(label)

    group = VGroup(bg, label)
    group.to_corner(position, buff=buff)
    group.set_z_index(15)

    return group
