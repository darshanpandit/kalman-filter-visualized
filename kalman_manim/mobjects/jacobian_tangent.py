"""JacobianTangent — visualizes linearization at a point on a nonlinear curve."""

from __future__ import annotations

from manim import *
import numpy as np

from kalman_manim.style import COLOR_HIGHLIGHT, COLOR_PREDICTION, SMALL_FONT_SIZE


class JacobianTangent(VGroup):
    """Draws a tangent line at a point on a parametric curve.

    Visualizes how the EKF linearizes a nonlinear function around the
    current state estimate using the Jacobian.

    Parameters
    ----------
    axes : Axes
        The coordinate system.
    func : callable(float) -> float
        The nonlinear function y = func(x).
    x_point : float
        The x-coordinate at which to draw the tangent.
    tangent_length : float
        Half-length of the tangent line in data coordinates.
    curve_color : str
        Color for the nonlinear curve.
    tangent_color : str
        Color for the tangent line.
    x_range : tuple
        (min, max) for drawing the full curve.
    show_point : bool
        Whether to show the linearization point dot.
    show_label : bool
        Whether to show "Jacobian" label on the tangent.
    """

    def __init__(
        self,
        axes: Axes,
        func,
        x_point: float,
        tangent_length: float = 2.0,
        curve_color: str = COLOR_PREDICTION,
        tangent_color: str = COLOR_HIGHLIGHT,
        x_range: tuple = (-3, 3),
        show_point: bool = True,
        show_label: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._axes = axes
        self._func = func
        self._x_point = x_point

        # Draw the nonlinear curve
        self.curve = axes.plot(func, x_range=list(x_range), color=curve_color)
        self.add(self.curve)

        # Compute tangent slope (numerical derivative)
        eps = 1e-5
        slope = (func(x_point + eps) - func(x_point - eps)) / (2 * eps)
        y_point = func(x_point)

        # Tangent line
        x_left = x_point - tangent_length
        x_right = x_point + tangent_length
        y_left = y_point + slope * (x_left - x_point)
        y_right = y_point + slope * (x_right - x_point)

        self.tangent = DashedLine(
            axes.c2p(x_left, y_left),
            axes.c2p(x_right, y_right),
            color=tangent_color,
            stroke_width=2.5,
        )
        self.add(self.tangent)

        # Linearization point
        if show_point:
            self.point_dot = Dot(
                axes.c2p(x_point, y_point),
                color=tangent_color,
                radius=0.06,
            )
            self.add(self.point_dot)

        # Label
        if show_label:
            self.label = MathTex(
                r"\mathbf{F} = \text{Jacobian}",
                font_size=SMALL_FONT_SIZE,
                color=tangent_color,
            )
            self.label.next_to(
                axes.c2p(x_right, y_right), UR, buff=0.15,
            )
            self.add(self.label)
