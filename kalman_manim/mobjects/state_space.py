"""StateSpace — a labeled 2D coordinate grid for state visualization."""

from manim import *

from kalman_manim.style import COLOR_GRID, COLOR_TEXT, BODY_FONT_SIZE


class StateSpace(VGroup):
    """A styled 2D axes setup for visualizing state vectors and covariance ellipses.

    Provides a NumberPlane with subtle grid lines and labeled axes,
    matching the Swiss/dark theme.

    Parameters
    ----------
    x_range : list
        [min, max, step] for x-axis.
    y_range : list
        [min, max, step] for y-axis.
    x_length : float
        Physical width on screen.
    y_length : float
        Physical height on screen.
    x_label : str
        LaTeX label for x-axis (e.g., r"x" or r"\\text{position}").
    y_label : str
        LaTeX label for y-axis (e.g., r"v" or r"\\text{velocity}").
    show_grid : bool
        Whether to show background grid lines.
    """

    def __init__(
        self,
        x_range=None,
        y_range=None,
        x_length: float = 10,
        y_length: float = 6,
        x_label: str = r"x",
        y_label: str = r"y",
        show_grid: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if x_range is None:
            x_range = [-5, 5, 1]
        if y_range is None:
            y_range = [-3, 3, 1]

        self.axes = Axes(
            x_range=x_range,
            y_range=y_range,
            x_length=x_length,
            y_length=y_length,
            axis_config={
                "color": COLOR_GRID,
                "include_tip": True,
                "tip_length": 0.2,
                "tip_width": 0.15,
            },
        )
        self.add(self.axes)

        # Grid
        if show_grid:
            self.grid = NumberPlane(
                x_range=x_range,
                y_range=y_range,
                x_length=x_length,
                y_length=y_length,
                background_line_style={
                    "stroke_color": COLOR_GRID,
                    "stroke_width": 0.5,
                    "stroke_opacity": 0.3,
                },
                axis_config={"stroke_opacity": 0},  # hide plane's own axes
                faded_line_style={
                    "stroke_color": COLOR_GRID,
                    "stroke_width": 0.3,
                    "stroke_opacity": 0.15,
                },
            )
            # Insert grid behind axes
            self.submobjects.insert(0, self.grid)

        # Axis labels
        self.x_label_mob = self.axes.get_x_axis_label(
            MathTex(x_label, color=COLOR_TEXT, font_size=BODY_FONT_SIZE)
        )
        self.y_label_mob = self.axes.get_y_axis_label(
            MathTex(y_label, color=COLOR_TEXT, font_size=BODY_FONT_SIZE)
        )
        self.add(self.x_label_mob, self.y_label_mob)

    def c2p(self, x, y):
        """Shortcut for axes.coords_to_point."""
        return self.axes.c2p(x, y)

    def p2c(self, point):
        """Shortcut for axes.point_to_coords."""
        return self.axes.p2c(point)
