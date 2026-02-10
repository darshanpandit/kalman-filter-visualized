"""Part 2, Scene 2: Linearization and the Jacobian

Shows a nonlinear function, zooms into the operating point, draws tangent
line (Jacobian), and shows how the EKF uses this local linearization.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *


class SceneLinearization(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────────
        title = Text("Linearization: The Key Idea", color=COLOR_TEXT,
                      font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Core insight ────────────────────────────────────────────────
        insight = Text(
            "Zoom in on any smooth curve and it looks like a line",
            color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
        )
        insight.next_to(title, DOWN, buff=STANDARD_BUFF)
        self.play(FadeIn(insight), run_time=NORMAL_ANIM)
        self.wait(PAUSE_SHORT)

        # ── Axes with nonlinear function ────────────────────────────────
        axes = Axes(
            x_range=[-3, 3, 1], y_range=[-2, 4, 1],
            x_length=8, y_length=5,
            axis_config={"color": COLOR_GRID, "include_tip": True},
        ).shift(DOWN * 0.5)

        x_label = axes.get_x_axis_label(
            MathTex(r"\mathbf{x}_{k-1}", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE))
        y_label = axes.get_y_axis_label(
            MathTex(r"f(\mathbf{x}_{k-1})", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE))

        self.play(
            FadeOut(insight),
            Create(axes), FadeIn(x_label), FadeIn(y_label),
            run_time=NORMAL_ANIM,
        )

        # Nonlinear function
        func = lambda x: 0.5 * x**2 - 0.1 * x**3 + 0.5
        curve = axes.plot(func, x_range=[-2.5, 2.8], color=COLOR_PREDICTION)
        curve_label = MathTex(r"f(\mathbf{x})", color=COLOR_PREDICTION,
                               font_size=SMALL_FONT_SIZE)
        curve_label.next_to(axes.c2p(2.5, func(2.5)), RIGHT, buff=0.15)

        self.play(Create(curve), FadeIn(curve_label), run_time=NORMAL_ANIM)
        self.wait(PAUSE_SHORT)

        # ── Operating point ─────────────────────────────────────────────
        x0 = 1.5
        y0 = func(x0)
        point = Dot(axes.c2p(x0, y0), color=COLOR_POSTERIOR, radius=0.07)
        point_label = MathTex(r"\hat{\mathbf{x}}_{k-1}", color=COLOR_POSTERIOR,
                               font_size=SMALL_FONT_SIZE)
        point_label.next_to(point, UL, buff=0.15)

        self.play(FadeIn(point, scale=1.5), FadeIn(point_label), run_time=NORMAL_ANIM)
        self.wait(PAUSE_SHORT)

        # ── Taylor expansion ────────────────────────────────────────────
        taylor = MathTex(
            r"f(\mathbf{x}) \approx f(\hat{\mathbf{x}}) + ",
            r"\underbrace{\frac{\partial f}{\partial \mathbf{x}}}",
            r"_{\mathbf{F}_k}",
            r"(\mathbf{x} - \hat{\mathbf{x}})",
            font_size=BODY_FONT_SIZE, color=COLOR_EQUATION,
        )
        taylor[1].set_color(COLOR_HIGHLIGHT)
        taylor[2].set_color(COLOR_HIGHLIGHT)
        taylor.to_edge(DOWN, buff=0.4)

        self.play(Write(taylor), run_time=SLOW_ANIM)
        self.wait(PAUSE_MEDIUM)

        # ── Tangent line (Jacobian) ─────────────────────────────────────
        eps = 1e-5
        slope = (func(x0 + eps) - func(x0 - eps)) / (2 * eps)

        tangent_func = lambda x: y0 + slope * (x - x0)
        tangent = axes.plot(tangent_func, x_range=[x0 - 2, x0 + 1.5],
                            color=COLOR_HIGHLIGHT)
        tangent_dashed = DashedVMobject(tangent, num_dashes=15)

        jacobian_label = MathTex(r"\mathbf{F}_k = \text{Jacobian}",
                                  color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE)
        jacobian_label.next_to(axes.c2p(x0 + 1.5, tangent_func(x0 + 1.5)),
                                UR, buff=0.15)

        self.play(Create(tangent_dashed), FadeIn(jacobian_label), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG)

        # ── Show error grows far from operating point ───────────────────
        far_point = 0.0
        error_line = Line(
            axes.c2p(far_point, tangent_func(far_point)),
            axes.c2p(far_point, func(far_point)),
            color=PURE_RED, stroke_width=2,
        )
        error_label = MathTex(r"\text{error}", color=PURE_RED,
                               font_size=SMALL_FONT_SIZE)
        error_label.next_to(error_line, LEFT, buff=0.1)

        self.play(Create(error_line), FadeIn(error_label), run_time=NORMAL_ANIM)

        warning = Text(
            "Far from the operating point, linearization error grows",
            color=PURE_RED, font_size=SMALL_FONT_SIZE,
        )
        warning.next_to(taylor, UP, buff=0.3)
        self.play(FadeIn(warning), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG * 2)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
