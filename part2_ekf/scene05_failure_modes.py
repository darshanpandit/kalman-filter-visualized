"""Part 2, Scene 5: EKF Failure Modes

Demonstrates when EKF linearization breaks down:
highly nonlinear functions or large uncertainty.
Teases the UKF as the solution.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.gaussian_ellipse import GaussianEllipse


class SceneEKFFailureModes(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("When the EKF Fails", color=COLOR_TEXT,
                      font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Setup: highly nonlinear function ────────────────────────────
        axes = Axes(
            x_range=[-3, 3, 1], y_range=[-2, 4, 1],
            x_length=6, y_length=4,
            axis_config={"color": COLOR_GRID},
        ).shift(LEFT * 2 + DOWN * 0.3)

        func = lambda x: np.sin(2 * x) + 0.3 * x**2
        curve = axes.plot(func, x_range=[-2.8, 2.8], color=COLOR_PREDICTION)

        self.play(Create(axes), Create(curve), run_time=NORMAL_ANIM)

        # ── Case 1: Large uncertainty → bad linearization ───────────────
        case1_title = Text("Case 1: Large uncertainty",
                            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE)
        case1_title.to_edge(RIGHT, buff=0.3).shift(UP * 2)
        self.play(FadeIn(case1_title), run_time=FAST_ANIM)

        # Show large ellipse on x-axis
        x0 = 0.5
        large_ellipse = GaussianEllipse(
            mean=np.array([x0, 0]),
            cov=np.array([[2.0, 0], [0, 0.01]]),
            color=COLOR_POSTERIOR,
            axes=axes,
            fill_opacity=0.2,
        )
        self.play(FadeIn(large_ellipse), run_time=NORMAL_ANIM)

        # Tangent at x0
        eps = 1e-5
        slope = (func(x0 + eps) - func(x0 - eps)) / (2 * eps)
        tangent = axes.plot(
            lambda x: func(x0) + slope * (x - x0),
            x_range=[x0 - 2.5, x0 + 2.5],
            color=COLOR_HIGHLIGHT,
        )
        tangent_dashed = DashedVMobject(tangent, num_dashes=12)
        self.play(Create(tangent_dashed), run_time=NORMAL_ANIM)

        case1_note = Text(
            "Tangent is a poor approximation\n"
            "over a wide uncertainty range",
            color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
            line_spacing=1.2,
        )
        case1_note.next_to(case1_title, DOWN, buff=0.3)
        self.play(FadeIn(case1_note), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG)

        # ── Case 2: Highly nonlinear region ─────────────────────────────
        self.play(
            FadeOut(large_ellipse), FadeOut(tangent_dashed),
            FadeOut(case1_title), FadeOut(case1_note),
            run_time=FAST_ANIM,
        )

        case2_title = Text("Case 2: Strong nonlinearity",
                            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE)
        case2_title.to_edge(RIGHT, buff=0.3).shift(UP * 2)
        self.play(FadeIn(case2_title), run_time=FAST_ANIM)

        # Point near inflection of sin(2x)
        x1 = 1.5
        small_ellipse = GaussianEllipse(
            mean=np.array([x1, 0]),
            cov=np.array([[0.3, 0], [0, 0.01]]),
            color=COLOR_POSTERIOR,
            axes=axes,
            fill_opacity=0.2,
        )
        slope1 = (func(x1 + eps) - func(x1 - eps)) / (2 * eps)
        tangent1 = axes.plot(
            lambda x: func(x1) + slope1 * (x - x1),
            x_range=[x1 - 1.5, x1 + 1.5],
            color=COLOR_HIGHLIGHT,
        )
        tangent1_dashed = DashedVMobject(tangent1, num_dashes=10)

        self.play(FadeIn(small_ellipse), Create(tangent1_dashed), run_time=NORMAL_ANIM)

        case2_note = Text(
            "Even small uncertainty can be\n"
            "distorted by strong curvature",
            color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
            line_spacing=1.2,
        )
        case2_note.next_to(case2_title, DOWN, buff=0.3)
        self.play(FadeIn(case2_note), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG)

        # ── Teaser ──────────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)

        teaser = Text(
            "What if we could avoid linearization entirely?",
            color=COLOR_HIGHLIGHT, font_size=TITLE_FONT_SIZE,
        )
        self.play(FadeIn(teaser, scale=0.9), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG)

        next_ep = Text(
            "Next: The Unscented Kalman Filter",
            color=COLOR_TEXT, font_size=HEADING_FONT_SIZE,
        )
        next_ep.next_to(teaser, DOWN, buff=LARGE_BUFF)
        self.play(FadeIn(next_ep, shift=UP * 0.2), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG * 2)

        self.play(FadeOut(teaser), FadeOut(next_ep), run_time=NORMAL_ANIM)
