"""Scene 3: 1D Gaussian Multiplication

Shows two 1D Gaussians (prediction & measurement) and derives their product.
Demonstrates that combining information always reduces uncertainty.
Introduces the Kalman gain K as a blending weight.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.utils import gaussian_product_1d, gaussian_1d_pdf


class SceneGaussian1D(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Parameters ──────────────────────────────────────────────────
        mu1, var1 = 2.0, 1.5    # Prediction
        mu2, var2 = 4.0, 0.8    # Measurement
        mu_new, var_new = gaussian_product_1d(mu1, var1, mu2, var2)

        # ── Axes ────────────────────────────────────────────────────────
        axes = Axes(
            x_range=[-1, 7, 1], y_range=[0, 0.6, 0.1],
            x_length=10, y_length=4,
            axis_config={"color": COLOR_GRID, "include_tip": True},
        ).shift(DOWN * 0.5)
        x_label = axes.get_x_axis_label(
            MathTex(r"x", color=COLOR_TEXT, font_size=BODY_FONT_SIZE))
        self.play(Create(axes), FadeIn(x_label), run_time=NORMAL_ANIM)

        # ── Title ───────────────────────────────────────────────────────
        title = Text("Multiplying Two Gaussians", color=COLOR_TEXT,
                      font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Prediction Gaussian (red) ──────────────────────────────────
        x_vals = np.linspace(-1, 7, 300)
        pred_curve = axes.plot(
            lambda x: gaussian_1d_pdf(np.array([x]), mu1, var1)[0],
            x_range=[-1, 7], color=COLOR_PREDICTION,
        )
        pred_label = MathTex(
            r"\mathcal{N}(\mu_1, \sigma_1^2)",
            color=COLOR_PREDICTION, font_size=SMALL_FONT_SIZE,
        )
        pred_label.next_to(axes.c2p(mu1, gaussian_1d_pdf(np.array([mu1]), mu1, var1)[0]),
                           UP, buff=0.2)

        pred_text = Text("Prediction", color=COLOR_PREDICTION,
                          font_size=SMALL_FONT_SIZE)
        pred_text.next_to(pred_label, UP, buff=0.15)

        self.play(Create(pred_curve), FadeIn(pred_label), FadeIn(pred_text),
                  run_time=NORMAL_ANIM)
        self.wait(PAUSE_SHORT)

        # ── Measurement Gaussian (blue) ────────────────────────────────
        meas_curve = axes.plot(
            lambda x: gaussian_1d_pdf(np.array([x]), mu2, var2)[0],
            x_range=[-1, 7], color=COLOR_MEASUREMENT,
        )
        meas_label = MathTex(
            r"\mathcal{N}(\mu_2, \sigma_2^2)",
            color=COLOR_MEASUREMENT, font_size=SMALL_FONT_SIZE,
        )
        meas_label.next_to(axes.c2p(mu2, gaussian_1d_pdf(np.array([mu2]), mu2, var2)[0]),
                           UP, buff=0.2)

        meas_text = Text("Measurement", color=COLOR_MEASUREMENT,
                          font_size=SMALL_FONT_SIZE)
        meas_text.next_to(meas_label, UP, buff=0.15)

        self.play(Create(meas_curve), FadeIn(meas_label), FadeIn(meas_text),
                  run_time=NORMAL_ANIM)
        self.wait(PAUSE_MEDIUM)

        # ── Derive product formulas ────────────────────────────────────
        formula_group = VGroup()

        mu_formula = MathTex(
            r"\mu_{\text{new}} = \frac{\sigma_2^2 \, \mu_1 + \sigma_1^2 \, \mu_2}"
            r"{\sigma_1^2 + \sigma_2^2}",
            font_size=BODY_FONT_SIZE, color=COLOR_EQUATION,
        )

        var_formula = MathTex(
            r"\sigma_{\text{new}}^2 = \frac{\sigma_1^2 \, \sigma_2^2}"
            r"{\sigma_1^2 + \sigma_2^2}",
            font_size=BODY_FONT_SIZE, color=COLOR_EQUATION,
        )

        formula_group = VGroup(mu_formula, var_formula).arrange(DOWN, buff=SMALL_BUFF)
        formula_group.to_edge(RIGHT, buff=0.4).shift(UP * 1.5)

        # Shrink axes to make room
        self.play(
            axes.animate.scale(0.65).to_edge(LEFT, buff=0.3).shift(DOWN * 0.3),
            pred_curve.animate.scale(0.65).to_edge(LEFT, buff=0.3).shift(DOWN * 0.3),
            meas_curve.animate.scale(0.65).to_edge(LEFT, buff=0.3).shift(DOWN * 0.3),
            FadeOut(pred_label), FadeOut(meas_label),
            FadeOut(pred_text), FadeOut(meas_text),
            FadeOut(x_label),
            run_time=NORMAL_ANIM,
        )

        self.play(Write(mu_formula), run_time=SLOW_ANIM)
        self.wait(PAUSE_SHORT)
        self.play(Write(var_formula), run_time=SLOW_ANIM)
        self.wait(PAUSE_MEDIUM)

        # ── Product Gaussian (gold) ────────────────────────────────────
        result_curve = axes.plot(
            lambda x: gaussian_1d_pdf(np.array([x]), mu_new, var_new)[0],
            x_range=[-1, 7], color=COLOR_POSTERIOR,
        )

        self.play(Create(result_curve), run_time=NORMAL_ANIM)

        insight = Text(
            "The product is always narrower\n"
            "than either input!",
            color=COLOR_POSTERIOR, font_size=SMALL_FONT_SIZE,
            line_spacing=1.2,
        )
        insight.next_to(var_formula, DOWN, buff=LARGE_BUFF)
        self.play(FadeIn(insight, shift=UP * 0.2), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG)

        # ── Rewrite with Kalman gain K ──────────────────────────────────
        self.play(FadeOut(insight), run_time=FAST_ANIM)

        kalman_gain_eq = MathTex(
            r"K = \frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2}",
            font_size=BODY_FONT_SIZE, color=COLOR_HIGHLIGHT,
        )
        kalman_rewrite = MathTex(
            r"\mu_{\text{new}} = \mu_1 + K \, (\mu_2 - \mu_1)",
            font_size=BODY_FONT_SIZE, color=COLOR_EQUATION,
        )

        kg_group = VGroup(kalman_gain_eq, kalman_rewrite).arrange(DOWN, buff=SMALL_BUFF)
        kg_group.next_to(formula_group, DOWN, buff=LARGE_BUFF)

        k_label = Text("Kalman Gain", color=COLOR_HIGHLIGHT,
                        font_size=SMALL_FONT_SIZE)
        k_label.next_to(kalman_gain_eq, UP, buff=0.15)

        self.play(FadeIn(k_label), Write(kalman_gain_eq), run_time=NORMAL_ANIM)
        self.wait(PAUSE_SHORT)
        self.play(Write(kalman_rewrite), run_time=NORMAL_ANIM)
        self.wait(PAUSE_MEDIUM)

        # ── K interpretation ────────────────────────────────────────────
        k_interp = VGroup(
            MathTex(r"K = 0", font_size=SMALL_FONT_SIZE, color=COLOR_PREDICTION),
            MathTex(r"\rightarrow", font_size=SMALL_FONT_SIZE, color=COLOR_TEXT),
            Text("trust prediction", font_size=SMALL_FONT_SIZE - 4,
                 color=COLOR_PREDICTION),
        ).arrange(RIGHT, buff=0.15)

        k_interp2 = VGroup(
            MathTex(r"K = 1", font_size=SMALL_FONT_SIZE, color=COLOR_MEASUREMENT),
            MathTex(r"\rightarrow", font_size=SMALL_FONT_SIZE, color=COLOR_TEXT),
            Text("trust measurement", font_size=SMALL_FONT_SIZE - 4,
                 color=COLOR_MEASUREMENT),
        ).arrange(RIGHT, buff=0.15)

        k_interps = VGroup(k_interp, k_interp2).arrange(DOWN, buff=0.2)
        k_interps.next_to(kg_group, DOWN, buff=STANDARD_BUFF)

        self.play(FadeIn(k_interps), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG * 2)

        # ── Fade out ───────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
