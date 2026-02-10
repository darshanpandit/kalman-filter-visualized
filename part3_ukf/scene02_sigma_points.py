"""Part 3, Scene 2: Sigma Points — Generation, Transformation, Recombination

The core visual of the UKF: sigma points placed on the covariance ellipse,
transformed through a nonlinear function, then recombined into a new Gaussian.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.gaussian_ellipse import GaussianEllipse
from kalman_manim.mobjects.sigma_points import SigmaPointCloud
from kalman_manim.utils import cov_to_ellipse_params
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneSigmaPoints(VoiceoverScene, Scene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        with self.voiceover(text="Here's the core of the UKF: the unscented transform. On the left is input space, on the right is output space, connected by a nonlinear function f.") as tracker:
            title = Text("Sigma Points", color=COLOR_TEXT,
                          font_size=TITLE_FONT_SIZE)
            title.to_edge(UP, buff=0.3)
            self.play(Write(title), run_time=NORMAL_ANIM)

            # ── Input space (left) ──────────────────────────────────────────
            in_axes = Axes(
                x_range=[-3, 3, 1], y_range=[-3, 3, 1],
                x_length=4.5, y_length=4.5,
                axis_config={"color": COLOR_GRID},
            ).shift(LEFT * 3.5 + DOWN * 0.5)
            in_label = Text("Input space", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE)
            in_label.next_to(in_axes, DOWN, buff=0.2)

            # ── Output space (right) ────────────────────────────────────────
            out_axes = Axes(
                x_range=[-3, 5, 1], y_range=[-3, 5, 1],
                x_length=4.5, y_length=4.5,
                axis_config={"color": COLOR_GRID},
            ).shift(RIGHT * 3.5 + DOWN * 0.5)
            out_label = Text("Output space", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE)
            out_label.next_to(out_axes, DOWN, buff=0.2)

            # Arrow between spaces
            transform_arrow = Arrow(
                in_axes.get_right() + RIGHT * 0.2,
                out_axes.get_left() + LEFT * 0.2,
                color=COLOR_HIGHLIGHT, stroke_width=3,
            )
            transform_label = MathTex(r"f(\cdot)", color=COLOR_HIGHLIGHT,
                                       font_size=BODY_FONT_SIZE)
            transform_label.next_to(transform_arrow, UP, buff=0.1)

            self.play(
                Create(in_axes), Create(out_axes),
                FadeIn(in_label), FadeIn(out_label),
                Create(transform_arrow), FadeIn(transform_label),
                run_time=NORMAL_ANIM,
            )

        # ── Step 1: Show Gaussian in input space ────────────────────────
        with self.voiceover(text="Start with a Gaussian distribution. Instead of using the whole continuous distribution, we pick 2n plus 1 sigma points, placed deterministically to match the mean and covariance exactly.") as tracker:
            step1 = Text("1. Start with Gaussian", color=COLOR_TEXT,
                          font_size=SMALL_FONT_SIZE)
            step1.to_corner(UL, buff=0.3)
            self.play(FadeIn(step1), run_time=FAST_ANIM)

            mean = np.array([0.5, 0.5])
            cov = np.array([[0.8, 0.3], [0.3, 0.6]])
            ellipse_in = GaussianEllipse(
                mean=mean, cov=cov,
                color=COLOR_PREDICTION, axes=in_axes,
                fill_opacity=0.15,
            )
            self.play(FadeIn(ellipse_in), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

            # ── Step 2: Place sigma points ──────────────────────────────────
            step2 = Text("2. Place sigma points", color=COLOR_HIGHLIGHT,
                          font_size=SMALL_FONT_SIZE)
            step2.next_to(step1, DOWN, buff=0.15)
            self.play(FadeIn(step2), run_time=FAST_ANIM)

            sigma_cloud = SigmaPointCloud(
                mean=mean, cov=cov,
                alpha=0.5, kappa=0.0,
                color=COLOR_HIGHLIGHT,
                center_color=COLOR_POSTERIOR,
                axes=in_axes,
            )
            self.play(FadeIn(sigma_cloud, lag_ratio=0.1), run_time=NORMAL_ANIM)

            n_points_label = MathTex(r"2n + 1 = 5 \text{ points}",
                                      color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE)
            n_points_label.next_to(step2, DOWN, buff=0.15)
            self.play(FadeIn(n_points_label), run_time=FAST_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Step 3: Transform each point through f ──────────────────────
        with self.voiceover(text="Now push each sigma point through the actual nonlinear function. No approximation, just evaluate f at each point. Watch them land in the output space.") as tracker:
            step3 = Text("3. Transform each point", color=COLOR_POSTERIOR,
                          font_size=SMALL_FONT_SIZE)
            step3.next_to(n_points_label, DOWN, buff=0.15)
            self.play(FadeIn(step3), run_time=FAST_ANIM)

            # Nonlinear function
            def nonlinear_f(xy):
                x, y = xy
                return np.array([
                    x + 0.5 * y**2,
                    y + 0.3 * np.sin(x * 2),
                ])

            transformed_cloud = sigma_cloud.get_transformed_cloud(
                nonlinear_f, color=COLOR_POSTERIOR, axes=out_axes,
            )

            # Animate each point flying to its transformed position
            self.play(
                LaggedStart(
                    *[FadeIn(d, scale=1.3) for d in transformed_cloud],
                    lag_ratio=0.15,
                ),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_SHORT)

        # ── Step 4: Fit new Gaussian ────────────────────────────────────
        with self.voiceover(text="Finally, compute the weighted mean and covariance of the transformed points to get the output Gaussian. This captures the nonlinear transformation without any linearization, and it's accurate to second order.") as tracker:
            step4 = Text("4. Fit new Gaussian", color=COLOR_POSTERIOR,
                          font_size=SMALL_FONT_SIZE)
            step4.next_to(step3, DOWN, buff=0.15)
            self.play(FadeIn(step4), run_time=FAST_ANIM)

            # Compute transformed mean and covariance
            transformed_pts = np.array([nonlinear_f(sp) for sp in sigma_cloud.sigma_points])
            Wm = sigma_cloud.weights
            t_mean = np.sum(Wm[:, None] * transformed_pts, axis=0)
            t_cov = np.zeros((2, 2))
            for i in range(len(transformed_pts)):
                d = transformed_pts[i] - t_mean
                t_cov += Wm[i] * np.outer(d, d)

            ellipse_out = GaussianEllipse(
                mean=t_mean, cov=t_cov,
                color=COLOR_POSTERIOR, axes=out_axes,
                fill_opacity=0.15,
            )
            self.play(FadeIn(ellipse_out), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

            result_text = Text(
                "Sigma points capture the nonlinear transformation\n"
                "without any linearization!",
                color=COLOR_POSTERIOR, font_size=BODY_FONT_SIZE,
                line_spacing=1.2,
            )
            result_text.to_edge(DOWN, buff=0.2)
            self.play(FadeIn(result_text), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
