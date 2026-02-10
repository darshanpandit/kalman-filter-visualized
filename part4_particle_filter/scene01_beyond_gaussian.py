"""Part 4, Scene 1: Beyond Gaussians

Motivates the particle filter: shows a multimodal distribution
that KF/EKF/UKF can't represent. Pedestrian in a building
could be on floor 1 or floor 2.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.gaussian_ellipse import GaussianEllipse
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneBeyondGaussian(VoiceoverScene, Scene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        with self.voiceover(text="We've covered three Kalman Filter variants, all assuming Gaussian distributions. But what happens when that assumption breaks? Imagine a pedestrian entering a building, are they on floor 1 or floor 2?") as tracker:
            title = Text("Beyond Gaussians", color=COLOR_TEXT,
                          font_size=TITLE_FONT_SIZE)
            title.to_edge(UP, buff=0.3)
            self.play(Write(title), run_time=NORMAL_ANIM)

            # ── Problem setup ───────────────────────────────────────────────
            problem = Text(
                "A pedestrian enters a building...\n"
                "They could be on floor 1 or floor 2.",
                color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
                line_spacing=1.2,
            )
            problem.next_to(title, DOWN, buff=STANDARD_BUFF)
            self.play(FadeIn(problem), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Show multimodal distribution ────────────────────────────────
        with self.voiceover(text="Your belief is bimodal: maybe 60 percent chance on floor 1, 40 percent on floor 2. Each mode is roughly Gaussian, but the overall distribution is not.") as tracker:
            self.play(FadeOut(problem), run_time=FAST_ANIM)

            axes = Axes(
                x_range=[-4, 4, 1], y_range=[-1, 5, 1],
                x_length=8, y_length=5,
                axis_config={"color": COLOR_GRID},
            ).shift(DOWN * 0.3)
            x_label = axes.get_x_axis_label(
                MathTex(r"\text{position}", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE))
            y_label = axes.get_y_axis_label(
                MathTex(r"\text{floor}", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE))

            self.play(Create(axes), FadeIn(x_label), FadeIn(y_label),
                      run_time=NORMAL_ANIM)

            # Two possible locations
            mean1 = np.array([1.0, 1.0])
            cov1 = np.array([[0.5, 0], [0, 0.1]])
            ellipse1 = GaussianEllipse(
                mean=mean1, cov=cov1, color=COLOR_PREDICTION, axes=axes,
            )
            label1 = Text("Floor 1", color=COLOR_PREDICTION, font_size=SMALL_FONT_SIZE)
            label1.next_to(axes.c2p(mean1[0], mean1[1]), RIGHT, buff=0.5)

            mean2 = np.array([-0.5, 3.0])
            cov2 = np.array([[0.5, 0], [0, 0.1]])
            ellipse2 = GaussianEllipse(
                mean=mean2, cov=cov2, color=COLOR_MEASUREMENT, axes=axes,
            )
            label2 = Text("Floor 2", color=COLOR_MEASUREMENT, font_size=SMALL_FONT_SIZE)
            label2.next_to(axes.c2p(mean2[0], mean2[1]), RIGHT, buff=0.5)

            self.play(FadeIn(ellipse1), FadeIn(label1), run_time=NORMAL_ANIM)
            self.play(FadeIn(ellipse2), FadeIn(label2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Show KF fails: single Gaussian can't represent this ─────────
        with self.voiceover(text="A single Gaussian would put the mean between the floors, physically impossible. We need particles: weighted samples that can represent any distribution shape.") as tracker:
            # A single Gaussian would be in between (wrong!)
            mean_bad = (mean1 + mean2) / 2
            cov_bad = np.array([[1.5, 0], [0, 2.0]])
            bad_ellipse = GaussianEllipse(
                mean=mean_bad, cov=cov_bad, color=PURE_RED, axes=axes,
                fill_opacity=0.1,
            )
            bad_label = Text("Single Gaussian?\nWrong!", color=PURE_RED,
                              font_size=SMALL_FONT_SIZE)
            bad_label.next_to(axes.c2p(mean_bad[0], mean_bad[1]), LEFT, buff=0.5)

            self.play(FadeIn(bad_ellipse), FadeIn(bad_label), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

            # ── Solution ────────────────────────────────────────────────────
            solution = Text(
                "Solution: Represent the distribution with particles\n"
                "— no shape assumption needed!",
                color=COLOR_POSTERIOR, font_size=BODY_FONT_SIZE,
                line_spacing=1.2,
            )
            solution.to_edge(DOWN, buff=0.3)
            self.play(FadeIn(solution), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
