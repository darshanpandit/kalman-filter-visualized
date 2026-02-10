"""Part 3, Scene 1: The Key UKF Insight

"It's easier to approximate a probability distribution than a nonlinear function."
Introduces the unscented transform concept.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneUKFInsight(VoiceoverScene, Scene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────────
        with self.voiceover(text="The Unscented Kalman Filter starts with a profound insight from Jeffrey Uhlmann: it is easier to approximate a probability distribution than to approximate a nonlinear function.") as tracker:
            title = Text("The Unscented Kalman Filter", color=COLOR_TEXT,
                          font_size=TITLE_FONT_SIZE)
            title.to_edge(UP, buff=0.3)
            self.play(Write(title), run_time=NORMAL_ANIM)

            # ── Key quote ───────────────────────────────────────────────────
            quote = Text(
                "\"It is easier to approximate a probability distribution\n"
                " than to approximate a nonlinear function.\"",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
                line_spacing=1.3,
            )
            quote.next_to(title, DOWN, buff=LARGE_BUFF)
            self.play(FadeIn(quote, shift=UP * 0.2), run_time=NORMAL_ANIM)

            attribution = Text("— Jeffrey Uhlmann", color=COLOR_TEXT,
                                 font_size=SMALL_FONT_SIZE)
            attribution.next_to(quote, DOWN, buff=SMALL_BUFF).align_to(quote, RIGHT)
            self.play(FadeIn(attribution), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # ── EKF approach (left) ─────────────────────────────────────────
        with self.voiceover(text="The EKF linearizes the function, replacing a curve with its tangent. The UKF takes a different approach: pick representative sample points, transform them through the actual function, and fit a new Gaussian to the results.") as tracker:
            self.play(FadeOut(quote), FadeOut(attribution), run_time=FAST_ANIM)

            ekf_title = Text("EKF approach:", color=COLOR_PREDICTION,
                              font_size=HEADING_FONT_SIZE)
            ekf_desc = Text(
                "Linearize the function,\n"
                "then push the Gaussian through",
                color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
                line_spacing=1.2,
            )
            ekf_group = VGroup(ekf_title, ekf_desc).arrange(DOWN, buff=0.2)
            ekf_group.shift(LEFT * 3 + DOWN * 0.5)

            ukf_title = Text("UKF approach:", color=COLOR_POSTERIOR,
                              font_size=HEADING_FONT_SIZE)
            ukf_desc = Text(
                "Pick representative points,\n"
                "push them through the function,\n"
                "then fit a new Gaussian",
                color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
                line_spacing=1.2,
            )
            ukf_group = VGroup(ukf_title, ukf_desc).arrange(DOWN, buff=0.2)
            ukf_group.shift(RIGHT * 3 + DOWN * 0.5)

            vs = Text("vs", color=COLOR_TEXT, font_size=HEADING_FONT_SIZE)

            self.play(FadeIn(ekf_group), FadeIn(vs), FadeIn(ukf_group),
                      run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Advantage highlight ─────────────────────────────────────────
        with self.voiceover(text="No Jacobians needed, no linearization error. The UKF approximates the distribution, not the function, and it does so more accurately.") as tracker:
            advantage = Text(
                "No Jacobians needed! No linearization error!",
                color=COLOR_POSTERIOR, font_size=BODY_FONT_SIZE,
            )
            advantage.to_edge(DOWN, buff=0.5)
            self.play(FadeIn(advantage, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
