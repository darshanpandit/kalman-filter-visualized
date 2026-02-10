"""Scene 8: Optimality & Teaser

States the optimality result: KF is the MMSE optimal linear estimator.
Brief proof sketch, then teases the next video (EKF for nonlinear systems).
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *


class SceneOptimality(VoiceoverScene, Scene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────────
        title = Text("Why the Kalman Filter is Optimal",
                      color=COLOR_TEXT, font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3)

        # ── MMSE statement ──────────────────────────────────────────────
        theorem = VGroup(
            Text("Theorem", color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE),
            Text(
                "Among all linear estimators, the Kalman Filter",
                color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
            ),
            Text(
                "minimizes the mean squared estimation error:",
                color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
            ),
        ).arrange(DOWN, buff=0.2)
        theorem.next_to(title, DOWN, buff=STANDARD_BUFF)

        with self.voiceover(text="The Kalman Filter isn't just good — it's provably optimal. Among all linear estimators, it minimizes the mean squared error.") as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(theorem), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

        # ── Objective function ──────────────────────────────────────────
        objective = MathTex(
            r"\min_{\hat{\mathbf{x}}} \; "
            r"\mathbb{E}\left[ "
            r"(\mathbf{x} - \hat{\mathbf{x}})^T "
            r"(\mathbf{x} - \hat{\mathbf{x}}) "
            r"\right]",
            font_size=EQUATION_FONT_SIZE, color=COLOR_EQUATION,
        )
        objective.next_to(theorem, DOWN, buff=STANDARD_BUFF)

        with self.voiceover(text="We want to minimize the expected squared difference between the true state and our estimate — the trace of the error covariance.") as tracker:
            self.play(Write(objective), run_time=SLOW_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Proof sketch ────────────────────────────────────────────────
        proof_title = Text("Proof sketch:", color=COLOR_TEXT,
                            font_size=BODY_FONT_SIZE)
        proof_title.next_to(objective, DOWN, buff=LARGE_BUFF)
        proof_title.align_to(objective, LEFT)

        step1 = MathTex(
            r"\text{1. Set } \frac{\partial}{\partial \mathbf{K}} "
            r"\text{tr}(\mathbf{P}_k) = 0",
            font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION,
        )
        step2 = MathTex(
            r"\text{2. Solve for } \mathbf{K} \text{ that minimizes tr}(\mathbf{P}_k)",
            font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION,
        )
        step3 = MathTex(
            r"\text{3. } \Rightarrow \mathbf{K} = "
            r"\mathbf{P}_k^{-} \mathbf{H}^T "
            r"(\mathbf{H} \mathbf{P}_k^{-} \mathbf{H}^T + \mathbf{R})^{-1}",
            font_size=SMALL_FONT_SIZE, color=COLOR_HIGHLIGHT,
        )

        proof_steps = VGroup(step1, step2, step3).arrange(DOWN, buff=0.2,
                                                           aligned_edge=LEFT)
        proof_steps.next_to(proof_title, DOWN, buff=SMALL_BUFF)
        proof_steps.align_to(proof_title, LEFT)

        with self.voiceover(text="The proof is elegant: take the derivative of the error covariance with respect to K, set it to zero, and solve. You get exactly the Kalman gain formula.") as tracker:
            self.play(FadeIn(proof_title), run_time=FAST_ANIM)
            for step in proof_steps:
                self.play(Write(step), run_time=NORMAL_ANIM)
                self.wait(PAUSE_SHORT)
            self.wait(PAUSE_LONG)

        # ── Conditions ──────────────────────────────────────────────────
        conditions = VGroup(
            Text("This holds when:", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
            MathTex(r"\bullet \text{ System is linear } (\mathbf{F}, \mathbf{H} \text{ are matrices})",
                    font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION),
            MathTex(r"\bullet \text{ Noise is Gaussian } (\mathbf{w}_k, \mathbf{v}_k \sim \mathcal{N})",
                    font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        conditions.to_edge(DOWN, buff=0.4)

        with self.voiceover(text="But this optimality requires two conditions: linear dynamics and Gaussian noise. When both hold, no estimator can beat the Kalman Filter.") as tracker:
            self.play(FadeIn(conditions), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Teaser for next video ───────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)

        teaser = Text(
            "But what if the system isn't linear?",
            color=COLOR_HIGHLIGHT, font_size=TITLE_FONT_SIZE,
        )
        next_ep = Text(
            "Next: The Extended Kalman Filter",
            color=COLOR_TEXT, font_size=HEADING_FONT_SIZE,
        )
        next_ep.next_to(teaser, DOWN, buff=LARGE_BUFF)

        with self.voiceover(text="But what if the system isn't linear? What if the pedestrian turns corners or the sensor model involves trigonometry? That's where the Extended Kalman Filter comes in. See you in Part 2.") as tracker:
            self.play(FadeIn(teaser, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)
            self.play(FadeIn(next_ep, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

        self.play(FadeOut(teaser), FadeOut(next_ep), run_time=NORMAL_ANIM)
