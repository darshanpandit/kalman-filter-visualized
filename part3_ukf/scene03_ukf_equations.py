"""Part 3, Scene 3: UKF Equations and Tuning Parameters

Shows the UKF algorithm steps and the meaning of alpha, beta, kappa.
"""

from __future__ import annotations

from manim import *
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneUKFEquations(VoiceoverScene, Scene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        title = Text("UKF Algorithm", color=COLOR_TEXT,
                      font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Algorithm steps ─────────────────────────────────────────────
        steps = VGroup(
            MathTex(r"\textbf{1. Generate sigma points:}",
                    font_size=SMALL_FONT_SIZE, color=COLOR_HIGHLIGHT),
            MathTex(r"\mathcal{X}_i \text{ from } (\hat{\mathbf{x}}, \mathbf{P}, \alpha, \kappa)",
                    font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION),
            MathTex(r"\textbf{2. Predict:}",
                    font_size=SMALL_FONT_SIZE, color=COLOR_HIGHLIGHT),
            MathTex(r"\mathcal{Y}_i = f(\mathcal{X}_i), \quad "
                    r"\hat{\mathbf{x}}^- = \sum W_i^m \mathcal{Y}_i",
                    font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION),
            MathTex(r"\mathbf{P}^- = \sum W_i^c (\mathcal{Y}_i - \hat{\mathbf{x}}^-)"
                    r"(\mathcal{Y}_i - \hat{\mathbf{x}}^-)^T + \mathbf{Q}",
                    font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION),
            MathTex(r"\textbf{3. Update:}",
                    font_size=SMALL_FONT_SIZE, color=COLOR_HIGHLIGHT),
            MathTex(r"\mathcal{Z}_i = h(\mathcal{X}_i^-), \quad "
                    r"\hat{\mathbf{z}} = \sum W_i^m \mathcal{Z}_i",
                    font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION),
            MathTex(r"\mathbf{S} = \sum W_i^c (\mathcal{Z}_i - \hat{\mathbf{z}})"
                    r"(\mathcal{Z}_i - \hat{\mathbf{z}})^T + \mathbf{R}",
                    font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION),
            MathTex(r"\mathbf{P}_{xz} = \sum W_i^c (\mathcal{Y}_i - \hat{\mathbf{x}}^-)"
                    r"(\mathcal{Z}_i - \hat{\mathbf{z}})^T",
                    font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION),
            MathTex(r"\mathbf{K} = \mathbf{P}_{xz} \mathbf{S}^{-1}, \quad "
                    r"\hat{\mathbf{x}} = \hat{\mathbf{x}}^- + \mathbf{K}(\mathbf{z} - \hat{\mathbf{z}})",
                    font_size=SMALL_FONT_SIZE, color=COLOR_POSTERIOR),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        steps.next_to(title, DOWN, buff=STANDARD_BUFF).shift(LEFT * 0.5)

        with self.voiceover(text="The full UKF algorithm. Step one: generate 2n plus 1 sigma points from the current estimate using the Cholesky decomposition of the covariance.") as tracker:
            for step in steps[:2]:
                self.play(Write(step), run_time=NORMAL_ANIM * 0.7)

        with self.voiceover(text="Step two: push each sigma point through the process model, then compute the weighted mean and covariance. Add process noise Q to get the predicted uncertainty.") as tracker:
            for step in steps[2:5]:
                self.play(Write(step), run_time=NORMAL_ANIM * 0.7)

        with self.voiceover(text="Step three: transform sigma points through the measurement model. Compute the innovation covariance S and the cross-covariance P x z between state and measurement predictions. Then compute the Kalman gain and update.") as tracker:
            for step in steps[5:]:
                self.play(Write(step), run_time=NORMAL_ANIM * 0.7)

        self.wait(PAUSE_MEDIUM)

        # ── Tuning parameters ───────────────────────────────────────────
        with self.voiceover(text="The tuning parameters alpha, beta, and kappa control the spread and weighting of sigma points. Alpha between 0 and 1 sets the spread, beta equals 2 is optimal for Gaussians, and kappa is secondary scaling.") as tracker:
            params = VGroup(
                Text("Tuning Parameters:", color=COLOR_HIGHLIGHT,
                      font_size=BODY_FONT_SIZE),
                MathTex(r"\alpha \in (0, 1]", r"\text{ — spread of sigma points}",
                        font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION),
                MathTex(r"\beta = 2", r"\text{ — optimal for Gaussian}",
                        font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION),
                MathTex(r"\kappa \geq 0", r"\text{ — secondary scaling}",
                        font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION),
            ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            params.to_edge(DOWN, buff=0.3)

            self.play(FadeIn(params), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
