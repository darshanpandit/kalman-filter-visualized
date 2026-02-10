"""Scene 5: Prediction Step

Derives and visualizes the Kalman prediction step:
  x̂_k⁻ = F · x̂_{k-1}
  P_k⁻  = F · P_{k-1} · F^T + Q

Shows the state transition matrix F shearing the ellipse, then process noise
Q expanding it.
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.gaussian_ellipse import GaussianEllipse
from kalman_manim.mobjects.state_space import StateSpace


class ScenePrediction(VoiceoverScene, Scene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────────
        title = Text("Prediction Step", color=COLOR_TEXT,
                      font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3)

        # ── State transition equation ───────────────────────────────────
        state_eq = MathTex(
            r"\hat{\mathbf{x}}_k^{-}",
            r"=",
            r"\mathbf{F}",
            r"\cdot",
            r"\hat{\mathbf{x}}_{k-1}",
            font_size=EQUATION_FONT_SIZE, color=COLOR_EQUATION,
        )
        state_eq[0].set_color(COLOR_PREDICTION)
        state_eq[2].set_color(COLOR_HIGHLIGHT)

        cov_eq = MathTex(
            r"\mathbf{P}_k^{-}",
            r"=",
            r"\mathbf{F}",
            r"\cdot",
            r"\mathbf{P}_{k-1}",
            r"\cdot",
            r"\mathbf{F}^T",
            r"+",
            r"\mathbf{Q}",
            font_size=EQUATION_FONT_SIZE, color=COLOR_EQUATION,
        )
        cov_eq[0].set_color(COLOR_PREDICTION)
        cov_eq[2].set_color(COLOR_HIGHLIGHT)
        cov_eq[6].set_color(COLOR_HIGHLIGHT)
        cov_eq[8].set_color(COLOR_PROCESS_NOISE)

        eq_group = VGroup(state_eq, cov_eq).arrange(DOWN, buff=SMALL_BUFF)
        eq_group.next_to(title, DOWN, buff=STANDARD_BUFF)

        with self.voiceover(text="Every Kalman Filter cycle begins with prediction. The state evolves through matrix F, and uncertainty grows through F P F-transpose plus Q.") as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.play(Write(state_eq), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(Write(cov_eq), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Show F matrix for pedestrian ────────────────────────────────
        dt_val = 1.0
        F_display = MathTex(
            r"\mathbf{F} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}",
            font_size=BODY_FONT_SIZE, color=COLOR_HIGHLIGHT,
        )
        F_note = Text(
            "Position updates with velocity",
            font_size=SMALL_FONT_SIZE, color=COLOR_TEXT,
        )
        F_group = VGroup(F_display, F_note).arrange(DOWN, buff=0.15)
        F_group.to_edge(RIGHT, buff=0.4).shift(UP * 0.5)

        with self.voiceover(text="For a pedestrian, F encodes constant-velocity motion: new position equals old position plus velocity times delta-t.") as tracker:
            self.play(Write(F_display), FadeIn(F_note), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

        # ── State space with ellipse ────────────────────────────────────
        ss = StateSpace(
            x_range=[-2, 6, 1], y_range=[-2, 3, 1],
            x_length=6, y_length=4,
            x_label=r"\text{pos}", y_label=r"\text{vel}",
        )
        ss.shift(DOWN * 1.2 + LEFT * 1.5)

        # Initial state
        mean0 = np.array([1.0, 1.0])
        P0 = np.array([[0.5, 0.2], [0.2, 0.3]])
        ellipse = GaussianEllipse(
            mean=mean0, cov=P0,
            color=COLOR_POSTERIOR, axes=ss.axes,
        )

        step_label = Text("Current estimate", color=COLOR_POSTERIOR,
                           font_size=SMALL_FONT_SIZE)
        step_label.to_edge(DOWN, buff=0.4)

        with self.voiceover(text="Here's our current estimate as an ellipse in state space. The center is our best guess, the shape captures uncertainty.") as tracker:
            self.play(Create(ss.axes), FadeIn(ss.x_label_mob), FadeIn(ss.y_label_mob),
                      run_time=NORMAL_ANIM)
            self.play(FadeIn(ellipse), run_time=NORMAL_ANIM)
            self.play(FadeIn(step_label), run_time=FAST_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Apply F: state transition ───────────────────────────────────
        F = np.array([[1, dt_val], [0, 1]])
        mean_pred = F @ mean0  # [1 + 1*1, 1] = [2, 1]
        P_after_F = F @ P0 @ F.T  # F P F^T (no Q yet)

        f_step_label = Text("Apply F: ellipse shears", color=COLOR_HIGHLIGHT,
                             font_size=SMALL_FONT_SIZE)
        f_step_label.to_edge(DOWN, buff=0.4)

        with self.voiceover(text="Now apply F. The ellipse shears — the state transition couples position and velocity. If velocity is high, position moves farther.") as tracker:
            self.play(FadeOut(step_label), run_time=FAST_ANIM)
            self.play(FadeIn(f_step_label), run_time=FAST_ANIM)
            self.play(
                ellipse.animate_to(mean_pred, P_after_F),
                run_time=SLOW_ANIM,
            )
            self.wait(PAUSE_MEDIUM)

        # ── Add process noise Q ─────────────────────────────────────────
        Q = np.array([[0.3, 0.0], [0.0, 0.2]])
        P_pred = P_after_F + Q

        q_label = Text("Add Q: uncertainty grows", color=COLOR_PROCESS_NOISE,
                        font_size=SMALL_FONT_SIZE)
        q_label.to_edge(DOWN, buff=0.4)

        # Change color to prediction (Swiss red)
        ellipse_pred = GaussianEllipse(
            mean=mean_pred, cov=P_pred,
            color=COLOR_PREDICTION, axes=ss.axes,
        )

        with self.voiceover(text="But motion isn't perfectly predictable. Process noise Q inflates the ellipse — we're now less certain than before.") as tracker:
            self.play(FadeOut(f_step_label), FadeIn(q_label), run_time=FAST_ANIM)
            self.play(Transform(ellipse, ellipse_pred), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        result_label = Text("Predicted state", color=COLOR_PREDICTION,
                             font_size=SMALL_FONT_SIZE)
        result_label.to_edge(DOWN, buff=0.4)

        with self.voiceover(text="The result is our predicted state in red — our best guess before seeing the next measurement.") as tracker:
            self.play(FadeOut(q_label), FadeIn(result_label), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG * 2)

        # ── Fade out ───────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
