"""Scene 4: State Space — Position + Velocity

Introduces the state vector x = [position, velocity]^T for a pedestrian.
Shows 2D state space with covariance ellipses, explains how correlation
rotates the ellipse.
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


class SceneStateSpace(VoiceoverScene, Scene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────────
        title = Text("The State Vector", color=COLOR_TEXT,
                      font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3)

        # ── State vector definition ─────────────────────────────────────
        state_eq = MathTex(
            r"\mathbf{x} = \begin{bmatrix} \text{position} \\ \text{velocity} \end{bmatrix}",
            font_size=EQUATION_FONT_SIZE, color=COLOR_EQUATION,
        )
        state_eq.next_to(title, DOWN, buff=STANDARD_BUFF)

        with self.voiceover(text="For a moving pedestrian, position alone isn't enough. We also need velocity to predict where they'll be next. So we bundle both into a state vector.") as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.play(Write(state_eq), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Move equation to top-right, build state space ───────────────
        ss = StateSpace(
            x_range=[-4, 4, 1], y_range=[-3, 3, 1],
            x_length=8, y_length=5,
            x_label=r"\text{position}", y_label=r"\text{velocity}",
        )
        ss.shift(DOWN * 0.3 + LEFT * 0.5)

        # ── Single point estimate ───────────────────────────────────────
        mean = np.array([1.0, 0.5])
        point = Dot(ss.c2p(mean[0], mean[1]), color=COLOR_POSTERIOR,
                     radius=DOT_RADIUS_MEDIUM)
        point_label = MathTex(
            r"\hat{\mathbf{x}} = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}",
            font_size=SMALL_FONT_SIZE, color=COLOR_POSTERIOR,
        )
        point_label.next_to(point, UR, buff=0.2)

        with self.voiceover(text="This gives us a two-dimensional state space. Every point represents a possible state — a position and a velocity.") as tracker:
            self.play(
                state_eq.animate.scale(0.7).to_corner(UR, buff=0.3),
                run_time=NORMAL_ANIM,
            )
            self.play(Create(ss.axes), FadeIn(ss.x_label_mob), FadeIn(ss.y_label_mob),
                      run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(point, scale=1.5), Write(point_label), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

        note = Text(
            "But how certain are we?",
            color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
        )
        note.to_edge(DOWN, buff=0.5)

        # ── Uncorrelated covariance (axis-aligned ellipse) ──────────────
        cov_uncorr = np.array([[1.0, 0.0], [0.0, 0.5]])
        ellipse1 = GaussianEllipse(
            mean=mean, cov=cov_uncorr,
            color=COLOR_POSTERIOR, axes=ss.axes,
            show_axes=True,
        )

        cov_label = MathTex(
            r"\mathbf{P} = \begin{bmatrix} 1.0 & 0 \\ 0 & 0.5 \end{bmatrix}",
            font_size=SMALL_FONT_SIZE, color=COLOR_POSTERIOR,
        )
        cov_label.next_to(state_eq, DOWN, buff=STANDARD_BUFF)

        uncorr_note = Text("No correlation: axis-aligned", color=COLOR_TEXT,
                            font_size=SMALL_FONT_SIZE)
        uncorr_note.to_edge(DOWN, buff=0.5)

        with self.voiceover(text="But how certain are we? The covariance matrix P captures this uncertainty as an ellipse. When there's no correlation, the ellipse is axis-aligned.") as tracker:
            self.play(FadeIn(note), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)
            self.play(FadeOut(note), FadeOut(point_label), run_time=FAST_ANIM)
            self.play(FadeIn(ellipse1), Write(cov_label), run_time=NORMAL_ANIM)
            self.play(FadeIn(uncorr_note), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # ── Correlated covariance (rotated ellipse) ─────────────────────
        cov_corr = np.array([[1.0, 0.7], [0.7, 0.5]])
        ellipse2 = GaussianEllipse(
            mean=mean, cov=cov_corr,
            color=COLOR_HIGHLIGHT, axes=ss.axes,
            show_axes=True,
        )

        cov_label2 = MathTex(
            r"\mathbf{P} = \begin{bmatrix} 1.0 & 0.7 \\ 0.7 & 0.5 \end{bmatrix}",
            font_size=SMALL_FONT_SIZE, color=COLOR_HIGHLIGHT,
        )
        cov_label2.move_to(cov_label)

        corr_note = Text(
            "Correlation: tilted ellipse\n"
            "\"If position is ahead, velocity is probably higher\"",
            color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
            line_spacing=1.2,
        )
        corr_note.to_edge(DOWN, buff=0.4)

        with self.voiceover(text="But when position and velocity are correlated — if the pedestrian is ahead of expected, they're probably moving faster — the off-diagonal term tilts the ellipse. This geometric shape encodes everything about our uncertainty.") as tracker:
            self.play(
                FadeOut(ellipse1), FadeOut(uncorr_note),
                FadeIn(ellipse2),
                Transform(cov_label, cov_label2),
                run_time=NORMAL_ANIM,
            )
            self.play(FadeIn(corr_note), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

        # ── Fade out ───────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
