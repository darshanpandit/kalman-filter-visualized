"""Part 9, Scene 1: The Prediction Step Revisited.

Data: Conceptual diagrams + comparison table

From the classical Kalman filter's F*x to continuous dx/dt = Ax to
learned neural dynamics f_theta(x). Shows the evolution of the
prediction step across the entire series.

Papers:
- Chen et al. (2018, NeurIPS Best Paper) — Neural ODEs
- Hafner et al. (2019, ICLR) — RSSM / PlaNet
- Greydanus et al. (2019, NeurIPS) — Hamiltonian Neural Networks
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.comparison_table import ComparisonTable
from kalman_manim.mobjects.observation_note import make_observation_note
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class ScenePredictionRevisited(VoiceoverScene, MovingCameraScene):
    """The prediction step revisited: from F*x to f_theta(x).

    Visual: Classical discretization animation, then comparison table of
    prediction models across the series.
    References: Chen et al. (2018), Hafner et al. (2019), Greydanus (2019).
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # -- Title --------------------------------------------------------
        with self.voiceover(
            text="Every filter we've studied has a prediction step. "
                 "In this part, we ask: what if we replace the hand-crafted "
                 "dynamics model with something learned from data?"
        ) as tracker:
            title = Text(
                "The Prediction Step Revisited",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # -- Euler discretization -----------------------------------------
        with self.voiceover(
            text="The Kalman filter's prediction is a single matrix multiply: "
                 "x predicted equals F times x. This is Euler discretization "
                 "of the continuous dynamics dx dt equals A times x."
        ) as tracker:
            eq_discrete = Text(
                "KF:  x' = F * x     (discrete, linear)",
                color=COLOR_PREDICTION, font_size=BODY_FONT_SIZE,
            )
            eq_continuous = Text(
                "Continuous:  dx/dt = A * x",
                color=TEAL, font_size=BODY_FONT_SIZE,
            )
            eq_learned = Text(
                "Learned:  dx/dt = f_theta(x)",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            )

            equations = VGroup(eq_discrete, eq_continuous, eq_learned)
            equations.arrange(DOWN, buff=0.5)
            equations.next_to(title, DOWN, buff=0.8)

            self.play(Write(eq_discrete), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

        # -- Continuous dynamics ------------------------------------------
        with self.voiceover(
            text="When we move to continuous time, we write dx dt equals A x. "
                 "The EKF extends this to a nonlinear function f of x. But "
                 "both require you to know f analytically."
        ) as tracker:
            self.play(Write(eq_continuous), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

        # -- Learned dynamics ---------------------------------------------
        with self.voiceover(
            text="The breakthrough idea: replace f with a neural network "
                 "f theta, trained end-to-end from data. The network learns "
                 "the dynamics model that the engineer used to hand-craft."
        ) as tracker:
            self.play(Write(eq_learned), run_time=NORMAL_ANIM)

        self.wait(PAUSE_MEDIUM)

        # -- Pipeline boxes -----------------------------------------------
        with self.voiceover(
            text="Let's visualize the evolution. The classical approach "
                 "requires a known model: F, H, Q, R. The neural approach "
                 "learns the transition function directly from trajectories."
        ) as tracker:
            self.play(FadeOut(equations), run_time=FAST_ANIM)

            # Classical box
            classical_box = RoundedRectangle(
                corner_radius=0.1, width=3.0, height=0.8,
                color=COLOR_PREDICTION, fill_opacity=0.2,
            )
            classical_label = Text(
                "F * x + B * u", color=COLOR_PREDICTION, font_size=20,
            )
            classical_label.move_to(classical_box)
            classical = VGroup(classical_box, classical_label)

            classical_tag = Text(
                "Known model", color=SLATE, font_size=16,
            )
            classical_tag.next_to(classical, DOWN, buff=0.15)

            # Learned box
            learned_box = RoundedRectangle(
                corner_radius=0.1, width=3.0, height=0.8,
                color=COLOR_HIGHLIGHT, fill_opacity=0.2,
            )
            learned_label = Text(
                "f_theta(x, u)", color=COLOR_HIGHLIGHT, font_size=20,
            )
            learned_label.move_to(learned_box)
            learned = VGroup(learned_box, learned_label)

            learned_tag = Text(
                "Learned from data", color=SLATE, font_size=16,
            )
            learned_tag.next_to(learned, DOWN, buff=0.15)

            # Arrow between them
            classical_grp = VGroup(classical, classical_tag)
            learned_grp = VGroup(learned, learned_tag)
            pipeline = VGroup(classical_grp, learned_grp)
            pipeline.arrange(RIGHT, buff=1.5)
            pipeline.next_to(title, DOWN, buff=0.8)

            arrow = Arrow(
                classical_grp.get_right(), learned_grp.get_left(),
                color=CREAM, stroke_width=2, buff=0.2,
            )
            arrow_label = Text("replace", color=CREAM, font_size=14)
            arrow_label.next_to(arrow, UP, buff=0.1)

            self.play(
                FadeIn(classical_grp, shift=LEFT * 0.3),
                FadeIn(learned_grp, shift=RIGHT * 0.3),
                Create(arrow), FadeIn(arrow_label),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # -- Comparison table ---------------------------------------------
        with self.voiceover(
            text="Here's the full landscape. The Kalman filter uses a linear "
                 "known model. The EKF uses a nonlinear known model. Neural "
                 "ODEs learn a continuous dynamics model. RSSMs learn a "
                 "discrete latent dynamics. And Hamiltonian Neural Networks "
                 "learn dynamics that conserve energy by construction."
        ) as tracker:
            self.play(
                FadeOut(pipeline), FadeOut(arrow), FadeOut(arrow_label),
                run_time=FAST_ANIM,
            )

            table = ComparisonTable(
                headers=["Method", "Dynamics", "Type", "Key Property"],
                rows=[
                    ["KF",         "F * x",          "Linear known",    "Optimal (Gaussian)"],
                    ["EKF",        "f(x)",           "Nonlinear known", "Jacobian approx"],
                    ["Neural ODE", "f_theta (cont)", "Continuous learn", "Adaptive depth"],
                    ["RSSM",       "f_theta (disc)", "Discrete learned", "Latent state"],
                    ["HNN",        "Hamilton's eq",  "Energy-conserv",  "~100x less drift"],
                ],
                row_colors=[
                    COLOR_FILTER_KF,
                    COLOR_FILTER_EKF,
                    TEAL,
                    "#3498db",
                    "#e67e22",
                ],
                title="Prediction Models Across the Series",
                width=10.5,
                font_size=18,
            )
            table.next_to(title, DOWN, buff=0.6)

            self.play(FadeIn(table.bg), run_time=FAST_ANIM)
            for anim in table.animate_rows():
                self.play(anim, run_time=0.4)

        self.wait(PAUSE_MEDIUM)

        # -- Key insight --------------------------------------------------
        with self.voiceover(
            text="The key insight: the prediction step is where physics "
                 "meets machine learning. The rest of this part explores "
                 "each of these learned dynamics approaches in detail."
        ) as tracker:
            insight = Text(
                "Prediction = where physics meets ML",
                color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
            )
            insight.to_edge(DOWN, buff=0.4)
            self.play(FadeIn(insight, scale=0.9), run_time=NORMAL_ANIM)

        note = make_observation_note(
            "Chen et al. (2018, NeurIPS Best Paper): Neural ODEs\n"
            "Key idea: replace hand-crafted f with learned f_theta"
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
