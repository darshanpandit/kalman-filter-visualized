"""Part 9, Scene 5: The Full Picture — grand taxonomy and series recap.

Data: Conceptual diagrams

Shows the GrandTaxonomyDiagram spanning all 9 parts of the series.
Grand recap: from the Kalman filter's linear predict-update loop to
Neural ODEs and Hamiltonian dynamics. Final takeaway: "Different
tools for different assumptions."

Papers:
- Kalman (1960) — Original KF
- Chen et al. (2018) — Neural ODEs
- Greydanus et al. (2019) — HNN
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.taxonomy_diagram import GrandTaxonomyDiagram
from kalman_manim.mobjects.comparison_table import ComparisonTable
from kalman_manim.mobjects.observation_note import make_observation_note
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneFullPicture(VoiceoverScene, MovingCameraScene):
    """The full picture: grand taxonomy and 9-part series recap.

    Visual: GrandTaxonomyDiagram, series timeline, final takeaway.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # -- Title --------------------------------------------------------
        with self.voiceover(
            text="We've come a long way — from the Kalman filter's simple "
                 "linear predict-update loop to neural ODEs that learn "
                 "continuous dynamics from data. Let's see the full picture."
        ) as tracker:
            title = Text(
                "The Full Picture",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # -- Grand Taxonomy Diagram ---------------------------------------
        with self.voiceover(
            text="Here is the grand taxonomy of state estimation. Classical "
                 "filters — KF, EKF, UKF, particle filters — all require "
                 "hand-crafted dynamics. Transformers and KalmanNet learn "
                 "to filter from data. IMM and PHD handle multiple models "
                 "and multiple targets. And the newest approaches — RSSMs, "
                 "Neural ODEs, Hamiltonian networks — learn the dynamics "
                 "themselves."
        ) as tracker:
            diagram = GrandTaxonomyDiagram(width=11.0)
            diagram.next_to(title, DOWN, buff=0.7)

            self.play(
                FadeIn(diagram.nodes, lag_ratio=0.1),
                run_time=2.0,
            )

        self.wait(PAUSE_MEDIUM)

        # -- Series timeline ----------------------------------------------
        with self.voiceover(
            text="Let's trace our journey. Part 1: the Kalman filter and "
                 "Bayesian foundations. Part 2: the Extended KF for nonlinear "
                 "systems. Part 3: the Unscented KF with sigma points. "
                 "Part 4: particle filters for arbitrary distributions. "
                 "Part 5: benchmarks on real pedestrian data."
        ) as tracker:
            self.play(FadeOut(diagram), run_time=FAST_ANIM)

            parts_data = [
                ("Part 1", "KF",          COLOR_FILTER_KF),
                ("Part 2", "EKF",         COLOR_FILTER_EKF),
                ("Part 3", "UKF",         COLOR_FILTER_UKF),
                ("Part 4", "PF",          COLOR_FILTER_PF),
                ("Part 5", "Benchmarks",  CREAM),
                ("Part 6", "Transformers", COLOR_FILTER_TF),
                ("Part 7", "Multi-Agent", COLOR_FILTER_IMM),
                ("Part 8", "World Models","#3498db"),
                ("Part 9", "Dynamics",    TEAL),
            ]

            timeline = Arrow(LEFT * 5.5, RIGHT * 5.5, color=CREAM, stroke_width=2)
            timeline.next_to(title, DOWN, buff=1.2)

            dots = VGroup()
            labels = VGroup()
            for i, (part, name, color) in enumerate(parts_data):
                x = LEFT * 5 + RIGHT * (i * 1.25)
                dot = Dot(
                    timeline.get_left() + RIGHT * (i * 1.25) + RIGHT * 0.25,
                    radius=0.06, color=color,
                )

                part_label = Text(part, color=color, font_size=12)
                part_label.next_to(dot, UP, buff=0.15)
                name_label = Text(name, color=SLATE, font_size=10)
                name_label.next_to(dot, DOWN, buff=0.15)

                dots.add(dot)
                labels.add(VGroup(part_label, name_label))

            self.play(Create(timeline), run_time=FAST_ANIM)
            self.play(
                FadeIn(dots, lag_ratio=0.15),
                FadeIn(labels, lag_ratio=0.15),
                run_time=2.0,
            )

        self.wait(PAUSE_SHORT)

        with self.voiceover(
            text="Part 6: transformers as learned filters. Part 7: multi-agent "
                 "tracking with IMM and PHD. Part 8: world models that learn "
                 "environment dynamics. And Part 9 — this one — where we "
                 "explored Neural ODEs, PINNs, and Hamiltonian networks for "
                 "learning the dynamics model itself."
        ) as tracker:
            # Highlight parts 6-9
            highlight_rect = SurroundingRectangle(
                VGroup(dots[5:], labels[5:]),
                color=COLOR_HIGHLIGHT, buff=0.15,
                stroke_width=1.5, corner_radius=0.1,
            )
            highlight_label = Text(
                "Learned dynamics", color=COLOR_HIGHLIGHT, font_size=14,
            )
            highlight_label.next_to(highlight_rect, DOWN, buff=0.15)

            self.play(
                Create(highlight_rect), FadeIn(highlight_label),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # -- Decision guide table -----------------------------------------
        with self.voiceover(
            text="Here's the practical decision guide spanning the entire "
                 "series. If you know your dynamics and they're linear, use "
                 "the Kalman filter — it's optimal. If they're nonlinear "
                 "but known, use EKF or UKF. If the distribution is "
                 "multi-modal, use a particle filter. If you don't know the "
                 "dynamics at all, learn them with Neural ODEs. And if your "
                 "system conserves energy, use a Hamiltonian Neural Network."
        ) as tracker:
            self.play(
                FadeOut(timeline), FadeOut(dots), FadeOut(labels),
                FadeOut(highlight_rect), FadeOut(highlight_label),
                run_time=FAST_ANIM,
            )

            table = ComparisonTable(
                headers=["Assumption", "Method", "Part"],
                rows=[
                    ["Linear + Gaussian",     "KF",          "1"],
                    ["Nonlinear, known f",    "EKF / UKF",   "2-3"],
                    ["Multi-modal",           "Particle F",   "4"],
                    ["Multi-model switching",  "IMM",         "7"],
                    ["Multiple targets",       "PHD",         "7"],
                    ["Unknown dynamics",       "Neural ODE",  "9"],
                    ["Partial physics known",  "PINN",        "9"],
                    ["Energy-conserving",      "HNN",         "9"],
                ],
                row_colors=[
                    COLOR_FILTER_KF,
                    COLOR_FILTER_EKF,
                    COLOR_FILTER_PF,
                    COLOR_FILTER_IMM,
                    COLOR_FILTER_PHD,
                    TEAL,
                    TEAL,
                    "#e67e22",
                ],
                title="Which Tool for Which Assumption?",
                width=9.5,
                font_size=18,
            )
            table.next_to(title, DOWN, buff=0.6)

            self.play(FadeIn(table.bg), run_time=FAST_ANIM)
            for anim in table.animate_rows():
                self.play(anim, run_time=0.3)

        self.wait(PAUSE_MEDIUM)

        # -- Final takeaway -----------------------------------------------
        with self.voiceover(
            text="The final takeaway: there is no single best filter or "
                 "dynamics model. Each method makes different assumptions "
                 "about your system. The art of state estimation is matching "
                 "the right tool to your problem's assumptions. Different "
                 "tools for different assumptions."
        ) as tracker:
            self.play(FadeOut(table), run_time=FAST_ANIM)

            takeaway = Text(
                "Different tools for different assumptions.",
                color=COLOR_HIGHLIGHT, font_size=TITLE_FONT_SIZE,
            )
            takeaway.move_to(ORIGIN)

            self.play(FadeIn(takeaway, scale=0.8), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # -- Series complete message --------------------------------------
        with self.voiceover(
            text="And with that, our nine-part journey through state "
                 "estimation is complete. From Rudolf Kalman's 1960 paper "
                 "to Hamiltonian Neural Networks in 2019, the core idea "
                 "remains the same: combine what you know with what you "
                 "observe, and handle the uncertainty honestly."
        ) as tracker:
            subtitle = Text(
                "Combine what you know with what you observe.",
                color=CREAM, font_size=BODY_FONT_SIZE,
            )
            subtitle.next_to(takeaway, DOWN, buff=0.5)
            self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=NORMAL_ANIM)

        note = make_observation_note(
            "From Kalman (1960) to Greydanus et al. (2019):\n"
            "60 years of state estimation, one unifying idea"
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
