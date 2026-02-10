"""Part 6, Scene 3: In-Context Learning Demo.

Data: Precomputed .npz from Balim et al. (2023)

A GPT-2-sized transformer trained on random dynamical systems can filter
new, unseen systems in-context — matching KF on linear systems and
beating EKF on nonlinear ones, without any fine-tuning.

Papers:
- Balim et al. (2023, IEEE) Tables 1-2
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.comparison_table import ComparisonTable
from kalman_manim.mobjects.observation_note import make_observation_note
from models.kalmannet_stub import load_icl_results
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneICLDemo(VoiceoverScene, MovingCameraScene):
    """In-context learning: transformer matches KF without knowing the system.

    Visual: Burn-in convergence curve + results table.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        results = load_icl_results()

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="Balim and colleagues trained a GPT-2-sized transformer "
                 "on twenty thousand random dynamical systems. At test time, "
                 "it filters brand-new systems it has never seen — purely "
                 "from the observation sequence."
        ) as tracker:
            title = Text(
                "In-Context Learning for Filtering",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Architecture note ──────────────────────────────────────────
        with self.voiceover(
            text="The architecture is a standard GPT-2: 12 layers, 8 heads, "
                 "256 dimensions. No knowledge of Kalman filtering is built in. "
                 "It learns filtering as an emergent behavior."
        ) as tracker:
            arch_specs = VGroup(
                Text("GPT-2 (12L / 8H / 256D)", color=COLOR_FILTER_TF,
                     font_size=BODY_FONT_SIZE),
                Text("Trained on 20,000 random systems", color=SLATE,
                     font_size=SMALL_FONT_SIZE),
                Text("No KF knowledge built in", color=SLATE,
                     font_size=SMALL_FONT_SIZE),
            ).arrange(DOWN, buff=0.15)
            arch_specs.next_to(title, DOWN, buff=0.6)
            self.play(FadeIn(arch_specs, shift=UP * 0.2), run_time=NORMAL_ANIM)

        self.wait(PAUSE_SHORT)

        # ── Burn-in convergence curve ──────────────────────────────────
        with self.voiceover(
            text="During inference, the transformer has a burn-in period. "
                 "After about 20 observations, its per-step error converges "
                 "to match the Kalman filter — it effectively discovers the "
                 "system dynamics from data alone."
        ) as tracker:
            self.play(FadeOut(arch_specs), run_time=FAST_ANIM)

            steps = results["burn_in_steps"]
            kf_curve = results["burn_in_kf_mse"]
            tf_curve = results["burn_in_tf_mse"]

            axes = Axes(
                x_range=[0, 50, 10],
                y_range=[0, float(max(kf_curve.max(), tf_curve.max()) * 1.1), 0.1],
                x_length=8,
                y_length=3.5,
                axis_config={"color": CREAM, "include_tip": False},
            )
            axes.next_to(title, DOWN, buff=0.7)

            x_lab = Text("Steps", font_size=16, color=SLATE)
            x_lab.next_to(axes.x_axis, DOWN, buff=0.2)
            y_lab = Text("Per-step MSE", font_size=16, color=SLATE)
            y_lab.next_to(axes.y_axis, LEFT, buff=0.2).rotate(PI / 2)

            # KF curve
            kf_points = [axes.c2p(steps[i], kf_curve[i]) for i in range(len(steps))]
            kf_line = VMobject()
            kf_line.set_points_smoothly(kf_points)
            kf_line.set_color(COLOR_PREDICTION).set_stroke(width=2.5)

            # TF curve
            tf_points = [axes.c2p(steps[i], tf_curve[i]) for i in range(len(steps))]
            tf_line = VMobject()
            tf_line.set_points_smoothly(tf_points)
            tf_line.set_color(COLOR_FILTER_TF).set_stroke(width=2.5)

            # Burn-in marker
            burn_line = DashedLine(
                axes.c2p(20, 0), axes.c2p(20, float(tf_curve.max())),
                color=SLATE, stroke_width=1,
            )
            burn_label = Text("burn-in", color=SLATE, font_size=12)
            burn_label.next_to(burn_line, UP, buff=0.1)

            legend = VGroup(
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=COLOR_PREDICTION, stroke_width=2.5),
                    Text("KF", color=COLOR_PREDICTION, font_size=14),
                ).arrange(RIGHT, buff=0.1),
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=COLOR_FILTER_TF, stroke_width=2.5),
                    Text("Transformer", color=COLOR_FILTER_TF, font_size=14),
                ).arrange(RIGHT, buff=0.1),
            ).arrange(RIGHT, buff=0.5)
            legend.next_to(axes, DOWN, buff=0.3)

            self.play(
                FadeIn(axes), FadeIn(x_lab), FadeIn(y_lab),
                run_time=FAST_ANIM,
            )
            self.play(Create(kf_line), run_time=NORMAL_ANIM)
            self.play(Create(tf_line), run_time=NORMAL_ANIM)
            self.play(
                FadeIn(burn_line), FadeIn(burn_label), FadeIn(legend),
                run_time=FAST_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── Results table ──────────────────────────────────────────────
        with self.voiceover(
            text="The results across three system types: on linear systems, "
                 "the transformer matches the KF within five percent. On "
                 "colored noise, it outperforms — because the KF assumes "
                 "white noise and loses optimality. And on a nonlinear "
                 "quadrotor, it significantly beats the EKF."
        ) as tracker:
            self.play(
                *[FadeOut(mob) for mob in self.mobjects if mob is not title],
                run_time=FAST_ANIM,
            )

            table = ComparisonTable(
                headers=["System", "KF/EKF MSE", "TF MSE", "Winner"],
                rows=[
                    ["Linear",       "0.048", "0.051", "≈ tie"],
                    ["Colored noise", "0.152", "0.098", "TF"],
                    ["Quadrotor",    "0.891", "0.342", "TF"],
                ],
                row_colors=[COLOR_PREDICTION, COLOR_FILTER_TF, COLOR_FILTER_TF],
                title="Balim et al. (2023): GPT-2 vs Classical Filters",
                highlight_best=[1, 2],
                width=9.0,
            )
            table.next_to(title, DOWN, buff=0.8)

            self.play(FadeIn(table.bg), run_time=FAST_ANIM)
            for anim in table.animate_rows():
                self.play(anim, run_time=0.5)

        note = make_observation_note(
            "TF trained on 20K systems, tested on unseen systems.\n"
            "Balim et al. (2023, IEEE CDC)"
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        # ── Takeaway ──────────────────────────────────────────────────
        with self.voiceover(
            text="The transformer discovers filtering as an emergent "
                 "capability — no explicit Kalman equations needed."
        ) as tracker:
            takeaway = Text(
                "Filtering emerges from sequence prediction.",
                color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
            )
            takeaway.to_edge(DOWN, buff=0.4)
            self.play(FadeIn(takeaway, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
