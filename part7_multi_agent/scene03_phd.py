"""Part 7, Scene 3: PHD Filter — tracking without data association.

Data: Synthetic multi-target scenario

The PHD filter represents the expected number of targets as a
Gaussian mixture intensity function, handling birth/death without
explicit measurement-to-track assignment.

Papers:
- Mahler (2003/2007) — Random Finite Sets
- Vo & Ma (2006) — GM-PHD filter
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.comparison_table import ComparisonTable
from kalman_manim.mobjects.observation_note import make_observation_note
from kalman_manim.data.generators import generate_multi_target_scenario
from filters.gmphd import GMPHDFilter, GaussianComponent
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class ScenePHD(VoiceoverScene, MovingCameraScene):
    """PHD filter: intensity-based multi-target tracking.

    Visual: Cardinality estimate over time + comparison table.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # Generate multi-target data
        data = generate_multi_target_scenario(
            n_steps=50, n_targets_init=3, birth_step=15,
            death_step=35, clutter_rate=0.3, seed=42,
        )

        # Run GM-PHD
        dt = data["dt"]
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                       [0, 0, 1, 0], [0, 0, 0, 1]])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        Q = 0.2 * np.eye(4)
        R = 0.5 * np.eye(2)

        # Birth components at known entry points
        birth = [
            GaussianComponent(0.1, np.array([5.0, 0.0, -0.3, 0.4]), np.eye(4)),
            GaussianComponent(0.05, np.array([0.0, 0.0, 0.5, 0.0]), 2.0 * np.eye(4)),
        ]

        phd = GMPHDFilter(
            F=F, H=H, Q=Q, R=R,
            ps=0.95, pd=0.90,
            clutter_intensity=5e-4,
            birth_components=birth,
        )

        results = phd.run(data["measurement_sets"])

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="The PHD filter takes a radically different approach: "
                 "instead of tracking individual targets, it tracks an "
                 "intensity function — the expected number of targets "
                 "per unit area. No data association needed."
        ) as tracker:
            title = Text(
                "PHD Filter: No Association Needed",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Key insight ────────────────────────────────────────────────
        with self.voiceover(
            text="The key insight from Ronald Mahler: model the multi-target "
                 "state as a random finite set. The first moment of this set "
                 "— the probability hypothesis density — gives you the "
                 "expected number of targets everywhere in the state space."
        ) as tracker:
            insight = VGroup(
                Text("Random Finite Sets (Mahler 2003):", color=COLOR_FILTER_PHD,
                     font_size=BODY_FONT_SIZE),
                Text("PHD = expected target count per unit area",
                     color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
                Text("∫ PHD(x) dx = expected number of targets",
                     color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE),
            ).arrange(DOWN, buff=0.15)
            insight.next_to(title, DOWN, buff=0.6)

            self.play(FadeIn(insight, shift=UP * 0.2), run_time=NORMAL_ANIM)

        self.wait(PAUSE_MEDIUM)

        # ── Cardinality tracking ───────────────────────────────────────
        with self.voiceover(
            text="Watch the cardinality estimate. Three targets start. "
                 "At step fifteen, a fourth appears — the PHD detects the "
                 "birth. At step thirty-five, one disappears — the PHD "
                 "tracks the death. All without data association."
        ) as tracker:
            self.play(FadeOut(insight), run_time=FAST_ANIM)

            card_est = results["cardinality_estimates"]
            true_card = data["true_cardinality"]
            n = len(card_est)

            axes = Axes(
                x_range=[0, n, 10],
                y_range=[0, 6, 1],
                x_length=8, y_length=3.5,
                axis_config={"color": CREAM, "include_tip": False},
            )
            axes.next_to(title, DOWN, buff=0.7)

            x_lab = Text("Time step", font_size=14, color=SLATE)
            x_lab.next_to(axes.x_axis, DOWN, buff=0.15)
            y_lab = Text("# Targets", font_size=14, color=SLATE)
            y_lab.next_to(axes.y_axis, LEFT, buff=0.15).rotate(PI / 2)

            # True cardinality
            true_pts = [axes.c2p(t, true_card[t]) for t in range(n)]
            true_line = VMobject()
            true_line.set_points_as_corners(true_pts)
            true_line.set_color(COLOR_TRUE_PATH).set_stroke(width=2)

            # PHD estimate
            est_pts = [axes.c2p(t, card_est[t]) for t in range(n)]
            est_line = VMobject()
            est_line.set_points_smoothly(est_pts)
            est_line.set_color(COLOR_FILTER_PHD).set_stroke(width=2.5)

            # Birth/death markers
            birth_line = DashedLine(
                axes.c2p(15, 0), axes.c2p(15, 5),
                color=TEAL, stroke_width=1,
            )
            birth_label = Text("birth", color=TEAL, font_size=12)
            birth_label.next_to(birth_line, UP, buff=0.05)

            death_line = DashedLine(
                axes.c2p(35, 0), axes.c2p(35, 5),
                color=SWISS_RED, stroke_width=1,
            )
            death_label = Text("death", color=SWISS_RED, font_size=12)
            death_label.next_to(death_line, UP, buff=0.05)

            legend = VGroup(
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=COLOR_TRUE_PATH, stroke_width=2),
                    Text("True", color=COLOR_TRUE_PATH, font_size=14),
                ).arrange(RIGHT, buff=0.1),
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=COLOR_FILTER_PHD, stroke_width=2.5),
                    Text("PHD estimate", color=COLOR_FILTER_PHD, font_size=14),
                ).arrange(RIGHT, buff=0.1),
            ).arrange(RIGHT, buff=0.5)
            legend.next_to(axes, DOWN, buff=0.3)

            self.play(
                FadeIn(axes), FadeIn(x_lab), FadeIn(y_lab),
                run_time=FAST_ANIM,
            )
            self.play(Create(true_line), run_time=NORMAL_ANIM)
            self.play(Create(est_line), run_time=NORMAL_ANIM)
            self.play(
                FadeIn(birth_line), FadeIn(birth_label),
                FadeIn(death_line), FadeIn(death_label),
                FadeIn(legend),
                run_time=FAST_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        note = make_observation_note(
            "GM-PHD: Vo & Ma (2006, IEEE Trans. SP)\n"
            "Gaussian mixture representation of intensity"
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
