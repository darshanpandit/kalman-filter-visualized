"""Part 7, Scene 5: Filtering vs Prediction — taxonomy and decision guide.

Data: Conceptual + recap

The full taxonomy: filtering (present), prediction (future),
smoothing (past). Where each method fits.

Papers:
- Bewley et al. (2016) SORT
- Wojke et al. (2017) DeepSORT
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


class SceneTaxonomy(VoiceoverScene, MovingCameraScene):
    """Filtering vs Prediction vs Smoothing taxonomy.

    Visual: Timeline diagram + decision guide table.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="Let's put everything in context. There are three "
                 "fundamental tasks in state estimation: filtering — "
                 "estimating the present, prediction — forecasting the "
                 "future, and smoothing — refining the past."
        ) as tracker:
            title = Text(
                "Filtering vs Prediction vs Smoothing",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Timeline diagram ───────────────────────────────────────────
        with self.voiceover(
            text="On a timeline: smoothing uses all data to refine past "
                 "estimates. Filtering uses data up to now. Prediction "
                 "extrapolates beyond the last observation."
        ) as tracker:
            # Timeline
            timeline = Arrow(LEFT * 5, RIGHT * 5, color=CREAM, stroke_width=2)
            timeline.shift(UP * 0.5)

            now_dot = Dot(ORIGIN + UP * 0.5, radius=0.08, color=COLOR_HIGHLIGHT)
            now_label = Text("now", color=COLOR_HIGHLIGHT, font_size=16)
            now_label.next_to(now_dot, DOWN, buff=0.15)

            # Regions
            smooth_region = Rectangle(
                width=4.5, height=0.4, color=TEAL,
                fill_opacity=0.2, stroke_width=1,
            ).move_to(LEFT * 2.5 + UP * 0.5)
            smooth_label = Text("Smoothing", color=TEAL, font_size=16)
            smooth_label.next_to(smooth_region, UP, buff=0.1)

            filter_region = Rectangle(
                width=0.8, height=0.4, color=COLOR_POSTERIOR,
                fill_opacity=0.3, stroke_width=1,
            ).move_to(ORIGIN + UP * 0.5)
            filter_label = Text("Filtering", color=COLOR_POSTERIOR, font_size=16)
            filter_label.next_to(filter_region, UP, buff=0.1)

            pred_region = Rectangle(
                width=4.0, height=0.4, color=COLOR_SOCIAL,
                fill_opacity=0.2, stroke_width=1,
            ).move_to(RIGHT * 2.5 + UP * 0.5)
            pred_label = Text("Prediction", color=COLOR_SOCIAL, font_size=16)
            pred_label.next_to(pred_region, UP, buff=0.1)

            self.play(
                Create(timeline), FadeIn(now_dot), FadeIn(now_label),
                run_time=NORMAL_ANIM,
            )
            self.play(
                FadeIn(smooth_region), FadeIn(smooth_label),
                FadeIn(filter_region), FadeIn(filter_label),
                FadeIn(pred_region), FadeIn(pred_label),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── Decision guide ─────────────────────────────────────────────
        with self.voiceover(
            text="Here's a practical decision guide. For real-time single "
                 "target tracking with a known model, use the Kalman filter "
                 "or EKF. For maneuvering targets, use IMM. For multiple "
                 "targets with unknown cardinality, use the PHD filter. "
                 "For future trajectory forecasting, use social prediction "
                 "models. And SORT or DeepSORT connect detection to tracking."
        ) as tracker:
            self.play(
                *[FadeOut(mob) for mob in self.mobjects if mob is not title],
                run_time=FAST_ANIM,
            )

            table = ComparisonTable(
                headers=["Scenario", "Method", "Key Property"],
                rows=[
                    ["Known linear model",    "KF",          "Optimal"],
                    ["Known nonlinear",       "EKF/UKF",     "Approximate"],
                    ["Multi-modal noise",     "PF",          "Flexible"],
                    ["Maneuvering target",    "IMM",         "Model switching"],
                    ["Unknown # targets",     "PHD",         "No association"],
                    ["Future forecasting",    "S-LSTM/GAN",  "Learned social"],
                    ["Detection → tracking",  "SORT",        "IoU + KF"],
                ],
                row_colors=[
                    COLOR_FILTER_KF, COLOR_FILTER_EKF, COLOR_FILTER_PF,
                    COLOR_FILTER_IMM, COLOR_FILTER_PHD, COLOR_SOCIAL,
                    SLATE,
                ],
                title="Multi-Agent Tracking Decision Guide",
                width=10.0,
                font_size=18,
            )
            table.next_to(title, DOWN, buff=0.6)

            self.play(FadeIn(table.bg), run_time=FAST_ANIM)
            for anim in table.animate_rows():
                self.play(anim, run_time=0.3)

        self.wait(PAUSE_MEDIUM)

        note = make_observation_note(
            "SORT: Bewley et al. (2016); DeepSORT: Wojke et al. (2017)\n"
            "Both use KF internally for track state estimation"
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
