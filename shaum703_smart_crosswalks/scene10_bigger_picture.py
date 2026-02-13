"""SHAUM703, Scene 10: The Bigger Picture.

Summary statistics, treatment decision guide, Vision Zero framing, and
closing statement connecting sensors, algorithms, and infrastructure.

Source: Cirillo, Pandit & Momeni Rad (2025). Evaluation of Smart
Pedestrian Crosswalk Technologies. MDOT SHA Research Report.
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.observation_note import make_observation_note
from shaum703_smart_crosswalks.data import *


class SceneBiggerPicture(VoiceoverScene, MovingCameraScene):
    """The Bigger Picture: from research to policy to saving lives."""

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        title = Text(
            "The Bigger Picture",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text="Let's step back and see the bigger picture. What do all "
                 "these sensors, trackers, and algorithms actually achieve?"
        ) as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Summary stats (appear one by one) ──────────────────────────
        stats_data = [
            ("45-55% crash reduction", COLOR_HIGHLIGHT),
            ("95%+ driver yielding", TEAL),
            ("HOTA > 0.95", COLOR_MEASUREMENT),
        ]

        stats = VGroup()
        for text_str, color in stats_data:
            t = Text(text_str, color=color, font_size=HEADING_FONT_SIZE + 4)
            stats.add(t)
        stats.arrange(DOWN, buff=0.6)
        stats.move_to(ORIGIN)

        with self.voiceover(
            text="Rectangular Rapid Flashing Beacons and Pedestrian Hybrid "
                 "Beacons deliver 45 to 55 percent crash reductions. Driver "
                 "yielding rates jump from as low as 10 percent to over 95 "
                 "percent. And our best tracker achieves a HOTA score above "
                 "0.95 — meaning near-perfect detection and tracking."
        ) as tracker:
            for stat in stats:
                self.play(
                    FadeIn(stat, shift=UP * 0.2, scale=0.9),
                    run_time=NORMAL_ANIM,
                )
                self.wait(PAUSE_SHORT)
            self.wait(PAUSE_MEDIUM)

        # ── Transition ────────────────────────────────────────────────
        self.play(FadeOut(stats), run_time=FAST_ANIM)

        # ── Treatment decision guide ──────────────────────────────────
        guide_title = Text(
            "Treatment Decision Guide",
            color=TEAL, font_size=HEADING_FONT_SIZE,
        )
        guide_title.next_to(title, DOWN, buff=0.55)

        row_data = [
            ("2 lanes, < 30 mph", "Marked crosswalk"),
            ("Multi-lane, 35+ mph", "PHB or MPS required"),
            ("High volume", "Signal + refuge island"),
        ]

        guide_rows = VGroup()
        for condition, treatment in row_data:
            cond_box = RoundedRectangle(
                corner_radius=0.12, width=4.0, height=0.7,
                color=COLOR_MEASUREMENT, fill_color=DARK_SLATE,
                fill_opacity=0.9, stroke_width=1.5,
            )
            cond_text = Text(
                condition, color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
            )
            cond_text.move_to(cond_box)
            cond_group = VGroup(cond_box, cond_text)

            arrow = Arrow(
                start=ORIGIN, end=RIGHT * 1.0,
                color=COLOR_HIGHLIGHT, stroke_width=3,
                max_tip_length_to_length_ratio=0.3,
            )

            treat_box = RoundedRectangle(
                corner_radius=0.12, width=4.0, height=0.7,
                color=COLOR_HIGHLIGHT, fill_color=DARK_SLATE,
                fill_opacity=0.9, stroke_width=1.5,
            )
            treat_text = Text(
                treatment, color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE,
            )
            treat_text.move_to(treat_box)
            treat_group = VGroup(treat_box, treat_text)

            row = VGroup(cond_group, arrow, treat_group).arrange(RIGHT, buff=0.3)
            guide_rows.add(row)

        guide_rows.arrange(DOWN, buff=0.35)
        guide_rows.next_to(guide_title, DOWN, buff=0.45)
        guide_group = VGroup(guide_title, guide_rows)
        guide_group.scale_to_fit_width(min(guide_group.width, 11.6))

        with self.voiceover(
            text="The research translates directly into a treatment decision "
                 "guide. A simple two-lane road under 30 miles per hour? A "
                 "marked crosswalk may suffice. Multi-lane roads at 35 or "
                 "more? You need a Pedestrian Hybrid Beacon or a Midblock "
                 "Pedestrian Signal. High-volume locations demand a full "
                 "signal with a pedestrian refuge island."
        ) as tracker:
            self.play(FadeIn(guide_title), run_time=FAST_ANIM)
            for row in guide_rows:
                self.play(
                    FadeIn(row[0], shift=LEFT * 0.2),
                    GrowArrow(row[1]),
                    FadeIn(row[2], shift=RIGHT * 0.2),
                    run_time=NORMAL_ANIM,
                )
                self.wait(0.3)
            self.wait(PAUSE_MEDIUM)

        # ── Transition to Vision Zero ─────────────────────────────────
        self.play(FadeOut(guide_group), run_time=FAST_ANIM)

        # ── Vision Zero ───────────────────────────────────────────────
        vz_text = Text(
            "Vision Zero: Zero fatalities",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        vz_text.move_to(ORIGIN + UP * 0.5)

        with self.voiceover(
            text="All of this serves one goal: Vision Zero. The commitment "
                 "that no loss of life on our roads is acceptable. Not one."
        ) as tracker:
            self.play(
                FadeIn(vz_text, scale=0.85),
                run_time=SLOW_ANIM,
            )
            self.wait(PAUSE_LONG)

        # ── Closing statement ─────────────────────────────────────────
        closing = Text(
            "Sensors see. Algorithms predict.\nInfrastructure saves.",
            color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
            line_spacing=1.3,
        )
        closing.next_to(vz_text, DOWN, buff=0.8)

        with self.voiceover(
            text="Sensors see. Algorithms predict. And infrastructure saves "
                 "lives. That is the pipeline — from a camera on a pole to "
                 "a Kalman filter tracking a pedestrian to a flashing beacon "
                 "that stops a driver just in time."
        ) as tracker:
            self.play(
                FadeIn(closing, shift=UP * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_LONG)

        # ── Citation ───────────────────────────────────────────────────
        citation = make_observation_note(
            "Cirillo, Pandit & Momeni Rad (2025)\n"
            "MDOT SHA Research Report"
        )

        with self.voiceover(
            text="This research is documented in the MDOT SHA report by "
                 "Cirillo, Pandit, and Momeni Rad, 2025. Thank you for "
                 "watching."
        ) as tracker:
            self.play(FadeIn(citation), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # ── Fade out ───────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
