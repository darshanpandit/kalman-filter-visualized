"""SHAUM703, Scene 1: The Invisible Crisis.

Data: FATALITY_STATS from data.py

Animated percentage counter and stat cards illustrating the pedestrian
fatality crisis at non-intersection locations — the motivation for smart
crosswalk technologies.

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
from shaum703_smart_crosswalks.data import FATALITY_STATS


class SceneInvisibleCrisis(VoiceoverScene, MovingCameraScene):
    """The Invisible Crisis: why pedestrian fatalities demand smart crosswalks."""

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ────────────────────────────────────────────────────────
        title = Text(
            "The Invisible Crisis",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text="Every year, thousands of pedestrians are killed on American "
                 "roads. But what's truly shocking is where most of these "
                 "fatalities happen."
        ) as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Animated percentage counter ──────────────────────────────────
        target_pct = FATALITY_STATS["non_intersection_pct"]  # 74

        pct_value = ValueTracker(0)
        pct_display = always_redraw(
            lambda: Text(
                f"{int(pct_value.get_value())}%",
                color=COLOR_HIGHLIGHT, font_size=96,
            ).move_to(ORIGIN + UP * 0.3)
        )
        pct_label = Text(
            "of pedestrian fatalities",
            color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
        )
        pct_sublabel = Text(
            "occur outside intersections",
            color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
        )
        label_group = VGroup(pct_label, pct_sublabel).arrange(DOWN, buff=0.15)
        label_group.next_to(pct_display, DOWN, buff=0.5)

        with self.voiceover(
            text="Seventy-four percent. Nearly three out of every four "
                 "pedestrian deaths happen outside of intersections — at "
                 "locations that often lack any safety infrastructure."
        ) as tracker:
            self.add(pct_display)
            self.play(
                pct_value.animate.set_value(target_pct),
                run_time=SLOW_ANIM,
                rate_func=smooth,
            )
            self.play(FadeIn(label_group, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Transition to stat cards ─────────────────────────────────────
        self.play(
            FadeOut(pct_display), FadeOut(label_group),
            run_time=NORMAL_ANIM,
        )

        # ── Three stat cards ─────────────────────────────────────────────
        card_data = [
            (f"{FATALITY_STATS['non_intersection_pct']}%",
             "Non-Intersection", COLOR_PREDICTION),
            (f"{FATALITY_STATS['unsignalized_midblock_pct']}%",
             "Unsignalized", COLOR_MEASUREMENT),
            (f"~{FATALITY_STATS['nighttime_pct']}%",
             "At Night", COLOR_HIGHLIGHT),
        ]

        cards = VGroup()
        for stat_val, stat_desc, color in card_data:
            bg = RoundedRectangle(
                corner_radius=0.15, width=3.2, height=2.2,
                color=color, fill_color=BG_COLOR,
                fill_opacity=0.85, stroke_width=2.5,
            )
            num = Text(stat_val, color=color, font_size=HEADING_FONT_SIZE + 8)
            desc = Text(stat_desc, color=COLOR_TEXT, font_size=SMALL_FONT_SIZE)
            content = VGroup(num, desc).arrange(DOWN, buff=0.25)
            content.move_to(bg)
            cards.add(VGroup(bg, content))

        cards.arrange(RIGHT, buff=0.5)
        cards.next_to(title, DOWN, buff=0.8)
        cards.scale_to_fit_width(min(cards.width, 11.6))

        with self.voiceover(
            text="The numbers paint a stark picture. Seventy-four percent of "
                 "pedestrian fatalities occur at non-intersection locations. "
                 "Ninety-three percent happen at unsignalized crossings — "
                 "places with no traffic signals at all. And nearly half of "
                 "all pedestrian deaths occur at night, when visibility is "
                 "at its worst."
        ) as tracker:
            for card in cards:
                self.play(
                    FadeIn(card[0]),
                    FadeIn(card[1], shift=UP * 0.15),
                    run_time=NORMAL_ANIM,
                )
                self.wait(PAUSE_SHORT)
            self.wait(PAUSE_MEDIUM)

        # ── Concluding insight ───────────────────────────────────────────
        insight = Text(
            "The most dangerous place is the unmarked crossing.",
            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
        )
        citation = make_observation_note(
            "Source: Cirillo, Pandit & Momeni Rad (2025)\n"
            "MDOT SHA Research Report"
        )
        bottom_group = VGroup(insight, citation).arrange(DOWN, buff=0.35)
        bottom_group.to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="The most dangerous place for a pedestrian is not a busy "
                 "intersection — it's the unmarked midblock crossing where "
                 "no one thought to put a signal. This is the invisible "
                 "crisis that smart crosswalk technologies aim to solve."
        ) as tracker:
            self.play(FadeIn(insight, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)
            self.play(FadeIn(citation), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # ── Fade out ─────────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
