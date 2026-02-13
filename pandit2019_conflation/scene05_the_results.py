"""Scene 05: The Results — Gumbel distributions, coverage gaps, discrepancy maps.

Azure multi-voice scene. Darshan presents the algorithm's output: bidirectional
matching, Gumbel score distributions, excess/missing coverage maps, and the
key quantitative findings (5.11% excess, 3.10% missing). Jenny provides
newscast-style statistical commentary.

Voices: narrator (Jenny, chat), narrator_newscast (Jenny, newscast),
        darshan (Tony, friendly), darshan slow (Tony, friendly, rate=-10%).
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.azure import AzureService
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from pandit2019_conflation.data import RESULTS, fig_path


class SceneTheResults(VoiceoverScene, MovingCameraScene):
    def construct(self):
        # ── Voice services ─────────────────────────────────────────
        narrator = AzureService(voice="en-US-JennyNeural", style="chat")
        narrator_newscast = AzureService(voice="en-US-JennyNeural", style="newscast")
        darshan = AzureService(voice="en-US-TonyNeural", style="friendly")
        self.set_speech_service(darshan)
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────
        title = Text(
            "The Results", color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text=(
                "We ran the algorithm in both directions. NPMRDS to HPMS, "
                "and HPMS to NPMRDS. You need both — one direction finds "
                "excess segments, the other finds missing ones."
            )
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Figure 6: Gumbel distributions ─────────────────────────
        fig_gumbel = ImageMobject(fig_path("fig6_gumbel_distributions.png"))
        fig_gumbel.scale_to_fit_width(10)
        fig_gumbel.next_to(title, DOWN, buff=0.4)

        self.set_speech_service(narrator_newscast)

        with self.voiceover(
            text=(
                "The final scores follow a Gumbel distribution — Extreme "
                "Value Type one. That's expected. You're taking the minimum "
                "among candidates, which is an extreme value operation. "
                "The long right tail is where the mismatches live."
            )
        ) as tracker:
            self.play(FadeIn(fig_gumbel, shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Darshan on cutoff ──────────────────────────────────────
        self.set_speech_service(darshan)

        with self.voiceover(
            text=(
                "The practical question is: where do you draw the cutoff? "
                "We discarded the top half-percent as unreliable matches."
            )
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Fade out Gumbel figure ─────────────────────────────────
        self.play(FadeOut(fig_gumbel), run_time=FAST_ANIM)

        # ── Figure 5: Excess and missing maps ──────────────────────
        fig_maps = ImageMobject(fig_path("fig5_excess_missing_maps.png"))
        fig_maps.scale_to_fit_width(10)
        fig_maps.next_to(title, DOWN, buff=0.4)

        with self.voiceover(
            text=(
                "Here's what the data looks like on a map. Yellow lines "
                "are NHS roads from HPMS. Black lines are NPMRDS. Where "
                "they don't overlap, something is wrong."
            )
        ) as tracker:
            self.play(FadeIn(fig_maps, shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Fade out maps figure ───────────────────────────────────
        self.play(FadeOut(fig_maps), run_time=FAST_ANIM)

        # ── Figure 7: Combined results map with stat overlay ───────
        fig_results = ImageMobject(fig_path("fig7_combined.png"))
        fig_results.scale_to_fit_width(10)
        fig_results.next_to(title, DOWN, buff=0.4)

        # ── Stat overlay cards at the bottom ───────────────────────
        stat_box = RoundedRectangle(
            width=9.0, height=1.5, corner_radius=0.12,
            stroke_color=SLATE, stroke_width=1.5,
            fill_color=BG_COLOR, fill_opacity=0.92,
        )
        stat_box.to_edge(DOWN, buff=0.3)

        # ValueTrackers for count-up animation
        excess_tracker = ValueTracker(0)
        missing_tracker = ValueTracker(0)

        excess_target = RESULTS["excess_tmc_pct"]   # 5.11
        missing_target = RESULTS["missing_npmrds_pct"]  # 3.10

        # Excess label (left side of stat box)
        excess_prefix = Text(
            "Excess: ", color=COLOR_PREDICTION, font_size=BODY_FONT_SIZE,
        )
        excess_number = always_redraw(
            lambda: Text(
                f"{excess_tracker.get_value():.2f}%",
                color=COLOR_PREDICTION, font_size=BODY_FONT_SIZE,
            ).next_to(excess_prefix, RIGHT, buff=0.1)
        )

        # Missing label (right side of stat box)
        missing_prefix = Text(
            "Missing: ", color=TEAL, font_size=BODY_FONT_SIZE,
        )
        missing_number = always_redraw(
            lambda: Text(
                f"{missing_tracker.get_value():.2f}%",
                color=TEAL, font_size=BODY_FONT_SIZE,
            ).next_to(missing_prefix, RIGHT, buff=0.1)
        )

        # Position the stat groups within the box
        excess_group = VGroup(excess_prefix, excess_number)
        missing_group = VGroup(missing_prefix, missing_number)

        excess_prefix.move_to(
            stat_box.get_center() + LEFT * 2.5
        )
        missing_prefix.move_to(
            stat_box.get_center() + RIGHT * 1.5
        )

        with self.voiceover(
            text=(
                "Red segments are excess TMC — roads in the travel time "
                "dataset that shouldn't be there. Purple segments are "
                "missing — NHS roads with no travel time coverage at all. "
                "In Delaware, Maryland and DC alone."
            )
        ) as tracker:
            self.play(FadeIn(fig_results, shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(
                FadeIn(stat_box),
                FadeIn(excess_prefix), FadeIn(excess_number),
                FadeIn(missing_prefix), FadeIn(missing_number),
                run_time=FAST_ANIM,
            )
            # Count-up animation
            self.play(
                excess_tracker.animate.set_value(excess_target),
                missing_tracker.animate.set_value(missing_target),
                run_time=SLOW_ANIM,
                rate_func=smooth,
            )
            self.wait(PAUSE_MEDIUM)

        # ── Narrator reacts ────────────────────────────────────────
        self.set_speech_service(narrator)

        with self.voiceover(
            text="Five percent excess. Three percent missing."
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Darshan slow closing ───────────────────────────────────
        self.set_speech_service(darshan)

        with self.voiceover(
            text=(
                "If your map is wrong, your congestion numbers are wrong. "
                "Your greenhouse gas estimates are wrong. The funding "
                "allocations based on those numbers — wrong. This is "
                "what we found when we looked at just three states."
            ),
            prosody={"rate": "-10%"},
        ) as tracker:
            self.wait(PAUSE_LONG)

        # ── Fade out ───────────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob is not title],
            run_time=NORMAL_ANIM,
        )
        self.wait(PAUSE_LONG)
        self.play(FadeOut(title), run_time=FAST_ANIM)
