"""Scene 1: The Map Problem — Two maps, same highway, different shapes.

Narrative lead: Jenny (mathematician/narrator)
Introduces the fundamental conflation problem: NPMRDS and HPMS datasets
describe the same road network but disagree on geometry and coverage.
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from pandit2019_conflation.data import RESULTS, STUDY_REGION, fig_path


class SceneMapProblem(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ─────────────────────────────────────────────────────
        title = Text(
            "The Map Problem",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        # ── Beat 1: Hook ──────────────────────────────────────────────
        hook = Text(
            "Two maps.  Same highway.  Different shapes.",
            color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
        )
        hook.next_to(title, DOWN, buff=0.8)

        with self.voiceover(
            text=(
                "Two maps. Same highway. Different shapes. "
                "Here's a problem that sounds simple but isn't. "
                "Before you can track a pedestrian, filter a GPS signal, "
                "or measure travel time, the road network itself must be correct. "
                "And it turns out, our maps don't agree."
            )
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(hook, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        self.play(FadeOut(hook), run_time=FAST_ANIM)

        # ── Beat 2: Two panels — NPMRDS vs HPMS ──────────────────────
        panel_w, panel_h = 5.2, 3.2

        # NPMRDS panel (left, blue)
        npmrds_box = RoundedRectangle(
            width=panel_w, height=panel_h, corner_radius=0.15,
            stroke_color=COLOR_MEASUREMENT, stroke_width=2,
            fill_color=DARK_SLATE, fill_opacity=0.6,
        )
        npmrds_box.shift(LEFT * 3.2 + DOWN * 0.5)

        npmrds_title = Text("NPMRDS", color=COLOR_MEASUREMENT, font_size=HEADING_FONT_SIZE)
        npmrds_title.next_to(npmrds_box, UP, buff=0.15)

        npmrds_lines = VGroup(
            Text("TMC segments", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
            Text("Probe-based travel times", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
            Text("5-min epochs from GPS probes", color=SLATE, font_size=SMALL_FONT_SIZE),
        )
        npmrds_lines.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        npmrds_lines.move_to(npmrds_box)

        # HPMS panel (right, gold)
        hpms_box = RoundedRectangle(
            width=panel_w, height=panel_h, corner_radius=0.15,
            stroke_color=COLOR_POSTERIOR, stroke_width=2,
            fill_color=DARK_SLATE, fill_opacity=0.6,
        )
        hpms_box.shift(RIGHT * 3.2 + DOWN * 0.5)

        hpms_title = Text("HPMS", color=COLOR_POSTERIOR, font_size=HEADING_FONT_SIZE)
        hpms_title.next_to(hpms_box, UP, buff=0.15)

        hpms_lines = VGroup(
            Text("Road attributes", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
            Text("AADT, lanes, road class", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
            Text("Federal highway inventory", color=SLATE, font_size=SMALL_FONT_SIZE),
        )
        hpms_lines.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        hpms_lines.move_to(hpms_box)

        with self.voiceover(
            text=(
                "On the left, the National Performance Management Research Data Set, "
                "or NPMRDS. It chops the road network into TMC segments and records "
                "probe-based travel times every five minutes from GPS-equipped vehicles. "
                "On the right, the Highway Performance Monitoring System, or HPMS. "
                "It stores road attributes: annual average daily traffic, lane counts, "
                "functional classification. Two federal datasets, both describing the "
                "same roads in Delaware, Maryland, and Washington D.C."
            )
        ) as tracker:
            self.play(
                FadeIn(npmrds_box), FadeIn(npmrds_title),
                FadeIn(hpms_box), FadeIn(hpms_title),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_SHORT)
            self.play(
                FadeIn(npmrds_lines, shift=UP * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_MEDIUM)
            self.play(
                FadeIn(hpms_lines, shift=UP * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_LONG)

        # ── Beat 3: The mismatch — excess and missing ─────────────────
        panels = VGroup(
            npmrds_box, npmrds_title, npmrds_lines,
            hpms_box, hpms_title, hpms_lines,
        )
        self.play(FadeOut(panels), run_time=FAST_ANIM)

        excess_pct = f"{RESULTS['excess_tmc_pct']}%"
        missing_pct = f"{RESULTS['missing_npmrds_pct']}%"

        card_excess = RoundedRectangle(
            width=5, height=1.6, corner_radius=0.12,
            stroke_color=COLOR_PREDICTION, stroke_width=2,
            fill_color=DARK_SLATE, fill_opacity=0.7,
        )
        card_excess.shift(LEFT * 3.2 + DOWN * 0.3)
        excess_num = Text(excess_pct, color=COLOR_PREDICTION, font_size=TITLE_FONT_SIZE)
        excess_label = Text("excess TMC segments", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE)
        excess_group = VGroup(excess_num, excess_label).arrange(DOWN, buff=0.15)
        excess_group.move_to(card_excess)

        card_missing = RoundedRectangle(
            width=5, height=1.6, corner_radius=0.12,
            stroke_color=TEAL, stroke_width=2,
            fill_color=DARK_SLATE, fill_opacity=0.7,
        )
        card_missing.shift(RIGHT * 3.2 + DOWN * 0.3)
        missing_num = Text(missing_pct, color=TEAL, font_size=TITLE_FONT_SIZE)
        missing_label = Text("missing NPMRDS coverage", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE)
        missing_group = VGroup(missing_num, missing_label).arrange(DOWN, buff=0.15)
        missing_group.move_to(card_missing)

        footnote = Text(
            "These gaps affect federal MAP-21 performance reporting.",
            color=SLATE, font_size=SMALL_FONT_SIZE,
        )
        footnote.to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text=(
                "But here's the mismatch. "
                f"Five point one one percent of TMC segments have no HPMS match at all, "
                f"and three point one zero percent of HPMS roads have no NPMRDS coverage. "
                "These aren't rounding errors. Under the MAP-21 federal law, states must "
                "report travel time reliability on the National Highway System. "
                "If your maps don't align, your performance numbers are wrong."
            )
        ) as tracker:
            self.play(
                FadeIn(card_excess), FadeIn(excess_group),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_MEDIUM)
            self.play(
                FadeIn(card_missing), FadeIn(missing_group),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_MEDIUM)
            self.play(FadeIn(footnote, shift=UP * 0.2), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # ── Beat 4: The question ──────────────────────────────────────
        question = Text(
            "How do you match segments across maps\n"
            "that don't share a coordinate system?",
            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            line_spacing=1.3,
        )
        question.move_to(ORIGIN)

        with self.voiceover(
            text=(
                "So the question becomes: how do you match segments across two maps "
                "that were never designed to talk to each other? "
                "They don't share IDs, they don't share a common segmentation, "
                "and their geometries only roughly overlap. "
                "That is the conflation problem."
            )
        ) as tracker:
            self.play(
                FadeOut(card_excess), FadeOut(excess_group),
                FadeOut(card_missing), FadeOut(missing_group),
                FadeOut(footnote),
                run_time=FAST_ANIM,
            )
            self.play(FadeIn(question, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

        # ── Fade out ──────────────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=NORMAL_ANIM,
        )
