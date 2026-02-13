"""Scene 01: The Map Problem — Two maps, same highway, different shapes.

Two-voice format (Jenny = narrator, Tony = Darshan) using Azure TTS.
Introduces the fundamental conflation problem: NPMRDS and HPMS datasets
describe the same road network but disagree on geometry and coverage.

Voices: narrator (chat), narrator_newscast, darshan (friendly)

Requires: pip install "manim-voiceover[azure]"
          Set AZURE_SUBSCRIPTION_KEY and AZURE_SERVICE_REGION in .env
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.azure import AzureService
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from pandit2019_conflation.data import RESULTS, STUDY_REGION, fig_path


class SceneMapProblem(VoiceoverScene, MovingCameraScene):
    """Beat 1 — The Map Under the Map."""

    def construct(self):
        # ── Voice setup ─────────────────────────────────────────────
        narrator = AzureService(voice="en-US-JennyNeural", style="chat")
        narrator_newscast = AzureService(
            voice="en-US-JennyNeural", style="newscast",
        )
        darshan = AzureService(voice="en-US-TonyNeural", style="friendly")
        self.set_speech_service(narrator)
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────
        title = Text(
            "The Map Under the Map",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        # ── Opening hook ────────────────────────────────────────────
        self.set_speech_service(narrator)
        with self.voiceover(
            text="Two maps. Same highway. Different shapes.",
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Darshan introduction ────────────────────────────────────
        self.set_speech_service(darshan)
        with self.voiceover(
            text=(
                "This is my problem. The first paper in my dissertation. "
                "I worked on this at the University of Maryland with "
                "Kartik Kaushik and my advisor Cinzia Cirillo."
            ),
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Two panels: NPMRDS vs HPMS ──────────────────────────────
        panel_w, panel_h = 5.0, 3.0

        # --- NPMRDS panel (left, blue) ---
        npmrds_box = RoundedRectangle(
            width=panel_w, height=panel_h, corner_radius=0.15,
            stroke_color=COLOR_MEASUREMENT, stroke_width=2,
            fill_color=DARK_SLATE, fill_opacity=0.6,
        )
        npmrds_box.shift(LEFT * 3.0 + DOWN * 0.5)

        npmrds_header = Text(
            "NPMRDS", color=COLOR_MEASUREMENT, font_size=HEADING_FONT_SIZE,
        )
        npmrds_header.move_to(npmrds_box.get_top() + DOWN * 0.35)

        npmrds_bullets = VGroup(
            Text("Probe-based travel times", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
            Text("TMC segments", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
            Text("5-minute GPS epochs", color=SLATE, font_size=SMALL_FONT_SIZE),
        )
        npmrds_bullets.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        npmrds_bullets.next_to(npmrds_header, DOWN, buff=0.35)

        npmrds_panel = VGroup(npmrds_box, npmrds_header, npmrds_bullets)

        # --- HPMS panel (right, gold) ---
        hpms_box = RoundedRectangle(
            width=panel_w, height=panel_h, corner_radius=0.15,
            stroke_color=COLOR_HIGHLIGHT, stroke_width=2,
            fill_color=DARK_SLATE, fill_opacity=0.6,
        )
        hpms_box.shift(RIGHT * 3.0 + DOWN * 0.5)

        hpms_header = Text(
            "HPMS", color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
        )
        hpms_header.move_to(hpms_box.get_top() + DOWN * 0.35)

        hpms_bullets = VGroup(
            Text("Road attributes", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
            Text("AADT, lanes, class", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
            Text("Federal inventory", color=SLATE, font_size=SMALL_FONT_SIZE),
        )
        hpms_bullets.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        hpms_bullets.next_to(hpms_header, DOWN, buff=0.35)

        hpms_panel = VGroup(hpms_box, hpms_header, hpms_bullets)

        self.set_speech_service(darshan)
        with self.voiceover(
            text=(
                "The United States government maintains two national road "
                "datasets. The NPMRDS gives you travel times from GPS probes "
                "— how long it takes to cross each road segment. The HPMS "
                "gives you everything else — traffic volume, lanes, road "
                "classification."
            ),
        ) as tracker:
            self.play(
                FadeIn(npmrds_box), FadeIn(npmrds_header),
                FadeIn(hpms_box), FadeIn(hpms_header),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_SHORT)
            self.play(
                FadeIn(npmrds_bullets, shift=UP * 0.15),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_SHORT)
            self.play(
                FadeIn(hpms_bullets, shift=UP * 0.15),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_MEDIUM)

        # ── Narrator: MAP-21 mandate ────────────────────────────────
        self.set_speech_service(narrator)
        with self.voiceover(
            text=(
                "And the federal government needs both fused together. "
                "The MAP-21 Act requires it."
            ),
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Darshan: segments don't line up ─────────────────────────
        self.set_speech_service(darshan)
        with self.voiceover(
            text=(
                "Right. Congress mandated performance reporting on every "
                "NHS road. But the segments in these two datasets don't "
                "line up. The geometries disagree about where the road "
                "actually is."
            ),
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Fade panels, show statistics ────────────────────────────
        panels = VGroup(npmrds_panel, hpms_panel)
        self.play(FadeOut(panels), run_time=FAST_ANIM)

        excess_pct = f"{RESULTS['excess_tmc_pct']}%"
        missing_pct = f"{RESULTS['missing_npmrds_pct']}%"

        # Excess card (center-left, red)
        excess_num = Text(
            excess_pct, color=COLOR_PREDICTION, font_size=TITLE_FONT_SIZE,
        )
        excess_label = Text(
            "excess TMC segments", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
        )
        excess_group = VGroup(excess_num, excess_label).arrange(DOWN, buff=0.15)
        excess_group.shift(LEFT * 2.8 + DOWN * 0.3)

        # Missing card (center-right, teal)
        missing_num = Text(
            missing_pct, color=TEAL, font_size=TITLE_FONT_SIZE,
        )
        missing_label = Text(
            "missing NPMRDS coverage", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
        )
        missing_group = VGroup(missing_num, missing_label).arrange(DOWN, buff=0.15)
        missing_group.shift(RIGHT * 2.8 + DOWN * 0.3)

        self.set_speech_service(narrator_newscast)
        with self.voiceover(
            text=(
                "Five point one one percent of TMC segments have no matching "
                "NHS road. Three point one zero percent of NHS roads have no "
                "NPMRDS coverage."
            ),
        ) as tracker:
            self.play(
                FadeIn(excess_group, shift=UP * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_SHORT)
            self.play(
                FadeIn(missing_group, shift=UP * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_MEDIUM)

        # ── Darshan: scale concern ──────────────────────────────────
        self.set_speech_service(darshan)
        with self.voiceover(
            text=(
                "We studied Delaware, Maryland and DC. Those numbers are "
                "just three states. But the MAP-21 reporting covers the "
                "entire national highway system. If this is what three "
                "states look like, think about what's happening everywhere "
                "else."
            ),
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Fade stats ──────────────────────────────────────────────
        self.play(
            FadeOut(excess_group), FadeOut(missing_group),
            run_time=FAST_ANIM,
        )
        self.wait(PAUSE_MEDIUM)

        # ── Closing question ────────────────────────────────────────
        question = Text(
            "How do you match maps\nthat were never designed to agree?",
            color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
            line_spacing=1.3,
        )
        question.move_to(ORIGIN)

        self.set_speech_service(narrator)
        with self.voiceover(
            text=(
                "So the question becomes: how do you match segments across "
                "maps that were never designed to agree?"
            ),
        ) as tracker:
            self.play(FadeIn(question, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Fade out all except title ───────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob is not title],
            run_time=NORMAL_ANIM,
        )
        self.wait(PAUSE_LONG)

        # ── Final fade ──────────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=NORMAL_ANIM,
        )
