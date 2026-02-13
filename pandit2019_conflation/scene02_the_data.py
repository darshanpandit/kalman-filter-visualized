"""Scene 2: The Data — Tour of NPMRDS and HPMS datasets.

Narrative lead: Tony (practitioner/skeptic)
Shows Fig 1 (segment length histograms) and Fig 2 (AADT distributions)
to highlight how fundamentally different these two datasets are.
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from pandit2019_conflation.data import RESULTS, STUDY_REGION, fig_path


class SceneTheData(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ─────────────────────────────────────────────────────
        title = Text(
            "The Data",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        # ── Beat 1: What NPMRDS actually is ───────────────────────────
        region_text = Text(
            f"{', '.join(STUDY_REGION['states'])}  |  "
            f"NPMRDS {STUDY_REGION['npmrds_year']}  |  "
            f"HPMS {STUDY_REGION['hpms_year']}",
            color=SLATE, font_size=SMALL_FONT_SIZE,
        )
        region_text.next_to(title, DOWN, buff=0.2)

        bullet_group = VGroup(
            Text("NPMRDS: probe vehicles report travel times", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
            Text("on TMC segments, every 5 minutes.", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
            Text("Think of it as crowdsourced speed data.", color=SLATE, font_size=SMALL_FONT_SIZE),
        )
        bullet_group.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        bullet_group.next_to(region_text, DOWN, buff=0.6)

        with self.voiceover(
            text=(
                "Let's look at what we're actually dealing with. "
                "The study covers Delaware, Maryland, and Washington D.C. "
                "NPMRDS is essentially crowdsourced speed data. "
                "GPS-equipped probe vehicles report travel times "
                "on TMC segments every five minutes. "
                "If you've ever looked at Google Maps traffic colors, "
                "it's the same idea, but for federal highway monitoring."
            )
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.play(FadeIn(region_text), run_time=FAST_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(
                FadeIn(bullet_group, shift=UP * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_LONG)

        self.play(FadeOut(bullet_group), FadeOut(region_text), run_time=FAST_ANIM)

        # ── Beat 2: Fig 1 — segment length histograms ────────────────
        fig1 = ImageMobject(fig_path("fig1_segment_histograms_fullwidth.png"))
        fig1.set_width(10.5)
        fig1.next_to(title, DOWN, buff=0.4)

        fig1_border = SurroundingRectangle(
            fig1, color=SLATE, buff=0.05, stroke_width=1,
        )

        caption1 = Text(
            "HPMS peaks at ~201 m.  NPMRDS segments can reach 20 km.",
            color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE,
        )
        caption1.next_to(fig1, DOWN, buff=0.25)

        with self.voiceover(
            text=(
                "But wait, why would matching be hard? "
                "Look at the segment length distributions. "
                "HPMS segments peak around 201 meters, "
                "which is about 10 chain lengths in surveyor units. "
                "But NPMRDS TMC segments can stretch up to 20 kilometers. "
                "One NPMRDS segment might overlap with dozens of HPMS records. "
                "It's not a one-to-one problem. It's a many-to-many mess."
            )
        ) as tracker:
            self.play(
                FadeIn(fig1), FadeIn(fig1_border),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_MEDIUM)
            self.play(FadeIn(caption1, shift=UP * 0.15), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        self.play(
            FadeOut(fig1), FadeOut(fig1_border), FadeOut(caption1),
            run_time=FAST_ANIM,
        )

        # ── Beat 3: Fig 2 — AADT distributions ───────────────────────
        fig2 = ImageMobject(fig_path("fig2_aadt_distributions.png"))
        fig2.set_width(9)
        fig2.next_to(title, DOWN, buff=0.4)

        fig2_border = SurroundingRectangle(
            fig2, color=SLATE, buff=0.05, stroke_width=1,
        )

        # Annotation cards for the exponential binning insight
        bin_small = Text(
            f"Smallest bin: {RESULTS['smallest_bin'][0]}"
            f"--{RESULTS['smallest_bin'][1]} AADT",
            color=TEAL, font_size=SMALL_FONT_SIZE,
        )
        bin_large = Text(
            f"Largest bin: {RESULTS['largest_bin'][0]:,}"
            f"--{RESULTS['largest_bin'][1]:,} AADT",
            color=COLOR_PREDICTION, font_size=SMALL_FONT_SIZE,
        )
        bin_info = VGroup(bin_small, bin_large).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        bin_info.next_to(fig2, DOWN, buff=0.25)

        with self.voiceover(
            text=(
                "Now here's a practical headache. "
                "Annual Average Daily Traffic, or AADT, spans an enormous range, "
                "from rural roads with maybe 500 vehicles a day "
                "to interstates carrying over 200,000. "
                "You can't put those on the same linear scale and compare them fairly. "
                "The paper uses exponential binning, 50 bins, "
                "where the smallest bin covers AADT 112 to 125, "
                "and the largest stretches from 246,000 to over 265,000. "
                "This ensures that each traffic class has enough samples "
                "for meaningful statistics."
            )
        ) as tracker:
            self.play(
                FadeIn(fig2), FadeIn(fig2_border),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_MEDIUM)
            self.play(FadeIn(bin_info, shift=UP * 0.15), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        self.play(
            FadeOut(fig2), FadeOut(fig2_border), FadeOut(bin_info),
            run_time=FAST_ANIM,
        )

        # ── Beat 4: Key insight ───────────────────────────────────────
        insight_lines = VGroup(
            Text(
                "These datasets were never designed",
                color=COLOR_TEXT, font_size=HEADING_FONT_SIZE,
            ),
            Text(
                "to talk to each other.",
                color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
            ),
        )
        insight_lines.arrange(DOWN, buff=0.2)
        insight_lines.move_to(ORIGIN)

        subtext = VGroup(
            Text("Different segmentation", color=SLATE, font_size=BODY_FONT_SIZE),
            Text("Different coordinate granularity", color=SLATE, font_size=BODY_FONT_SIZE),
            Text("Different update cycles", color=SLATE, font_size=BODY_FONT_SIZE),
        )
        subtext.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        subtext.next_to(insight_lines, DOWN, buff=0.6)

        with self.voiceover(
            text=(
                "Here's the core insight. "
                "These two datasets were never designed to talk to each other. "
                "Different segmentation schemes, "
                "different coordinate granularity, "
                "different update cycles. "
                "The NPMRDS data is from 2017, the HPMS from 2016. "
                "And yet, federal reporting demands that we combine them. "
                "So, we need a principled way to decide: "
                "which HPMS segment corresponds to which TMC segment? "
                "That's what the conflation algorithm will solve."
            )
        ) as tracker:
            self.play(FadeIn(insight_lines, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)
            self.play(
                LaggedStart(
                    *[FadeIn(s, shift=RIGHT * 0.3) for s in subtext],
                    lag_ratio=0.3,
                ),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_LONG * 2)

        # ── Fade out ──────────────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=NORMAL_ANIM,
        )
