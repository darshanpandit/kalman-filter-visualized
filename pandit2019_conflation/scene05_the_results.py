"""Scene 5: The Results — Gumbel distributions, coverage gaps, discrepancy maps.

Shows the algorithm's output: Gumbel score distributions (extreme-value behavior),
excess/missing coverage maps, and key quantitative findings (5.11% excess, 3.10% missing).
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from pandit2019_conflation.data import RESULTS, fig_path


class SceneTheResults(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Beat 1: Title ────────────────────────────────────────────
        title = Text(
            "The Results", color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text=(
                "We ran the algorithm in both directions: "
                "NPMRDS to HPMS, and HPMS to NPMRDS. "
                "Let's look at what the scores tell us."
            )
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Beat 2: Gumbel distributions ─────────────────────────────
        fig_gumbel = ImageMobject(fig_path("fig6_gumbel_distributions.png"))
        fig_gumbel.scale_to_fit_width(9.5)
        fig_gumbel.next_to(title, DOWN, buff=0.4)

        gumbel_caption = Text(
            "Gumbel (Extreme Value Type I) distribution of matching scores",
            color=SLATE, font_size=SMALL_FONT_SIZE,
        )
        gumbel_caption.next_to(fig_gumbel, DOWN, buff=0.2)

        with self.voiceover(
            text=(
                "The matching scores follow a Gumbel distribution, also known as "
                "Extreme Value Type I. Why Gumbel? Because for each source segment, "
                "we pick the minimum score among all candidates within the buffer. "
                "Taking the minimum of many values is an extreme value operation, "
                "and the Gumbel distribution is exactly what governs that behavior. "
                "Notice the long right tail. Most matches are good, with low scores "
                "near the mode. But a few are terrible, way out in the tail. "
                "Those outliers are the discrepancies we are looking for."
            )
        ) as tracker:
            self.play(FadeIn(fig_gumbel, shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(gumbel_caption), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG * 2)

        self.play(
            FadeOut(fig_gumbel), FadeOut(gumbel_caption), run_time=FAST_ANIM,
        )

        # ── Beat 3: Excess and missing maps ──────────────────────────
        fig_maps = ImageMobject(fig_path("fig5_excess_missing_maps.png"))
        fig_maps.scale_to_fit_width(10.0)
        fig_maps.next_to(title, DOWN, buff=0.4)

        maps_caption = Text(
            "Yellow = NHS roads    Black = NPMRDS coverage",
            color=SLATE, font_size=SMALL_FONT_SIZE,
        )
        maps_caption.next_to(fig_maps, DOWN, buff=0.2)

        with self.voiceover(
            text=(
                "Here is the coverage picture. The yellow lines are the National "
                "Highway System roads. The black lines are NPMRDS segments overlaid "
                "on top. Where they don't align, we have a problem. "
                "Some regions have thick NPMRDS coverage but the NHS geometry "
                "curves away. Other stretches of highway simply have no NPMRDS "
                "data at all. These are the gaps that the algorithm must find."
            )
        ) as tracker:
            self.play(FadeIn(fig_maps, shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(maps_caption), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG * 2)

        self.play(
            FadeOut(fig_maps), FadeOut(maps_caption), run_time=FAST_ANIM,
        )

        # ── Beat 4: Algorithm results map ────────────────────────────
        fig_results = ImageMobject(fig_path("fig7_combined.png"))
        fig_results.scale_to_fit_width(10.5)
        fig_results.next_to(title, DOWN, buff=0.35)

        with self.voiceover(
            text=(
                "And here is what the algorithm found. "
                "Red segments are excess: TMC segments in NPMRDS that have no "
                "matching NHS road in HPMS. "
                "Purple segments are missing: NHS roads that have no NPMRDS "
                "coverage at all. "
                "The numbers: five point one one percent excess, "
                "three point one zero percent missing."
            )
        ) as tracker:
            self.play(FadeIn(fig_results, shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

        # Key numbers overlay
        excess_pct = f"{RESULTS['excess_tmc_pct']}%"
        missing_pct = f"{RESULTS['missing_npmrds_pct']}%"

        stat_box = RoundedRectangle(
            width=8.5, height=1.4, corner_radius=0.12,
            stroke_color=SLATE, stroke_width=1.5,
            fill_color=BG_COLOR, fill_opacity=0.9,
        )
        stat_box.to_edge(DOWN, buff=0.35)

        excess_text = Text(
            f"Excess (red):  {excess_pct}",
            color=COLOR_PREDICTION, font_size=BODY_FONT_SIZE,
        )
        missing_text = Text(
            f"Missing (purple):  {missing_pct}",
            color="#9b59b6", font_size=BODY_FONT_SIZE,
        )
        stats = VGroup(excess_text, missing_text).arrange(RIGHT, buff=1.5)
        stats.move_to(stat_box)

        with self.voiceover(
            text=(
                "These numbers matter for a practical reason. "
                "MAP-21 requires every state to report travel time reliability "
                "on the National Highway System. If your underlying map is wrong, "
                "if segments are excess or missing, then your congestion metrics "
                "are wrong too. Getting the map right is not optional. "
                "It is a federal requirement."
            )
        ) as tracker:
            self.play(
                FadeIn(stat_box), FadeIn(stats), run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_LONG * 2)

        # ── Fade out ─────────────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=NORMAL_ANIM,
        )
