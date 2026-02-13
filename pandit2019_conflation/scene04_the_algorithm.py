"""Scene 04: The Algorithm — Buffer search and 5-measure scoring system.

Practical methodology walkthrough: how conflation candidates are found
via buffer search and ranked with a weighted composite score.
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from pandit2019_conflation.data import *


class SceneTheAlgorithm(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ─────────────────────────────────────────────────────
        title = Text(
            "The Algorithm", color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text="Here is how the conflation actually works, step by step. "
                 "Given a road segment from one dataset, "
                 "how do we find its best match in the other?"
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Step 1: Buffer search visualization ───────────────────────
        step_lbl = Text(
            "Step 1: Buffer Search", color=COLOR_HIGHLIGHT,
            font_size=HEADING_FONT_SIZE,
        )
        step_lbl.next_to(title, DOWN, buff=0.4)

        # Main road segment (thick blue line)
        road_anchors = [
            np.array([-4.0, -0.5, 0]),
            np.array([-1.5, 0.3, 0]),
            np.array([1.0, -0.2, 0]),
            np.array([3.5, 0.4, 0]),
        ]
        road = VMobject()
        road.set_points_smoothly(road_anchors)
        road.set_color(COLOR_MEASUREMENT)
        road.set_stroke(width=6)

        road_label = Text(
            "TMC Segment", color=COLOR_MEASUREMENT, font_size=SMALL_FONT_SIZE,
        )
        road_label.next_to(road, DOWN, buff=0.15)

        with self.voiceover(
            text="Start with a TMC segment, a traffic monitoring road segment. "
                 "We need to search for matching HPMS segments nearby."
        ) as tracker:
            self.play(
                FadeIn(step_lbl),
                Create(road), FadeIn(road_label),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_SHORT)

        # Buffer zone (translucent expanded region)
        # Create a thick transparent stroke around the road to simulate a buffer
        buffer_zone = road.copy()
        buffer_zone.set_stroke(
            color=COLOR_MEASUREMENT, width=80, opacity=0.12,
        )
        buffer_zone.set_fill(opacity=0)

        buffer_label = Text(
            "150m buffer", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
        )
        buffer_label.next_to(road, UP, buff=1.2)

        # Candidate segments (green lines appearing inside buffer)
        candidate_data = [
            ([(-3.8, 0.3), (-1.2, 0.9), (0.5, 0.5)], True),
            ([(-3.5, -1.0), (-1.0, -0.6), (1.5, -0.9)], True),
            ([(0.5, 0.8), (2.0, 0.2), (3.8, 0.7)], True),
            ([(-4.5, 1.8), (-2.0, 2.2), (0.0, 1.9)], False),  # outside
        ]

        candidates = VGroup()
        inside_label_shown = False
        for anchors, is_inside in candidate_data:
            pts = [np.array([x, y, 0]) for x, y in anchors]
            seg = VMobject()
            seg.set_points_smoothly(pts)
            if is_inside:
                seg.set_color("#27ae60")  # green
                seg.set_stroke(width=3)
            else:
                seg.set_color(SLATE)
                seg.set_stroke(width=2, opacity=0.4)
            candidates.add(seg)

        with self.voiceover(
            text="Draw a 150-meter buffer around that segment. "
                 "Any HPMS segments that fall within this buffer "
                 "become candidates for matching. "
                 "Segments outside the buffer are ignored entirely, "
                 "saving massive computation."
        ) as tracker:
            self.play(
                FadeIn(buffer_zone),
                FadeIn(buffer_label),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_SHORT)
            self.play(
                LaggedStart(
                    *[Create(c) for c in candidates],
                    lag_ratio=0.3,
                ),
                run_time=SLOW_ANIM,
            )
            self.wait(PAUSE_SHORT)
            # Dim the outside candidate
            self.play(
                candidates[-1].animate.set_stroke(opacity=0.15),
                run_time=FAST_ANIM,
            )
            self.wait(PAUSE_MEDIUM)

        # ── Show Fig 3 from the paper ─────────────────────────────────
        buffer_group = VGroup(
            road, road_label, buffer_zone, buffer_label, candidates, step_lbl,
        )
        self.play(FadeOut(buffer_group), run_time=FAST_ANIM)

        fig3 = ImageMobject(fig_path("fig3_buffer_search.png"))
        fig3.scale_to_fit_width(9).move_to(ORIGIN + DOWN * 0.3)

        with self.voiceover(
            text="Here is a real example from the paper: "
                 "the I-95 and I-495 interchange near Washington D.C. "
                 "Notice how the buffer captures nearby candidates "
                 "without an exhaustive search over the entire network."
        ) as tracker:
            self.play(FadeIn(fig3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        self.play(FadeOut(fig3), run_time=FAST_ANIM)

        # ── Step 2: The 5 scoring measures ────────────────────────────
        step2_lbl = Text(
            "Step 2: Score Each Candidate", color=COLOR_HIGHLIGHT,
            font_size=HEADING_FONT_SIZE,
        )
        step2_lbl.next_to(title, DOWN, buff=0.4)

        sm = SCORING_MEASURES

        # Build 5 cards in a row
        cards = VGroup()
        card_width = 2.2
        card_height = 2.8
        total_width = 5 * card_width + 4 * 0.15
        start_x = -total_width / 2 + card_width / 2

        for i in range(5):
            is_geo = sm["types"][i] == "geometric"
            card_color = TEAL if is_geo else COLOR_HIGHLIGHT
            type_label = "Geometric" if is_geo else "Semantic"

            bg = RoundedRectangle(
                width=card_width, height=card_height, corner_radius=0.15,
                fill_color=DARK_SLATE, fill_opacity=0.85,
                stroke_color=card_color, stroke_width=2,
            )
            x_pos = start_x + i * (card_width + 0.15)
            bg.move_to(np.array([x_pos, -0.4, 0]))

            name_text = Text(
                sm["names"][i], color=card_color,
                font_size=CHART_LABEL_FONT_SIZE,
            ).move_to(bg.get_top() + DOWN * 0.45)

            weight_text = Text(
                f"Weight: {sm['weights'][i]}",
                color=CREAM, font_size=CHART_LABEL_FONT_SIZE,
            ).next_to(name_text, DOWN, buff=0.3)

            type_text = Text(
                type_label, color=SLATE, font_size=CHART_TICK_FONT_SIZE,
            ).next_to(weight_text, DOWN, buff=0.2)

            symbol_text = Text(
                sm["symbols"][i], color=card_color,
                font_size=HEADING_FONT_SIZE,
            ).next_to(type_text, DOWN, buff=0.2)

            card = VGroup(bg, name_text, weight_text, type_text, symbol_text)
            cards.add(card)

        with self.voiceover(
            text="Once we have candidates, we score each one on five measures. "
                 "Three are geometric: angular parallelism, "
                 "Frechet distance, and Hausdorff distance. "
                 "Two are semantic: road number matching and road name matching, "
                 "both using Levenshtein edit distance."
        ) as tracker:
            self.play(FadeIn(step2_lbl), run_time=FAST_ANIM)
            self.play(
                LaggedStart(
                    *[FadeIn(card, shift=UP * 0.3) for card in cards],
                    lag_ratio=0.15,
                ),
                run_time=SLOW_ANIM,
            )
            self.wait(PAUSE_LONG)

        # Highlight the weights
        with self.voiceover(
            text="Notice the weights. Road number gets the highest weight of 4 "
                 "because it is the most consistent attribute across datasets. "
                 "Road name gets only 1 because it is often inconsistent in HPMS. "
                 "Frechet and angular parallelism each get 3, "
                 "and Hausdorff gets 2."
        ) as tracker:
            for i, w in enumerate(sm["weights"]):
                if w >= 3:
                    self.play(
                        cards[i][0].animate.set_stroke(width=4),
                        cards[i][2].animate.set_color(COLOR_HIGHLIGHT),
                        run_time=0.3,
                    )
            self.wait(PAUSE_LONG)

        # ── Step 3: The composite score equation ──────────────────────
        self.play(
            FadeOut(cards), FadeOut(step2_lbl),
            run_time=FAST_ANIM,
        )

        step3_lbl = Text(
            "Step 3: Select Best Match", color=COLOR_HIGHLIGHT,
            font_size=HEADING_FONT_SIZE,
        )
        step3_lbl.next_to(title, DOWN, buff=0.4)

        # Score equation as Text
        eq_line1 = Text(
            "Score = 3 * Angle + 3 * Frechet + 2 * Hausdorff",
            color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
        )
        eq_line2 = Text(
            "         + 4 * RoadNum + 1 * RoadName",
            color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
        )
        eq_line2.next_to(eq_line1, DOWN, buff=0.15, aligned_edge=LEFT)
        eq_group = VGroup(eq_line1, eq_line2).move_to(ORIGIN + UP * 0.3)

        best_match = Text(
            "Best match = candidate with minimum Score",
            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
        )
        best_match.next_to(eq_group, DOWN, buff=0.6)

        # Visual explanation box
        note_bg = RoundedRectangle(
            width=9, height=1.2, corner_radius=0.15,
            fill_color=DARK_SLATE, fill_opacity=0.7,
            stroke_color=TEAL, stroke_width=1.5,
        )
        note_bg.next_to(best_match, DOWN, buff=0.4)
        note_text = Text(
            "All measures are normalized to [0,1] before weighting.\n"
            "Lower score = better geometric + semantic alignment.",
            color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
        )
        note_text.move_to(note_bg)
        note = VGroup(note_bg, note_text)

        with self.voiceover(
            text="Each measure is normalized to a zero-to-one scale, "
                 "then multiplied by its weight and summed. "
                 "The candidate with the lowest composite score wins. "
                 "It is simple, interpretable, and effective."
        ) as tracker:
            self.play(FadeIn(step3_lbl), run_time=FAST_ANIM)
            self.play(
                FadeIn(eq_group, shift=UP * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_SHORT)
            self.play(
                FadeIn(best_match, shift=UP * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(note), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # Final voiceover summarizing the pipeline
        with self.voiceover(
            text="So the full pipeline is: buffer search to find candidates, "
                 "compute five similarity measures, weight and combine them, "
                 "pick the minimum score. "
                 "Elegant in its simplicity."
        ) as tracker:
            self.wait(PAUSE_LONG)

        # ── Fade out ──────────────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=NORMAL_ANIM,
        )
