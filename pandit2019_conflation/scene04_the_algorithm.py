"""Scene 04: The Algorithm — Buffer search and 5-measure scoring system.

Azure multi-voice scene. Darshan walks through the practical methodology:
buffer search for candidates, five scoring measures with weights, and
composite score selection. Jenny provides narrator counterpoint.

Voices: narrator (Jenny, chat), narrator_whisper (Jenny, whispering),
        darshan (Tony, friendly), darshan_unfriendly (Tony, unfriendly).
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.azure import AzureService
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from pandit2019_conflation.data import RESULTS, SCORING_MEASURES, fig_path


class SceneTheAlgorithm(VoiceoverScene, MovingCameraScene):
    def construct(self):
        # ── Voice services ─────────────────────────────────────────
        narrator = AzureService(voice="en-US-JennyNeural", style="chat")
        narrator_whisper = AzureService(voice="en-US-JennyNeural", style="whispering")
        darshan = AzureService(voice="en-US-TonyNeural", style="friendly")
        darshan_unfriendly = AzureService(voice="en-US-TonyNeural", style="unfriendly")
        self.set_speech_service(darshan)
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────
        title = Text(
            "The Algorithm", color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text="Let me walk you through the whole thing."
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

        # ── Buffer search visualization ────────────────────────────
        # Main road segment (curved blue line)
        road_anchors = [
            np.array([-4.0, -0.3, 0]),
            np.array([-1.5, 0.5, 0]),
            np.array([1.0, -0.1, 0]),
            np.array([3.5, 0.6, 0]),
        ]
        road = VMobject()
        road.set_points_smoothly(road_anchors)
        road.set_color(COLOR_MEASUREMENT)
        road.set_stroke(width=6)

        road_label = Text(
            "TMC Segment", color=COLOR_MEASUREMENT, font_size=SMALL_FONT_SIZE,
        )
        road_label.next_to(road, DOWN, buff=0.15)

        # Buffer zone — thick translucent stroke around the road
        buffer_zone = road.copy()
        buffer_zone.set_stroke(
            color=COLOR_MEASUREMENT, width=80, opacity=0.12,
        )
        buffer_zone.set_fill(opacity=0)

        buffer_label = Text(
            "150 m buffer", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
        )
        buffer_label.next_to(road, UP, buff=1.2)

        # Candidate segments: 3 inside (green), 1 outside (grey)
        candidate_data = [
            ([(-3.8, 0.4), (-1.2, 1.0), (0.5, 0.6)], True),
            ([(-3.5, -0.9), (-1.0, -0.5), (1.5, -0.8)], True),
            ([(0.5, 0.9), (2.0, 0.3), (3.8, 0.8)], True),
            ([(-4.5, 1.9), (-2.0, 2.3), (0.0, 2.0)], False),
        ]

        candidates = VGroup()
        for anchors, is_inside in candidate_data:
            pts = [np.array([x, y, 0]) for x, y in anchors]
            seg = VMobject()
            seg.set_points_smoothly(pts)
            if is_inside:
                seg.set_color("#27ae60")
                seg.set_stroke(width=3)
            else:
                seg.set_color(SLATE)
                seg.set_stroke(width=2, opacity=0.4)
            candidates.add(seg)

        with self.voiceover(
            text=(
                "Step one: buffer search. Take a segment from the primary "
                "dataset. Build a hundred-fifty-meter buffer around it. "
                "Every segment from the other dataset that touches this "
                "buffer becomes a candidate."
            )
        ) as tracker:
            self.play(Create(road), FadeIn(road_label), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(
                FadeIn(buffer_zone), FadeIn(buffer_label),
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
            self.wait(PAUSE_SHORT)

        # ── Fade out buffer visualization ──────────────────────────
        buffer_group = VGroup(
            road, road_label, buffer_zone, buffer_label, candidates,
        )
        self.play(FadeOut(buffer_group), run_time=FAST_ANIM)

        # ── Figure 3: Buffer search from paper ─────────────────────
        fig3 = ImageMobject(fig_path("fig3_buffer_search.png"))
        fig3.scale_to_fit_width(9).move_to(ORIGIN + DOWN * 0.3)

        with self.voiceover(
            text=(
                "This is the I-95 and Capital Beltway interchange in "
                "Maryland. The blue line is our TMC segment. The purple "
                "is the buffer. The green lines are all the HPMS "
                "candidates it pulled in."
            )
        ) as tracker:
            self.play(FadeIn(fig3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Narrator interjection ──────────────────────────────────
        self.set_speech_service(narrator)

        with self.voiceover(
            text="That's a lot of candidates at an interchange."
        ) as tracker:
            self.wait(PAUSE_SHORT)

        # ── Darshan responds ───────────────────────────────────────
        self.set_speech_service(darshan)

        with self.voiceover(
            text=(
                "Exactly. And at interchanges is where you need "
                "scoring the most."
            )
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Fade out figure ────────────────────────────────────────
        self.play(FadeOut(fig3), run_time=FAST_ANIM)

        # ── Five scoring measure cards ─────────────────────────────
        sm = SCORING_MEASURES
        cards = VGroup()
        card_width = 2.15
        card_height = 2.9
        gap = 0.12

        for i in range(5):
            is_geo = sm["types"][i] == "geometric"
            card_color = TEAL if is_geo else COLOR_HIGHLIGHT
            type_label = "Geometric" if is_geo else "Semantic"

            bg = RoundedRectangle(
                width=card_width, height=card_height, corner_radius=0.15,
                fill_color=DARK_SLATE, fill_opacity=0.85,
                stroke_color=card_color, stroke_width=2,
            )

            name_text = Text(
                sm["names"][i], color=card_color,
                font_size=CHART_LABEL_FONT_SIZE,
            ).move_to(bg.get_top() + DOWN * 0.5)

            weight_badge_bg = RoundedRectangle(
                width=1.1, height=0.45, corner_radius=0.08,
                fill_color=card_color, fill_opacity=0.2,
                stroke_color=card_color, stroke_width=1.5,
            )
            weight_badge_text = Text(
                f"w = {sm['weights'][i]}",
                color=CREAM, font_size=CHART_LABEL_FONT_SIZE,
            )
            weight_badge_bg.next_to(name_text, DOWN, buff=0.25)
            weight_badge_text.move_to(weight_badge_bg)
            weight_badge = VGroup(weight_badge_bg, weight_badge_text)

            type_text = Text(
                type_label, color=SLATE, font_size=CHART_TICK_FONT_SIZE,
            ).next_to(weight_badge, DOWN, buff=0.2)

            symbol_text = Text(
                sm["symbols"][i], color=card_color,
                font_size=HEADING_FONT_SIZE,
            ).next_to(type_text, DOWN, buff=0.15)

            card = VGroup(bg, name_text, weight_badge, type_text, symbol_text)
            cards.add(card)

        cards.arrange(RIGHT, buff=gap)
        cards.next_to(title, DOWN, buff=0.5)
        # Scale to fit safe frame width if needed
        if cards.width > 11.4:
            cards.scale_to_fit_width(11.4)

        with self.voiceover(
            text=(
                "Five measures. Three geometric, two semantic. Angular "
                "parallelism tells you if segments run parallel — the sine "
                "of the absolute angle, scaled to a hundred. Frechet and "
                "Hausdorff measure shape similarity — we just covered those."
            )
        ) as tracker:
            self.play(
                LaggedStart(
                    *[FadeIn(card, shift=UP * 0.3) for card in cards],
                    lag_ratio=0.15,
                ),
                run_time=SLOW_ANIM,
            )
            self.wait(PAUSE_LONG)

        # ── Darshan on semantic weights ────────────────────────────
        with self.voiceover(
            text=(
                "Then Levenshtein distance for the text. Road number gets "
                "weight four — the highest weight in the entire system. "
                "I-95 is always I-95. It's the most reliable data in "
                "both datasets."
            )
        ) as tracker:
            # Pulse the Road Number card (index 3)
            self.play(
                cards[3][0].animate.set_stroke(width=4),
                run_time=FAST_ANIM,
            )
            self.wait(PAUSE_LONG)

        # ── Narrator asks about road name ──────────────────────────
        self.set_speech_service(narrator)

        with self.voiceover(
            text="And road name?"
        ) as tracker:
            self.wait(PAUSE_SHORT)

        # ── Darshan_unfriendly on road name ────────────────────────
        self.set_speech_service(darshan_unfriendly)

        with self.voiceover(
            text=(
                "Weight one. Road names in HPMS are a disaster. Missing, "
                "inconsistent, sometimes just the number repeated. "
                "I learned that the hard way."
            )
        ) as tracker:
            # Dim the Road Name card (index 4) border to emphasize low weight
            self.play(
                cards[4][0].animate.set_stroke(opacity=0.5),
                run_time=FAST_ANIM,
            )
            self.wait(PAUSE_LONG)

        # ── Highlight high-weight cards ────────────────────────────
        self.set_speech_service(darshan)

        # Indices with weight >= 3: 0 (Angle, w=3), 1 (Frechet, w=3), 3 (Road Number, w=4)
        high_weight_indices = [i for i, w in enumerate(sm["weights"]) if w >= 3]

        # Build highlight animations: pulse borders of high-weight cards
        highlight_anims = []
        for idx in high_weight_indices:
            highlight_anims.append(
                cards[idx][0].animate.set_stroke(
                    color=COLOR_HIGHLIGHT, width=4,
                )
            )

        best_match_text = Text(
            "Best match = min(Score)",
            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
        )
        best_match_text.next_to(cards, DOWN, buff=0.5)

        with self.voiceover(
            text=(
                "Weighted sum of all five. Pick the candidate with the "
                "lowest score. That's your best match."
            )
        ) as tracker:
            self.play(*highlight_anims, run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(best_match_text, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Narrator whisper close ─────────────────────────────────
        self.set_speech_service(narrator_whisper)

        with self.voiceover(
            text="A weighted sum. Simple. But effective."
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Fade out ───────────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob is not title],
            run_time=NORMAL_ANIM,
        )
        self.wait(PAUSE_MEDIUM)
        self.play(FadeOut(title), run_time=FAST_ANIM)
