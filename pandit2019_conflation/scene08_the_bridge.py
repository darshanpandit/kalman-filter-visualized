"""Scene 8: The Bridge — Grand unification of conflation and tracking.

The intellectual climax of the conflation chapter. Connects geospatial map
conflation (Pandit et al. 2019) to the Kalman filter / object tracking series,
showing they are the same mathematical problem. Ends with the dissertation arc:
Data Infrastructure -> Signal Processing -> Applied System.
"""
from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from pandit2019_conflation.data import TRACKING_CONNECTION, PAPER_INFO
from kalman_manim.mobjects.observation_note import make_observation_note


class SceneTheBridge(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        title = Text("The Bridge", color=COLOR_TEXT, font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3).set_z_index(10)

        # ── Beat 1: Title ────────────────────────────────────────────
        with self.voiceover(
            text="Here is the connection that ties this entire dissertation together."
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Beat 2: Two columns — Conflation vs Tracking ────────────
        col_w, col_h = 5.0, 3.6

        def make_column(label, color, items, x_pos):
            box = RoundedRectangle(width=col_w, height=col_h, corner_radius=0.15,
                                   stroke_color=color, stroke_width=2,
                                   fill_color=DARK_SLATE, fill_opacity=0.6)
            box.move_to([x_pos, -0.3, 0])
            hdr = Text(label, color=color, font_size=HEADING_FONT_SIZE)
            hdr.move_to(box.get_top() + DOWN * 0.4)
            lines = VGroup(*[Text(t, color=c, font_size=fs) for t, c, fs in items])
            lines.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            lines.next_to(hdr, DOWN, buff=0.3)
            return box, hdr, lines

        l_box, l_hdr, l_items = make_column("Map Conflation", COLOR_MEASUREMENT, [
            ("Match road segments", COLOR_TEXT, BODY_FONT_SIZE),
            ("across datasets", COLOR_TEXT, BODY_FONT_SIZE),
            ("Min-cost assignment", COLOR_HIGHLIGHT, SMALL_FONT_SIZE),
            ("Minimize total", SLATE, SMALL_FONT_SIZE),
            ("matching score", SLATE, SMALL_FONT_SIZE),
        ], -3.2)

        r_box, r_hdr, r_items = make_column("Object Tracking", TEAL, [
            ("Match bounding boxes", COLOR_TEXT, BODY_FONT_SIZE),
            ("across frames", COLOR_TEXT, BODY_FONT_SIZE),
            ("Hungarian algorithm", COLOR_HIGHLIGHT, SMALL_FONT_SIZE),
            ("Minimize total", SLATE, SMALL_FONT_SIZE),
            ("IoU cost", SLATE, SMALL_FONT_SIZE),
        ], 3.2)

        with self.voiceover(
            text="On the left: map conflation. Match road segments across "
                 "two datasets using minimum-cost assignment. Minimize the "
                 "total matching score. "
                 "On the right: object tracking. Match bounding boxes across "
                 "video frames using the Hungarian algorithm. Minimize the "
                 "total IoU cost."
        ) as tracker:
            self.play(FadeIn(l_box), FadeIn(l_hdr), FadeIn(r_box), FadeIn(r_hdr),
                      run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(l_items, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(r_items, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # Connecting arrows between matching concepts
        pairs = [(0, 0), (2, 2), (3, 3)]  # Match segments, algorithm, minimize
        arrows = VGroup(*[
            DoubleArrow(l_items[i].get_right() + RIGHT*0.15,
                        r_items[j].get_left() + LEFT*0.15,
                        color=COLOR_HIGHLIGHT, stroke_width=2, tip_length=0.15, buff=0.05)
            for i, j in pairs
        ])
        same_lbl = Text(TRACKING_CONNECTION["shared_math"],
                         color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE)
        same_lbl.to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="Conflation and tracking are the same mathematical problem. "
                 "Both assign elements from one set to another. "
                 "Both minimize a cost function. "
                 "The Hungarian algorithm that SORT uses to match detections "
                 "to tracks is the same optimal assignment framework "
                 "that underlies map conflation."
        ) as tracker:
            self.play(LaggedStart(*[GrowFromCenter(a) for a in arrows], lag_ratio=0.25),
                      run_time=NORMAL_ANIM)
            self.play(FadeIn(same_lbl, shift=UP * 0.15), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        cols = VGroup(l_box, l_hdr, l_items, r_box, r_hdr, r_items, arrows, same_lbl)
        self.play(FadeOut(cols), run_time=FAST_ANIM)

        # ── Beat 3: Frechet-Wasserstein connection ───────────────────
        fw_box = RoundedRectangle(width=10.0, height=2.8, corner_radius=0.2,
                                   stroke_color=COLOR_FILTER_TF, stroke_width=2.5,
                                   fill_color=DARK_SLATE, fill_opacity=0.8)
        fw_box.move_to(DOWN * 0.2)
        fw_lines = VGroup(
            Text("Frechet distance measures curve similarity.",
                 color=COLOR_MEASUREMENT, font_size=BODY_FONT_SIZE),
            Text("Wasserstein distance measures distribution similarity.",
                 color=COLOR_FILTER_TF, font_size=BODY_FONT_SIZE),
            Text("Both are optimal transport: moving mass from",
                 color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
            Text("one configuration to another at minimum cost.",
                 color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE),
        )
        fw_lines.arrange(DOWN, buff=0.15).move_to(fw_box)

        with self.voiceover(
            text="And the connection goes deeper. "
                 "Frechet distance measures how similar two curves are. "
                 "The 2-Wasserstein distance measures how similar two "
                 "probability distributions are. "
                 "They are both instances of optimal transport: "
                 "moving mass from one configuration to another at minimum cost. "
                 "The Frechet distance is, in fact, the Levy-Frechet metric "
                 "applied to curves, just as Wasserstein is applied to distributions."
        ) as tracker:
            self.play(FadeIn(fw_box), run_time=FAST_ANIM)
            self.play(LaggedStart(*[FadeIn(l, shift=UP*0.15) for l in fw_lines],
                                  lag_ratio=0.3), run_time=SLOW_ANIM)
            self.wait(PAUSE_LONG * 2)

        self.play(FadeOut(fw_box), FadeOut(fw_lines), run_time=FAST_ANIM)

        # ── Beat 4: Grand dissertation arc ───────────────────────────
        arc_data = [
            ("Applied System", "SHAUM703 (Smart Crosswalks)", "Save lives", COLOR_PREDICTION),
            ("Signal Processing", "Parts 1-9 (Kalman Filters)", "Filter the noise", COLOR_HIGHLIGHT),
            ("Data Infrastructure", "Pandit et al. 2019 (Conflation)", "Get the map right", COLOR_MEASUREMENT),
        ]
        arc_cards = VGroup()
        for layer, source, purpose, color in arc_data:
            bg = RoundedRectangle(width=10.0, height=1.3, corner_radius=0.12,
                                  stroke_color=color, stroke_width=2.5,
                                  fill_color=DARK_SLATE, fill_opacity=0.75)
            lbl = Text(layer, color=color, font_size=HEADING_FONT_SIZE)
            src = Text(source, color=COLOR_TEXT, font_size=SMALL_FONT_SIZE)
            prp = Text(purpose, color=SLATE, font_size=SMALL_FONT_SIZE)
            lbl.move_to(bg.get_left() + RIGHT * 2.0)
            src.move_to(bg)
            prp.move_to(bg.get_right() + LEFT * 1.5)
            arc_cards.add(VGroup(bg, lbl, src, prp))
        arc_cards.arrange(DOWN, buff=0.25).next_to(title, DOWN, buff=0.5)

        a1 = Arrow(arc_cards[2].get_top(), arc_cards[1].get_bottom(), color=SLATE, stroke_width=2, buff=0.08)
        a2 = Arrow(arc_cards[1].get_top(), arc_cards[0].get_bottom(), color=SLATE, stroke_width=2, buff=0.08)

        with self.voiceover(
            text="And so the dissertation arc becomes clear. "
                 "At the foundation: data infrastructure. "
                 "Pandit et al. 2019. Get the map right. "
                 "Without accurate road geometry, nothing downstream works. "
                 "In the middle: signal processing. "
                 "The Kalman filter series, parts 1 through 9. "
                 "Filter the noise from GPS, from sensors, from the world. "
                 "At the top: the applied system. "
                 "Smart crosswalks. SHAUM 703. Save lives."
        ) as tracker:
            self.play(FadeIn(arc_cards[2], shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(GrowArrow(a1), run_time=FAST_ANIM)
            self.play(FadeIn(arc_cards[1], shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(GrowArrow(a2), run_time=FAST_ANIM)
            self.play(FadeIn(arc_cards[0], shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Beat 5: Citation + closing ───────────────────────────────
        cite = make_observation_note(
            f"{PAPER_INFO['authors']} ({PAPER_INFO['year']}). "
            f"TRR {PAPER_INFO['volume']}, pp. {PAPER_INFO['pages']}")

        with self.voiceover(
            text="Before you can track pedestrians, you need to know where "
                 "the road is. Before you can filter signals, you need the "
                 "right model of the world. "
                 "This paper, Pandit, Kaushik, and Cirillo, 2019, "
                 "is the foundation layer. The data infrastructure "
                 "that everything else depends on."
        ) as tracker:
            self.play(FadeIn(cite), run_time=FAST_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(arc_cards[2][0].animate.set_stroke(width=4), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG * 2)

        # ── Fade out ─────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
