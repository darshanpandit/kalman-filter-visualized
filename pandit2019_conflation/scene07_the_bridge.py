"""Scene 7: The Bridge — Grand unification of conflation and tracking.

The intellectual climax of the conflation chapter. Connects geospatial map
conflation (Pandit et al. 2019) to the Kalman filter / object tracking series,
showing they are the same mathematical problem. Includes the Frechet-Wasserstein
visual connection and the three-layer dissertation arc.

Voices: narrator (Jenny, chat), narrator_newscast, narrator rate=-15%,
        narrator_hopeful, darshan (Tony, friendly), darshan rate=-15%
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.azure import AzureService
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from pandit2019_conflation.data import TRACKING_CONNECTION, PAPER_INFO
from kalman_manim.mobjects.observation_note import make_observation_note


class SceneTheBridge(VoiceoverScene, MovingCameraScene):
    """Beat 7 — The Bridge."""

    def construct(self):
        # ── Voice setup ─────────────────────────────────────────────
        narrator = AzureService(voice="en-US-JennyNeural", style="chat")
        narrator_newscast = AzureService(voice="en-US-JennyNeural", style="newscast")
        narrator_hopeful = AzureService(voice="en-US-JennyNeural", style="hopeful")
        darshan = AzureService(voice="en-US-TonyNeural", style="friendly")
        self.set_speech_service(narrator_newscast)
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────
        title = Text(
            "The Bridge", color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        # ── Beat 1: Title + newscast hook ───────────────────────────
        with self.voiceover(
            text=(
                "Here is the connection that ties this entire "
                "dissertation together."
            ),
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Beat 2: Two-column comparison ───────────────────────────
        col_w, col_h = 5.0, 3.6

        def _make_column(header, color, lines, x_pos):
            box = RoundedRectangle(
                width=col_w, height=col_h, corner_radius=0.15,
                stroke_color=color, stroke_width=2,
                fill_color=DARK_SLATE, fill_opacity=0.6,
            )
            box.move_to([x_pos, -0.3, 0])
            hdr = Text(header, color=color, font_size=HEADING_FONT_SIZE)
            hdr.move_to(box.get_top() + DOWN * 0.4)
            items = VGroup(*[
                Text(txt, color=clr, font_size=fs)
                for txt, clr, fs in lines
            ])
            items.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            items.next_to(hdr, DOWN, buff=0.3)
            return box, hdr, items

        l_box, l_hdr, l_items = _make_column(
            "Map Conflation", COLOR_MEASUREMENT, [
                ("Match road segments", COLOR_TEXT, BODY_FONT_SIZE),
                ("across datasets", COLOR_TEXT, BODY_FONT_SIZE),
                ("Greedy best-match", COLOR_HIGHLIGHT, SMALL_FONT_SIZE),
                ("Minimize local score", SLATE, SMALL_FONT_SIZE),
            ], -3.2,
        )
        r_box, r_hdr, r_items = _make_column(
            "Object Tracking", TEAL, [
                ("Match bounding boxes", COLOR_TEXT, BODY_FONT_SIZE),
                ("across frames", COLOR_TEXT, BODY_FONT_SIZE),
                ("Hungarian algorithm", COLOR_HIGHLIGHT, SMALL_FONT_SIZE),
                ("Minimize global cost", SLATE, SMALL_FONT_SIZE),
            ], 3.2,
        )

        # Double arrows connecting parallel concepts between columns
        arrow_pairs = [(0, 0), (2, 2), (3, 3)]
        arrows = VGroup(*[
            DoubleArrow(
                l_items[i].get_right() + RIGHT * 0.15,
                r_items[j].get_left() + LEFT * 0.15,
                color=COLOR_HIGHLIGHT, stroke_width=2,
                tip_length=0.15, buff=0.05,
            )
            for i, j in arrow_pairs
        ])

        # ── Darshan introduces the connection ───────────────────────
        self.set_speech_service(darshan)

        with self.voiceover(
            text=(
                "When I was working on the smart crosswalk, SHAUM 703, "
                "I needed to track pedestrians across video frames. "
                "And I realized the structure of the problem was the "
                "same as what I'd already solved."
            ),
        ) as tracker:
            self.play(
                FadeIn(l_box), FadeIn(l_hdr),
                FadeIn(r_box), FadeIn(r_hdr),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(l_items, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(r_items, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        with self.voiceover(
            text=(
                "Match road segments across datasets. Match bounding boxes "
                "across frames. Both are assignment problems. My conflation "
                "algorithm does it greedily, best local match. SORT does it "
                "globally with the Hungarian algorithm. Same mathematical "
                "structure, different optimization."
            ),
        ) as tracker:
            self.play(
                LaggedStart(
                    *[GrowFromCenter(a) for a in arrows],
                    lag_ratio=0.25,
                ),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_LONG)

        # ── Narrator: "Same problem. Different solvers." ────────────
        self.set_speech_service(narrator)

        same_lbl = Text(
            "Both are assignment problems",
            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
        )
        same_lbl.to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="Same problem. Different solvers.",
        ) as tracker:
            self.play(FadeIn(same_lbl, shift=UP * 0.15), run_time=FAST_ANIM)
            self.wait(PAUSE_MEDIUM)

        # Fade columns
        cols = VGroup(l_box, l_hdr, l_items, r_box, r_hdr, r_items, arrows, same_lbl)
        self.play(FadeOut(cols), run_time=FAST_ANIM)

        # ── Beat 3: Frechet-Wasserstein side-by-side ────────────────
        # LEFT half: mini dog-walking animation
        person_anchors = [
            np.array([-5.2, 0.8, 0]),
            np.array([-4.0, 1.5, 0]),
            np.array([-3.0, 0.5, 0]),
            np.array([-2.0, 1.2, 0]),
        ]
        dog_anchors = [
            np.array([-5.2, -0.5, 0]),
            np.array([-4.0, -1.2, 0]),
            np.array([-3.0, -0.4, 0]),
            np.array([-2.0, -0.9, 0]),
        ]

        person_path = VMobject()
        person_path.set_points_smoothly(person_anchors)
        person_path.set_color(COLOR_PREDICTION)
        person_path.set_stroke(width=3)

        dog_path = VMobject()
        dog_path.set_points_smoothly(dog_anchors)
        dog_path.set_color(COLOR_MEASUREMENT)
        dog_path.set_stroke(width=3)

        person_dot = Dot(radius=0.09, color=CREAM).set_z_index(5)
        dog_dot = Dot(radius=0.09, color=COLOR_HIGHLIGHT).set_z_index(5)
        person_dot.move_to(person_path.point_from_proportion(0))
        dog_dot.move_to(dog_path.point_from_proportion(0))

        leash = always_redraw(
            lambda: DashedLine(
                person_dot.get_center(), dog_dot.get_center(),
                color=CREAM, stroke_width=1.5, stroke_opacity=0.6,
                dash_length=0.08,
            )
        )

        frechet_label = Text(
            "Frechet", color=COLOR_MEASUREMENT, font_size=BODY_FONT_SIZE,
        )
        frechet_sublabel = Text(
            "Optimal coupling along curves",
            color=SLATE, font_size=CHART_LABEL_FONT_SIZE,
        )
        frechet_labels = VGroup(frechet_label, frechet_sublabel)
        frechet_labels.arrange(DOWN, buff=0.1)
        frechet_labels.move_to(np.array([-3.6, -2.3, 0]))

        # RIGHT half: two blobs morphing toward each other
        blob_a_pts = [
            np.array([1.5, 0.8, 0]),
            np.array([2.8, 1.3, 0]),
            np.array([3.8, 0.5, 0]),
            np.array([3.2, -0.3, 0]),
            np.array([1.8, -0.1, 0]),
        ]
        blob_b_pts = [
            np.array([2.0, 0.3, 0]),
            np.array([3.0, 0.8, 0]),
            np.array([4.2, 0.2, 0]),
            np.array([3.5, -0.7, 0]),
            np.array([2.2, -0.5, 0]),
        ]

        blob_a = VMobject()
        blob_a.set_points_smoothly([*blob_a_pts, blob_a_pts[0]])
        blob_a.set_fill(COLOR_MEASUREMENT, opacity=0.25)
        blob_a.set_stroke(COLOR_MEASUREMENT, width=2.5)

        blob_b = VMobject()
        blob_b.set_points_smoothly([*blob_b_pts, blob_b_pts[0]])
        blob_b.set_fill(TEAL, opacity=0.25)
        blob_b.set_stroke(TEAL, width=2.5)

        # Target merged blob (average of a and b)
        blob_merged_pts = [
            (np.array(a) + np.array(b)) / 2
            for a, b in zip(blob_a_pts, blob_b_pts)
        ]
        blob_a_target = VMobject()
        blob_a_target.set_points_smoothly([*blob_merged_pts, blob_merged_pts[0]])
        blob_a_target.set_fill(COLOR_MEASUREMENT, opacity=0.25)
        blob_a_target.set_stroke(COLOR_MEASUREMENT, width=2.5)

        blob_b_target = VMobject()
        blob_b_target.set_points_smoothly([*blob_merged_pts, blob_merged_pts[0]])
        blob_b_target.set_fill(TEAL, opacity=0.25)
        blob_b_target.set_stroke(TEAL, width=2.5)

        wasserstein_label = Text(
            "Wasserstein", color=TEAL, font_size=BODY_FONT_SIZE,
        )
        wasserstein_sublabel = Text(
            "Optimal mass transport",
            color=SLATE, font_size=CHART_LABEL_FONT_SIZE,
        )
        wasserstein_labels = VGroup(wasserstein_label, wasserstein_sublabel)
        wasserstein_labels.arrange(DOWN, buff=0.1)
        wasserstein_labels.move_to(np.array([3.0, -2.3, 0]))

        # Divider line between left and right halves
        divider = DashedLine(
            np.array([0.0, 2.0, 0]), np.array([0.0, -1.8, 0]),
            color=SLATE, stroke_width=1, dash_length=0.12,
            stroke_opacity=0.4,
        )

        self.set_speech_service(narrator)

        with self.voiceover(
            text=(
                "And the connection goes deeper. Watch. On the left: "
                "the Frechet distance. You optimize how to couple points "
                "along two curves. On the right: the Wasserstein distance. "
                "You optimize how to transport mass between two "
                "distributions."
            ),
            prosody={"rate": "-15%"},
        ) as tracker:
            # Show paths and blobs
            self.play(
                Create(person_path), Create(dog_path),
                FadeIn(blob_a), FadeIn(blob_b),
                Create(divider),
                run_time=NORMAL_ANIM,
            )
            self.play(
                FadeIn(frechet_labels, shift=UP * 0.15),
                FadeIn(wasserstein_labels, shift=UP * 0.15),
                run_time=FAST_ANIM,
            )

            # Left: dog-walking animation
            self.play(
                FadeIn(person_dot, scale=1.5), FadeIn(dog_dot, scale=1.5),
                run_time=FAST_ANIM,
            )
            self.add(leash)

            progress = ValueTracker(0)
            person_dot.add_updater(
                lambda m: m.move_to(person_path.point_from_proportion(
                    np.clip(progress.get_value(), 0, 1)
                ))
            )
            dog_dot.add_updater(
                lambda m: m.move_to(dog_path.point_from_proportion(
                    np.clip(progress.get_value(), 0, 1)
                ))
            )

            # Animate both halves simultaneously
            self.play(
                progress.animate.set_value(1),
                Transform(blob_a, blob_a_target),
                Transform(blob_b, blob_b_target),
                run_time=4.0,
                rate_func=smooth,
            )
            self.wait(PAUSE_MEDIUM)

        # Clean up updaters
        person_dot.clear_updaters()
        dog_dot.clear_updaters()

        # ── Narrator: "Both ask the same question" ──────────────────
        with self.voiceover(
            text=(
                "Both ask the same question: what is the cheapest way "
                "to match one shape to another? Both optimize over all "
                "valid couplings to minimize cost."
            ),
        ) as tracker:
            self.wait(PAUSE_LONG)

        # ── Insight card ────────────────────────────────────────────
        # Fade side-by-side
        side_by_side = VGroup(
            person_path, dog_path, person_dot, dog_dot,
            frechet_labels, blob_a, blob_b, wasserstein_labels, divider,
        )
        self.remove(leash)
        self.play(FadeOut(side_by_side), run_time=FAST_ANIM)

        insight_card = RoundedRectangle(
            width=10.0, height=3.2, corner_radius=0.2,
            stroke_color=COLOR_FILTER_TF, stroke_width=2.5,
            fill_color=DARK_SLATE, fill_opacity=0.8,
        )
        insight_card.move_to(ORIGIN + DOWN * 0.2)

        insight_lines = VGroup(
            Text("Frechet: optimal coupling of curves.",
                 color=COLOR_MEASUREMENT, font_size=BODY_FONT_SIZE),
            Text("Wasserstein: optimal coupling of distributions.",
                 color=TEAL, font_size=BODY_FONT_SIZE),
            Text("Same mathematical structure:",
                 color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
            Text("optimize over couplings, minimize cost.",
                 color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE),
        )
        insight_lines.arrange(DOWN, buff=0.15).move_to(insight_card)

        self.play(FadeIn(insight_card), run_time=FAST_ANIM)
        self.play(
            LaggedStart(
                *[FadeIn(l, shift=UP * 0.15) for l in insight_lines],
                lag_ratio=0.3,
            ),
            run_time=SLOW_ANIM,
        )
        self.wait(PAUSE_LONG)

        # Fade card
        self.play(FadeOut(insight_card), FadeOut(insight_lines), run_time=FAST_ANIM)

        # ── Beat 4: Grand dissertation arc ──────────────────────────
        arc_data = [
            ("Data Infrastructure", f"{PAPER_INFO['authors']} {PAPER_INFO['year']}",
             "Get the map right", COLOR_MEASUREMENT),
            ("Signal Processing", "Parts 1-9 (Kalman Filters)",
             "Filter the noise", COLOR_HIGHLIGHT),
            ("Applied System", "SHAUM703 (Smart Crosswalks)",
             "Save lives", COLOR_PREDICTION),
        ]

        arc_cards = VGroup()
        for layer, source, purpose, color in arc_data:
            bg = RoundedRectangle(
                width=10.0, height=1.3, corner_radius=0.12,
                stroke_color=color, stroke_width=2.5,
                fill_color=DARK_SLATE, fill_opacity=0.75,
            )
            lbl = Text(layer, color=color, font_size=HEADING_FONT_SIZE)
            src = Text(source, color=COLOR_TEXT, font_size=SMALL_FONT_SIZE)
            prp = Text(purpose, color=SLATE, font_size=SMALL_FONT_SIZE)
            lbl.move_to(bg.get_left() + RIGHT * 2.0)
            src.move_to(bg)
            prp.move_to(bg.get_right() + LEFT * 1.5)
            arc_cards.add(VGroup(bg, lbl, src, prp))

        # arc_cards[0]=Data Infra (bottom), [1]=Signal (middle), [2]=Applied (top)
        # Arrange top-to-bottom visually: Applied, Signal, Data Infra
        arc_hierarchy = VGroup(arc_cards[2], arc_cards[1], arc_cards[0])
        arc_hierarchy.arrange(DOWN, buff=0.25).next_to(title, DOWN, buff=0.5)

        arc_arr1 = Arrow(
            arc_cards[0].get_top(), arc_cards[1].get_bottom(),
            color=SLATE, stroke_width=2, buff=0.08,
        )
        arc_arr2 = Arrow(
            arc_cards[1].get_top(), arc_cards[2].get_bottom(),
            color=SLATE, stroke_width=2, buff=0.08,
        )

        # ── Darshan narrates the arc, bottom-to-top ─────────────────
        self.set_speech_service(darshan)

        with self.voiceover(
            text=(
                "And so the dissertation arc becomes clear. At the "
                "foundation: data infrastructure. Get the map right. "
                "Without accurate road geometry, nothing downstream works."
            ),
        ) as tracker:
            self.play(FadeIn(arc_cards[0], shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        with self.voiceover(
            text=(
                "In the middle: signal processing. The Kalman filter "
                "series. Filter the noise."
            ),
        ) as tracker:
            self.play(GrowArrow(arc_arr1), run_time=FAST_ANIM)
            self.play(FadeIn(arc_cards[1], shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        with self.voiceover(
            text=(
                "At the top: the applied system. Smart crosswalks. "
                "Track pedestrians at intersections. Save lives."
            ),
        ) as tracker:
            self.play(GrowArrow(arc_arr2), run_time=FAST_ANIM)
            self.play(FadeIn(arc_cards[2], shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Bottom card (Data Infrastructure) stroke pulses ─────────
        self.play(
            arc_cards[0][0].animate.set_stroke(width=4),
            run_time=FAST_ANIM,
        )

        # Darshan delivers this line slowly
        with self.voiceover(
            text=(
                "Before you can track pedestrians, you need to know "
                "where the road is. Before you can filter signals, you "
                "need the right model of the world."
            ),
            prosody={"rate": "-15%"},
        ) as tracker:
            self.wait(PAUSE_LONG)

        # ── Citation note ───────────────────────────────────────────
        cite = make_observation_note(
            f"{PAPER_INFO['authors']} ({PAPER_INFO['year']}). "
            f"TRR {PAPER_INFO['volume']}, pp. {PAPER_INFO['pages']}.",
        )

        self.play(FadeIn(cite), run_time=FAST_ANIM)

        # ── Narrator hopeful: "This paper is the foundation layer." ─
        self.set_speech_service(narrator_hopeful)

        with self.voiceover(
            text=(
                "This paper is the foundation layer. "
                "The map under the map."
            ),
        ) as tracker:
            self.wait(PAUSE_LONG)

        # ── Fade out all ────────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=NORMAL_ANIM,
        )
