"""Scene 6: The Blind Spot — Local matching ignores network topology.

The intellectual pivot: the algorithm matches segments independently, but roads
form a graph.  At intersections, independent matches can produce inconsistent
junctions.  This sets up the graph-matching idea in Scene 07.
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *


def _make_network(positions, edges, color, offset):
    """Build a small graph (dots + lines) shifted by offset."""
    off = np.array([*offset, 0])
    pts = [np.array([*p, 0]) + off for p in positions]
    dots = VGroup(*[Dot(point=p, radius=0.08, color=color) for p in pts])
    lines = VGroup(*[
        Line(pts[i], pts[j], stroke_color=color, stroke_width=3.5)
        for i, j in edges
    ])
    return dots, lines, VGroup(lines, dots)


class SceneBlindSpot(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Beat 1: Title + opening question ─────────────────────────
        title = Text(
            "The Blind Spot", color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        question = Text(
            "But here's something the algorithm can't see.",
            color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
        )
        question.next_to(title, DOWN, buff=0.8)

        with self.voiceover(
            text=(
                "But here's something the algorithm can't see. "
                "Everything we've built so far treats each road segment "
                "as an independent curve. Match it, score it, move on. "
                "That works surprisingly well. But it has a blind spot."
            )
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(question, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        self.play(FadeOut(question), run_time=FAST_ANIM)

        # ── Beat 2: Two overlapping road networks ────────────────────
        nodes = [
            [-2.5, 1.0],   # 0: top-left
            [0.0, 1.5],    # 1: top-center (junction)
            [2.5, 1.0],    # 2: top-right
            [0.0, -1.0],   # 3: bottom-center (junction)
            [-2.0, -1.5],  # 4: bottom-left
        ]
        edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 2)]

        _, _, net_a = _make_network(nodes, edges, COLOR_MEASUREMENT, [-0.15, 0.1])
        _, _, net_b = _make_network(nodes, edges, COLOR_PREDICTION, [0.15, -0.1])
        networks = VGroup(net_a, net_b).shift(DOWN * 0.5)

        label_a = Text("NPMRDS", color=COLOR_MEASUREMENT, font_size=SMALL_FONT_SIZE)
        label_b = Text("HPMS", color=COLOR_PREDICTION, font_size=SMALL_FONT_SIZE)
        label_a.next_to(networks, LEFT, buff=0.3).shift(UP * 0.5)
        label_b.next_to(networks, RIGHT, buff=0.3).shift(DOWN * 0.5)

        with self.voiceover(
            text=(
                "Roads are not isolated curves. They form a network: "
                "a graph, with intersections as nodes and road segments as edges. "
                "Here are two overlapping networks. The blue is NPMRDS, "
                "the red is HPMS. They describe the same roads, but they're "
                "slightly offset, as real datasets always are."
            )
        ) as tracker:
            self.play(FadeIn(net_a, shift=LEFT * 0.2), run_time=NORMAL_ANIM)
            self.play(FadeIn(label_a), run_time=FAST_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(net_b, shift=RIGHT * 0.2), run_time=NORMAL_ANIM)
            self.play(FadeIn(label_b), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # ── Beat 3: Highlight the junction problem ───────────────────
        junc_a = np.array([nodes[1][0] - 0.15, nodes[1][1] + 0.1, 0]) + DOWN * 0.5
        junc_b = np.array([nodes[1][0] + 0.15, nodes[1][1] - 0.1, 0]) + DOWN * 0.5

        ring_a = Circle(radius=0.35, stroke_color=COLOR_MEASUREMENT, stroke_width=3)
        ring_a.move_to(junc_a)
        ring_b = Circle(radius=0.35, stroke_color=COLOR_PREDICTION, stroke_width=3)
        ring_b.move_to(junc_b)

        gap_line = DashedLine(
            junc_a, junc_b,
            stroke_color=COLOR_HIGHLIGHT, stroke_width=2.5, dash_length=0.1,
        )
        gap_label = Text(
            "Same intersection,\ndifferent positions",
            color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE, line_spacing=1.2,
        )
        gap_label.next_to(VGroup(ring_a, ring_b), UP, buff=0.3)

        with self.voiceover(
            text=(
                "Look at this junction where three roads meet. "
                "The algorithm matches each edge independently. "
                "It finds that this blue edge matches that red edge. Good. "
                "And this blue edge matches another red edge. Also good. "
                "But the matched edges meet at different points in the two networks. "
                "The junction doesn't line up. The matches are locally correct "
                "but globally inconsistent."
            )
        ) as tracker:
            self.play(Create(ring_a), Create(ring_b), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(Create(gap_line), run_time=NORMAL_ANIM)
            self.play(FadeIn(gap_label, shift=DOWN * 0.2), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG * 2)

        self.play(
            FadeOut(ring_a), FadeOut(ring_b),
            FadeOut(gap_line), FadeOut(gap_label),
            FadeOut(networks), FadeOut(label_a), FadeOut(label_b),
            run_time=FAST_ANIM,
        )

        # ── Beat 4: The insight ──────────────────────────────────────
        insight_box = RoundedRectangle(
            width=10, height=2.2, corner_radius=0.15,
            stroke_color=TEAL, stroke_width=2,
            fill_color=DARK_SLATE, fill_opacity=0.7,
        )
        insight_box.move_to(ORIGIN + DOWN * 0.3)

        insight_text = VGroup(
            Text("Roads are not isolated curves", color=COLOR_TEXT, font_size=HEADING_FONT_SIZE),
            Text("they form a network.", color=TEAL, font_size=HEADING_FONT_SIZE),
            Text("Topology carries information that local features ignore.",
                 color=SLATE, font_size=SMALL_FONT_SIZE),
        ).arrange(DOWN, buff=0.2).move_to(insight_box)

        with self.voiceover(
            text=(
                "Roads are not isolated curves. They form a network. "
                "The algorithm treats each segment independently, and that is "
                "its strength: it scales, it's simple, it works. "
                "But topology, how segments connect at intersections, "
                "carries information that local features completely ignore."
            )
        ) as tracker:
            self.play(FadeIn(insight_box), run_time=NORMAL_ANIM)
            self.play(FadeIn(insight_text, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

        # ── Beat 5: Tease ───────────────────────────────────────────
        tease = Text(
            "What if we could match the networks\nas graphs, not just segments?",
            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE, line_spacing=1.3,
        )
        tease.move_to(ORIGIN + DOWN * 0.3)

        with self.voiceover(
            text=(
                "What if we could match the networks as graphs, "
                "not just as collections of independent segments? "
                "That's where the mathematics gets really interesting."
            )
        ) as tracker:
            self.play(FadeOut(insight_box), FadeOut(insight_text), run_time=FAST_ANIM)
            self.play(FadeIn(tease, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

        # ── Fade out ─────────────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=NORMAL_ANIM,
        )
