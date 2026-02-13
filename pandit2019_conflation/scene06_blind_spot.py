"""Scene 6: The Blind Spot — Local matching ignores network topology (merged with topology).

MERGED beat: combines the old "Blind Spot" and "From Curves to Graphs" into one
continuous scene. Per Nolan's direction: NO full fadeout between the blind spot
revelation and the graph-matching answer. The blind spot bleeds directly into
the answer.

Voices: narrator (Jenny, chat), narrator rate=-10%, narrator_hopeful,
        darshan (Tony, friendly)
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.azure import AzureService
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from pandit2019_conflation.data import MATH_HIERARCHY


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
    """Beat 6 — The Blind Spot (merged with topology)."""

    def construct(self):
        # ── Voice setup ─────────────────────────────────────────────
        narrator = AzureService(voice="en-US-JennyNeural", style="chat")
        narrator_hopeful = AzureService(voice="en-US-JennyNeural", style="hopeful")
        darshan = AzureService(voice="en-US-TonyNeural", style="friendly")
        self.set_speech_service(narrator)
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────
        title = Text(
            "The Blind Spot", color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        # ── Beat 1: Opening — "something your algorithm cannot see" ─
        with self.voiceover(
            text="But here's something your algorithm cannot see.",
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Beat 2: Two overlapping road networks ───────────────────
        nodes = [
            [-2.5, 1.0],   # 0: top-left
            [0.0, 1.5],    # 1: top-center (junction — 3 roads meet)
            [2.5, 1.0],    # 2: top-right
            [0.0, -1.0],   # 3: bottom-center
            [-2.0, -1.5],  # 4: bottom-left
        ]
        edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 2)]

        dots_a, _, net_a = _make_network(nodes, edges, COLOR_MEASUREMENT, [-0.15, 0.1])
        dots_b, _, net_b = _make_network(nodes, edges, COLOR_PREDICTION, [0.15, -0.1])
        networks = VGroup(net_a, net_b).shift(DOWN * 0.5)

        label_a = Text("NPMRDS", color=COLOR_MEASUREMENT, font_size=SMALL_FONT_SIZE)
        label_b = Text("HPMS", color=COLOR_PREDICTION, font_size=SMALL_FONT_SIZE)
        label_a.next_to(networks, LEFT, buff=0.3).shift(UP * 0.5)
        label_b.next_to(networks, RIGHT, buff=0.3).shift(DOWN * 0.5)

        with self.voiceover(
            text=(
                "Your algorithm matches each segment independently. "
                "For each primary segment, find the best-scoring candidate. "
                "Move on to the next."
            ),
        ) as tracker:
            self.play(
                FadeIn(net_a, shift=LEFT * 0.2),
                FadeIn(label_a),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_SHORT)
            self.play(
                FadeIn(net_b, shift=RIGHT * 0.2),
                FadeIn(label_b),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_MEDIUM)

        # ── Beat 3: Highlight the junction problem ──────────────────
        # Junction node index 1: where 3 roads meet (edges 0-1, 1-2, 1-3)
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
        gap_label.next_to(networks, DOWN, buff=0.4)

        with self.voiceover(
            text=(
                "But roads don't exist in isolation. They form a network. "
                "At this junction, three roads meet. If you match each edge "
                "independently, the matched segments might not connect "
                "properly at the node."
            ),
        ) as tracker:
            self.play(Create(ring_a), Create(ring_b), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(Create(gap_line), run_time=NORMAL_ANIM)
            self.play(FadeIn(gap_label, shift=DOWN * 0.2), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # ── Darshan defends: "The matches are still locally correct." ─
        self.set_speech_service(darshan)

        with self.voiceover(
            text="The matches are still locally correct.",
        ) as tracker:
            self.wait(PAUSE_SHORT)

        # ── Narrator: "Locally, yes. But globally..." ───────────────
        self.set_speech_service(narrator)

        with self.voiceover(
            text=(
                "Locally, yes. But globally, you've broken the topology. "
                "The graph structure carries information that your greedy, "
                "segment-by-segment approach cannot capture."
            ),
            prosody={"rate": "-10%"},
        ) as tracker:
            self.wait(PAUSE_LONG)

        # ── Insight box ─────────────────────────────────────────────
        # Fade junction highlights but keep networks visible
        self.play(
            FadeOut(ring_a), FadeOut(ring_b),
            FadeOut(gap_line), FadeOut(gap_label),
            run_time=FAST_ANIM,
        )

        insight_box = RoundedRectangle(
            width=10, height=2.8, corner_radius=0.15,
            stroke_color=TEAL, stroke_width=2,
            fill_color=DARK_SLATE, fill_opacity=0.7,
        )
        insight_box.next_to(networks, DOWN, buff=0.4)

        insight_text = VGroup(
            Text("Roads are not isolated curves", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
            Text("they form a network.", color=TEAL, font_size=BODY_FONT_SIZE),
            Text("Topology carries information", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
            Text("that local features ignore.", color=TEAL, font_size=BODY_FONT_SIZE),
        ).arrange(DOWN, buff=0.15).move_to(insight_box)

        self.play(FadeIn(insight_box), run_time=NORMAL_ANIM)
        self.play(FadeIn(insight_text, shift=UP * 0.2), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG)

        # ── Darshan's defense: scale argument ───────────────────────
        self.set_speech_service(darshan)

        with self.voiceover(
            text=(
                "I know. I knew that when I wrote it. The algorithm was "
                "designed for the national scale. A hundred fifty thousand "
                "miles of N H S. It had to be fast. Local measures scale "
                "linearly. Graph matching is combinatorial. Not feasible "
                "at that size. Not in twenty nineteen."
            ),
        ) as tracker:
            self.wait(PAUSE_LONG)

        # ── Narrator hopeful: "But the mathematics doesn't care..." ─
        self.set_speech_service(narrator_hopeful)

        with self.voiceover(
            text=(
                "But the mathematics doesn't care about scale. The question "
                "is still there. What would a graph-aware version look like?"
            ),
        ) as tracker:
            self.wait(PAUSE_LONG)

        # ══════════════════════════════════════════════════════════════
        # CRITICAL MERGE POINT: fade networks and insight box, keep title.
        # No new title. Just continue.
        # ══════════════════════════════════════════════════════════════
        self.play(
            FadeOut(networks), FadeOut(label_a), FadeOut(label_b),
            FadeOut(insight_box), FadeOut(insight_text),
            run_time=NORMAL_ANIM,
        )

        # ── Beat 4: Topology enters — Kim et al. 2022 ──────────────
        self.set_speech_service(narrator)

        # Build two simple road networks (4 nodes, 5 edges each)
        src_pos = [
            np.array([-4.5, 0.8, 0]),   # A
            np.array([-2.5, 1.6, 0]),    # B
            np.array([-2.5, -0.2, 0]),   # C
            np.array([-0.8, 0.7, 0]),    # D
        ]
        tgt_pos = [
            np.array([0.8, 0.8, 0]),    # A'
            np.array([2.8, 1.7, 0]),    # B'
            np.array([2.8, -0.3, 0]),   # C'
            np.array([4.5, 0.6, 0]),    # D'
        ]
        src_edges_idx = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)]
        tgt_edges_idx = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)]
        src_names = ["A", "B", "C", "D"]
        tgt_names = ["A'", "B'", "C'", "D'"]

        def build_network(positions, edge_indices, names, color):
            n_grp = VGroup()
            dots = []
            for i, (pos, name) in enumerate(zip(positions, names)):
                dot = Dot(pos, radius=0.12, color=color).set_z_index(5)
                lbl = Text(name, color=COLOR_TEXT, font_size=CHART_LABEL_FONT_SIZE)
                lbl.next_to(dot, DOWN, buff=0.1)
                dots.append(dot)
                n_grp.add(dot, lbl)
            e_grp = VGroup()
            for i, j in edge_indices:
                e_grp.add(Line(
                    positions[i], positions[j],
                    color=color, stroke_width=3,
                ))
            return dots, n_grp, e_grp

        s_dots, s_ngrp, s_egrp = build_network(
            src_pos, src_edges_idx, src_names, COLOR_MEASUREMENT,
        )
        t_dots, t_ngrp, t_egrp = build_network(
            tgt_pos, tgt_edges_idx, tgt_names, TEAL,
        )

        # Shift everything down from title
        topo_group = VGroup(s_ngrp, s_egrp, t_ngrp, t_egrp).shift(DOWN * 0.3)

        s_label = Text(
            "Source", color=COLOR_MEASUREMENT, font_size=SMALL_FONT_SIZE,
        )
        s_label.next_to(s_egrp, DOWN, buff=0.5)
        t_label = Text(
            "Target", color=TEAL, font_size=SMALL_FONT_SIZE,
        )
        t_label.next_to(t_egrp, DOWN, buff=0.5)

        with self.voiceover(
            text=(
                "In twenty twenty-two, Kim and colleagues answered exactly "
                "that. Their algorithm, A P S G, matches road networks by "
                "exploiting graph topology."
            ),
        ) as tracker:
            self.play(
                LaggedStart(*[Create(e) for e in s_egrp], lag_ratio=0.2),
                run_time=NORMAL_ANIM,
            )
            self.play(FadeIn(s_ngrp), FadeIn(s_label), run_time=FAST_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(
                LaggedStart(*[Create(e) for e in t_egrp], lag_ratio=0.2),
                run_time=NORMAL_ANIM,
            )
            self.play(FadeIn(t_ngrp), FadeIn(t_label), run_time=FAST_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Node-matching arrows (gold) ─────────────────────────────
        match_arrows = VGroup(*[
            Arrow(
                s_dots[i].get_center(), t_dots[i].get_center(),
                color=COLOR_HIGHLIGHT, stroke_width=2, buff=0.15,
                max_tip_length_to_length_ratio=0.15,
            )
            for i in range(4)
        ])

        with self.voiceover(
            text=(
                "Instead of matching edges in isolation, you match nodes "
                "first, preserving degree, the number of connections. "
                "A four-way intersection maps to a four-way intersection. "
                "Then the edges between matched nodes are determined "
                "automatically."
            ),
        ) as tracker:
            self.play(
                LaggedStart(
                    *[GrowArrow(a) for a in match_arrows],
                    lag_ratio=0.15,
                ),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_LONG)

        # ── Darshan: "That's elegant." ──────────────────────────────
        self.set_speech_service(darshan)

        with self.voiceover(
            text=(
                "That's elegant. Using the structure to constrain the "
                "matching. I wish I'd had that in twenty nineteen."
            ),
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Fade networks ───────────────────────────────────────────
        net_grp = VGroup(
            s_ngrp, s_egrp, s_label,
            t_ngrp, t_egrp, t_label,
            match_arrows,
        )
        self.play(FadeOut(net_grp), run_time=FAST_ANIM)

        # ── Beat 5: Three-level hierarchy ───────────────────────────
        self.set_speech_service(narrator)

        h = MATH_HIERARCHY
        levels_data = [
            (h["local"], COLOR_MEASUREMENT),
            (h["topology"], TEAL),
            (h["optimal_transport"], COLOR_FILTER_TF),
        ]

        cards = VGroup()
        for info, color in levels_data:
            methods_str = ", ".join(info["methods"][:3])
            if len(info["methods"]) > 3:
                methods_str += ", ..."
            bg = RoundedRectangle(
                width=10.0, height=1.35, corner_radius=0.15,
                stroke_color=color, stroke_width=2.5,
                fill_color=DARK_SLATE, fill_opacity=0.75,
            )
            nm = Text(info["label"], color=color, font_size=HEADING_FONT_SIZE)
            ds = Text(methods_str, color=COLOR_TEXT, font_size=SMALL_FONT_SIZE)
            pp = Text(info["paper"], color=SLATE, font_size=CHART_LABEL_FONT_SIZE)
            nm.move_to(bg.get_left() + RIGHT * 2.2)
            ds.next_to(nm, RIGHT, buff=0.6)
            pp.move_to(bg.get_right() + LEFT * 1.5)
            cards.add(VGroup(bg, nm, ds, pp))

        # Visual order top-to-bottom: OT, topology, local
        # cards[0]=local, cards[1]=topology, cards[2]=OT
        hierarchy = VGroup(cards[2], cards[1], cards[0])
        hierarchy.arrange(DOWN, buff=0.3).next_to(title, DOWN, buff=0.5)

        arr_bottom_mid = Arrow(
            cards[0].get_top(), cards[1].get_bottom(),
            color=SLATE, stroke_width=2, buff=0.08,
        )
        arr_mid_top = Arrow(
            cards[1].get_top(), cards[2].get_bottom(),
            color=SLATE, stroke_width=2, buff=0.08,
        )

        # Appear bottom-to-top with narration
        with self.voiceover(
            text=(
                "And this reveals a hierarchy. At the bottom: local features. "
                "Segment-by-segment comparison. That's your twenty nineteen paper."
            ),
        ) as tracker:
            self.play(FadeIn(cards[0], shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        with self.voiceover(
            text=(
                "In the middle: graph topology. Exploit the network "
                "structure. That's twenty twenty-two."
            ),
        ) as tracker:
            self.play(GrowArrow(arr_bottom_mid), run_time=FAST_ANIM)
            self.play(FadeIn(cards[1], shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        with self.voiceover(
            text=(
                "At the top: optimal transport. Treat the entire road "
                "network as a distribution and find the cheapest way to "
                "transform one into the other."
            ),
            prosody={"rate": "-10%"},
        ) as tracker:
            self.play(GrowArrow(arr_mid_top), run_time=FAST_ANIM)
            self.play(FadeIn(cards[2], shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Top card pulses, lower cards fade ───────────────────────
        self.play(
            cards[2][0].animate.set_stroke(width=4),
            cards[1][0].animate.set_stroke(opacity=0.4),
            cards[0][0].animate.set_stroke(opacity=0.4),
            run_time=NORMAL_ANIM,
        )

        # ── Darshan: "We started at the bottom." ────────────────────
        self.set_speech_service(darshan)

        with self.voiceover(
            text=(
                "We started at the bottom. And honestly, starting at the "
                "bottom was the right move. You have to understand the "
                "local structure before you can reason about the global one."
            ),
        ) as tracker:
            self.wait(PAUSE_LONG)

        # ── Narrator: "And now you can see where the mathematics leads."
        self.set_speech_service(narrator)

        with self.voiceover(
            text="And now you can see where the mathematics leads.",
        ) as tracker:
            self.wait(PAUSE_LONG)

        # ── Fade out all except title, then title ───────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob is not title],
            run_time=NORMAL_ANIM,
        )
        self.wait(PAUSE_LONG)
        self.play(FadeOut(title), run_time=NORMAL_ANIM)
