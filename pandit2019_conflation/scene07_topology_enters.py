"""Scene 7: Topology Enters — From curves to graphs, and the mathematical hierarchy.

Introduces Kim et al. (2022) APSG approach, showing how graph topology improves
conflation beyond local curve comparison. Builds the three-level hierarchy:
Local Features -> Graph Topology -> Optimal Transport.
"""
from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from pandit2019_conflation.data import MATH_HIERARCHY


class SceneTopologyEnters(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        title = Text("From Curves to Graphs", color=COLOR_TEXT, font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3).set_z_index(10)

        # ── Beat 1: Title + hook ─────────────────────────────────────
        with self.voiceover(
            text="In 2022, Kim and colleagues took a fundamentally different "
                 "approach to map conflation. Instead of comparing individual "
                 "curve segments, they asked: what if we match the network "
                 "structure itself?"
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Beat 2: Two road networks with topology-aware matching ───
        src_pos = {"A": LEFT*4.5+UP, "B": LEFT*2.5+UP*1.8, "C": LEFT*2.5+DOWN*0.2, "D": LEFT*0.8+UP*0.8}
        tgt_pos = {"A'": RIGHT*0.8+UP, "B'": RIGHT*2.8+UP*1.9, "C'": RIGHT*2.8+DOWN*0.3, "D'": RIGHT*4.5+UP*0.7}
        src_edges = [("A","B"), ("A","C"), ("B","D"), ("C","D")]
        tgt_edges = [("A'","B'"), ("A'","C'"), ("B'","D'"), ("C'","D'")]

        def build_net(positions, edges, color):
            nodes = {}
            n_grp, e_grp = VGroup(), VGroup()
            for name, pos in positions.items():
                dot = Dot(pos, radius=0.12, color=color).set_z_index(5)
                lbl = Text(name, color=COLOR_TEXT, font_size=CHART_LABEL_FONT_SIZE)
                lbl.next_to(dot, DOWN, buff=0.1)
                nodes[name] = dot
                n_grp.add(dot, lbl)
            for u, v in edges:
                e_grp.add(Line(nodes[u].get_center(), nodes[v].get_center(),
                               color=color, stroke_width=3))
            return nodes, n_grp, e_grp

        s_nodes, s_ngrp, s_egrp = build_net(src_pos, src_edges, COLOR_MEASUREMENT)
        t_nodes, t_ngrp, t_egrp = build_net(tgt_pos, tgt_edges, TEAL)
        s_lbl = Text("Source network", color=COLOR_MEASUREMENT, font_size=SMALL_FONT_SIZE)
        s_lbl.next_to(s_egrp, DOWN, buff=0.5)
        t_lbl = Text("Target network", color=TEAL, font_size=SMALL_FONT_SIZE)
        t_lbl.next_to(t_egrp, DOWN, buff=0.5)

        with self.voiceover(
            text="Here are two road networks. The local approach from Pandit "
                 "et al. would compare each edge independently using Frechet "
                 "and Hausdorff distances. But topology-aware matching does "
                 "something smarter. It matches nodes first, preserving the "
                 "degree of each node. Then the edges follow naturally."
        ) as tracker:
            self.play(LaggedStart(*[Create(e) for e in s_egrp], lag_ratio=0.2), run_time=NORMAL_ANIM)
            self.play(FadeIn(s_ngrp), FadeIn(s_lbl), run_time=FAST_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(LaggedStart(*[Create(e) for e in t_egrp], lag_ratio=0.2), run_time=NORMAL_ANIM)
            self.play(FadeIn(t_ngrp), FadeIn(t_lbl), run_time=FAST_ANIM)
            self.wait(PAUSE_MEDIUM)

        # Node-matching arrows
        match_arrows = VGroup(*[
            Arrow(s_nodes[s].get_center(), t_nodes[t].get_center(),
                  color=COLOR_HIGHLIGHT, stroke_width=2, buff=0.15,
                  max_tip_length_to_length_ratio=0.15)
            for s, t in [("A","A'"), ("B","B'"), ("C","C'"), ("D","D'")]
        ])
        degree_note = Text("Degree preserved: each node keeps its connection count",
                           color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE)
        degree_note.to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="Notice: node A has degree 2 in both networks. Node D also "
                 "has degree 2. This structural constraint dramatically "
                 "reduces the search space compared to matching edges blindly."
        ) as tracker:
            self.play(LaggedStart(*[GrowArrow(a) for a in match_arrows], lag_ratio=0.15),
                      run_time=NORMAL_ANIM)
            self.play(FadeIn(degree_note, shift=UP * 0.15), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        net_grp = VGroup(s_ngrp, s_egrp, s_lbl, t_ngrp, t_egrp, t_lbl,
                         match_arrows, degree_note)
        self.play(FadeOut(net_grp), run_time=FAST_ANIM)

        # ── Beat 3: Three-level hierarchy ────────────────────────────
        h = MATH_HIERARCHY
        levels = [
            (h["optimal_transport"], COLOR_FILTER_TF),
            (h["topology"], TEAL),
            (h["local"], COLOR_MEASUREMENT),
        ]
        cards = VGroup()
        for info, color in levels:
            bg = RoundedRectangle(width=9.0, height=1.35, corner_radius=0.15,
                                  stroke_color=color, stroke_width=2.5,
                                  fill_color=DARK_SLATE, fill_opacity=0.75)
            nm = Text(info["label"], color=color, font_size=HEADING_FONT_SIZE)
            ds = Text(info["what"], color=COLOR_TEXT, font_size=SMALL_FONT_SIZE)
            pp = Text(info["paper"], color=SLATE, font_size=CHART_LABEL_FONT_SIZE)
            nm.move_to(bg.get_left() + RIGHT * 2.2)
            ds.next_to(nm, RIGHT, buff=0.6)
            pp.move_to(bg.get_right() + LEFT * 1.3)
            cards.add(VGroup(bg, nm, ds, pp))
        cards.arrange(DOWN, buff=0.3).next_to(title, DOWN, buff=0.5)

        arr1 = Arrow(cards[2].get_top(), cards[1].get_bottom(), color=SLATE, stroke_width=2, buff=0.08)
        arr2 = Arrow(cards[1].get_top(), cards[0].get_bottom(), color=SLATE, stroke_width=2, buff=0.08)

        with self.voiceover(
            text="This gives us a mathematical hierarchy with three levels. "
                 "At the bottom, local features: Frechet, Hausdorff, angular "
                 "parallelism. This is Pandit et al. 2019. "
                 "In the middle, graph topology: the APSG approach of Kim et al. "
                 "2022. Exploit network connectivity. "
                 "And at the top, optimal transport: treat the entire network "
                 "as a probability distribution and find the cheapest way "
                 "to transform one into the other."
        ) as tracker:
            self.play(FadeIn(cards[2], shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(GrowArrow(arr1), run_time=FAST_ANIM)
            self.play(FadeIn(cards[1], shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(GrowArrow(arr2), run_time=FAST_ANIM)
            self.play(FadeIn(cards[0], shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Beat 4: Highlight optimal transport frontier ─────────────
        with self.voiceover(
            text="Local features compare curves. Topology exploits connectivity. "
                 "But the ultimate framework, optimal transport, treats the "
                 "entire network as a probability distribution and finds the "
                 "cheapest way to transform one into the other. "
                 "That is the mathematical frontier of map conflation."
        ) as tracker:
            self.play(cards[0][0].animate.set_stroke(width=4),
                      cards[1][0].animate.set_stroke(opacity=0.4),
                      cards[2][0].animate.set_stroke(opacity=0.4),
                      run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

        # ── Fade out ─────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
