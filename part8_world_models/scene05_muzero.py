"""Part 8, Scene 5: MuZero — Learned Models for Planning.

Data: Conceptual diagrams + published Atari/board game results

MuZero learns three functions: representation h (encoder), dynamics g
(transition in latent space), and prediction f (policy + value from
latent state). It then uses MCTS in latent space for planning — without
knowing the game rules. Matches AlphaZero on Go, chess, shogi while
also mastering Atari.

Papers:
- Schrittwieser et al. (2020, Nature) — MuZero
- Silver et al. (2018, Science) — AlphaZero (requires known rules)
- Silver et al. (2017, Nature) — AlphaGo Zero
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.comparison_table import ComparisonTable
from kalman_manim.mobjects.observation_note import make_observation_note
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneMuZero(VoiceoverScene, MovingCameraScene):
    """MuZero: learned model for planning via MCTS in latent space.

    Visual: Three functions diagram + MCTS concept + results table.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="MuZero asks a radical question: what if we don't even "
                 "need to know the rules of the game? It learns a world "
                 "model purely from experience and uses it for planning — "
                 "matching AlphaZero without any prior knowledge."
        ) as tracker:
            title = Text(
                "MuZero: Learned Models for Planning",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── The three learned functions ───────────────────────────────
        with self.voiceover(
            text="MuZero learns three functions. First, the representation "
                 "function h: it encodes a raw observation into a latent "
                 "state. Second, the dynamics function g: given a latent "
                 "state and an action, it predicts the next latent state "
                 "and the immediate reward. Third, the prediction function "
                 "f: from any latent state, it outputs a policy and a value "
                 "estimate."
        ) as tracker:
            # Representation h
            h_box = RoundedRectangle(
                corner_radius=0.1, width=3.5, height=1.0,
                color=COLOR_MEASUREMENT, fill_opacity=0.2, stroke_width=2,
            )
            h_title = Text("h: Representation", color=COLOR_MEASUREMENT,
                           font_size=20)
            h_detail = Text("observation -> latent state",
                            color=SLATE, font_size=14)
            h_group = VGroup(h_title, h_detail).arrange(DOWN, buff=0.05)
            h_group.move_to(h_box)
            h_block = VGroup(h_box, h_group)

            # Dynamics g
            g_box = RoundedRectangle(
                corner_radius=0.1, width=3.5, height=1.0,
                color=COLOR_PREDICTION, fill_opacity=0.2, stroke_width=2,
            )
            g_title = Text("g: Dynamics", color=COLOR_PREDICTION,
                           font_size=20)
            g_detail = Text("(state, action) -> next state, reward",
                            color=SLATE, font_size=14)
            g_group = VGroup(g_title, g_detail).arrange(DOWN, buff=0.05)
            g_group.move_to(g_box)
            g_block = VGroup(g_box, g_group)

            # Prediction f
            f_box = RoundedRectangle(
                corner_radius=0.1, width=3.5, height=1.0,
                color=COLOR_POSTERIOR, fill_opacity=0.2, stroke_width=2,
            )
            f_title = Text("f: Prediction", color=COLOR_POSTERIOR,
                           font_size=20)
            f_detail = Text("state -> policy, value",
                            color=SLATE, font_size=14)
            f_group = VGroup(f_title, f_detail).arrange(DOWN, buff=0.05)
            f_group.move_to(f_box)
            f_block = VGroup(f_box, f_group)

            functions = VGroup(h_block, g_block, f_block)
            functions.arrange(DOWN, buff=0.4)
            functions.next_to(title, DOWN, buff=0.6)

            # Arrows showing flow
            func_arrows = VGroup(
                Arrow(h_block.get_bottom(), g_block.get_top(),
                      color=SLATE, stroke_width=2, buff=0.1),
                Arrow(g_block.get_bottom(), f_block.get_top(),
                      color=SLATE, stroke_width=2, buff=0.1),
            )

            self.play(
                FadeIn(h_block, shift=RIGHT * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.play(
                FadeIn(g_block, shift=RIGHT * 0.2),
                Create(func_arrows[0]),
                run_time=NORMAL_ANIM,
            )
            self.play(
                FadeIn(f_block, shift=RIGHT * 0.2),
                Create(func_arrows[1]),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── KF mapping ───────────────────────────────────────────────
        with self.voiceover(
            text="The Kalman filter connection: the representation function "
                 "h is the encoder — mapping observations to hidden state. "
                 "The dynamics function g is the state transition — our "
                 "learned F. And the prediction function f produces the "
                 "output from the latent state — analogous to the "
                 "observation model H."
        ) as tracker:
            # Add KF annotations next to each block
            h_kf = Text("= Encoder (obs -> state)", color=SLATE,
                        font_size=16)
            h_kf.next_to(h_block, RIGHT, buff=0.3)

            g_kf = Text("= Learned F (transition)", color=SLATE,
                        font_size=16)
            g_kf.next_to(g_block, RIGHT, buff=0.3)

            f_kf = Text("= H (readout)", color=SLATE,
                        font_size=16)
            f_kf.next_to(f_block, RIGHT, buff=0.3)

            self.play(
                FadeIn(h_kf, shift=LEFT * 0.2),
                FadeIn(g_kf, shift=LEFT * 0.2),
                FadeIn(f_kf, shift=LEFT * 0.2),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── MCTS in latent space ──────────────────────────────────────
        with self.voiceover(
            text="The key innovation: Monte Carlo Tree Search in latent "
                 "space. AlphaZero used MCTS with the real game simulator. "
                 "MuZero replaces the simulator with the learned dynamics "
                 "function g. At each node, it applies g to simulate "
                 "possible futures, then uses f to evaluate them. Planning "
                 "without knowing the rules."
        ) as tracker:
            self.play(
                FadeOut(functions), FadeOut(func_arrows),
                FadeOut(h_kf), FadeOut(g_kf), FadeOut(f_kf),
                run_time=FAST_ANIM,
            )

            # Simple tree visualization
            root = Dot(ORIGIN + UP * 1.5, radius=0.12, color=COLOR_HIGHLIGHT)
            root_label = Text("s_0", color=COLOR_HIGHLIGHT, font_size=14)
            root_label.next_to(root, UP, buff=0.1)

            # Level 1 nodes
            l1_positions = [LEFT * 2.5, LEFT * 0.8, RIGHT * 0.8, RIGHT * 2.5]
            l1_nodes = VGroup()
            l1_arrows = VGroup()
            for i, pos in enumerate(l1_positions):
                node = Dot(pos + UP * 0.0, radius=0.1, color=COLOR_PREDICTION)
                l1_nodes.add(node)
                l1_arrows.add(
                    Arrow(root.get_center(), node.get_center(),
                          color=SLATE, stroke_width=1.5, buff=0.15)
                )

            # Level 2 nodes (selected branches)
            l2_positions = [LEFT * 3.0 + DOWN * 1.2, LEFT * 2.0 + DOWN * 1.2,
                            RIGHT * 0.3 + DOWN * 1.2, RIGHT * 1.3 + DOWN * 1.2]
            l2_nodes = VGroup()
            l2_arrows = VGroup()
            l2_parents = [0, 0, 2, 2]
            for i, pos in enumerate(l2_positions):
                node = Dot(pos, radius=0.08, color=COLOR_FILTER_TF)
                l2_nodes.add(node)
                l2_arrows.add(
                    Arrow(l1_nodes[l2_parents[i]].get_center(),
                          node.get_center(),
                          color=SLATE, stroke_width=1, buff=0.12)
                )

            tree = VGroup(root, root_label, l1_nodes, l1_arrows,
                          l2_nodes, l2_arrows)
            tree.move_to(ORIGIN + UP * 0.3)

            mcts_label = Text(
                "MCTS in Latent Space",
                color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
            )
            mcts_label.next_to(tree, UP, buff=0.3)

            explain = VGroup(
                Text("Each branch: g(state, action) -> next state",
                     color=COLOR_PREDICTION, font_size=SMALL_FONT_SIZE),
                Text("Each leaf: f(state) -> policy, value",
                     color=COLOR_POSTERIOR, font_size=SMALL_FONT_SIZE),
                Text("No game rules needed!",
                     color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE),
            ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            explain.next_to(tree, DOWN, buff=0.3)

            self.play(
                FadeIn(mcts_label),
                FadeIn(root), FadeIn(root_label),
                run_time=FAST_ANIM,
            )
            self.play(
                FadeIn(l1_nodes), Create(l1_arrows),
                run_time=NORMAL_ANIM,
            )
            self.play(
                FadeIn(l2_nodes), Create(l2_arrows),
                run_time=NORMAL_ANIM,
            )
            self.play(FadeIn(explain), run_time=FAST_ANIM)

        self.wait(PAUSE_MEDIUM)

        # ── Results: matches AlphaZero ────────────────────────────────
        with self.voiceover(
            text="The results are remarkable. On Go, chess, and shogi, "
                 "MuZero matches AlphaZero — which had access to the "
                 "perfect game rules. On Atari, MuZero achieves a new "
                 "state of the art. It does this all with a single "
                 "algorithm, learning the rules from scratch."
        ) as tracker:
            self.play(
                FadeOut(tree), FadeOut(mcts_label), FadeOut(explain),
                run_time=FAST_ANIM,
            )

            table = ComparisonTable(
                headers=["Domain", "AlphaZero", "MuZero", "Rules?"],
                rows=[
                    ["Go",        "5185 Elo", "5170 Elo", "No"],
                    ["Chess",     "4737 Elo", "4695 Elo", "No"],
                    ["Shogi",     "4542 Elo", "4565 Elo", "No"],
                    ["Atari (57)", "N/A",      "731% mean", "No"],
                ],
                row_colors=[
                    COLOR_HIGHLIGHT, COLOR_HIGHLIGHT,
                    COLOR_HIGHLIGHT, COLOR_FILTER_TF,
                ],
                title="MuZero vs AlphaZero (Schrittwieser et al. 2020)",
                width=10.0,
                font_size=18,
            )
            table.next_to(title, DOWN, buff=0.6)

            self.play(FadeIn(table.bg), run_time=FAST_ANIM)
            for anim in table.animate_rows():
                self.play(anim, run_time=0.4)

        self.wait(PAUSE_MEDIUM)

        # ── The deep insight ──────────────────────────────────────────
        with self.voiceover(
            text="The deep insight: MuZero doesn't try to reconstruct "
                 "observations. It only predicts what matters — reward, "
                 "value, and policy. The latent state is optimized for "
                 "decision-making, not observation reconstruction. This "
                 "is a fundamentally different objective than the Deep "
                 "Kalman Filter's ELBO."
        ) as tracker:
            insight = VGroup(
                Text("RSSM/DKF: learn to reconstruct observations",
                     color=COLOR_SSM, font_size=SMALL_FONT_SIZE),
                Text("MuZero: learn to predict decisions",
                     color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE),
            ).arrange(DOWN, buff=0.2)
            insight.to_edge(DOWN, buff=0.4)

            self.play(FadeIn(insight, shift=UP * 0.2), run_time=NORMAL_ANIM)

        self.wait(PAUSE_MEDIUM)

        # ── Grand connection ──────────────────────────────────────────
        with self.voiceover(
            text="From the Kalman filter to MuZero: the same core idea "
                 "runs through all of them. Maintain a belief about the "
                 "hidden state. Predict what happens next. Update when "
                 "you observe something. The only difference is what you "
                 "learn and what you assume."
        ) as tracker:
            self.play(FadeOut(insight), run_time=FAST_ANIM)

            grand = Text(
                "Predict, update, decide — from KF to MuZero",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            )
            grand.to_edge(DOWN, buff=0.4)
            self.play(FadeIn(grand, scale=0.9), run_time=NORMAL_ANIM)

        note = make_observation_note(
            "Schrittwieser et al. (2020, Nature): MuZero.\n"
            "Matches AlphaZero without knowing game rules."
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
