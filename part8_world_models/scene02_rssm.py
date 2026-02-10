"""Part 8, Scene 2: RSSM = Predict/Update.

Data: RSSMSim teaching model

The Recurrent State-Space Model (RSSM) from PlaNet/Dreamer decomposes
the latent state into deterministic (h_t, GRU) and stochastic (s_t)
components. The prior is predict, the posterior is update — exactly
the Kalman filter loop with learned dynamics.

Papers:
- Hafner et al. (2019, ICML) PlaNet: Learning Latent Dynamics
- Hafner et al. (2020, ICLR) Dream to Control: DreamerV1
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.rssm_diagram import RSSMDiagram
from kalman_manim.mobjects.comparison_table import ComparisonTable
from kalman_manim.mobjects.observation_note import make_observation_note
from models.rssm_sim import RSSMSim
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneRSSM(VoiceoverScene, MovingCameraScene):
    """RSSM architecture annotated with KF predict/update labels.

    Visual: RSSMDiagram + KF correspondence table + live prior/posterior.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="The Recurrent State-Space Model, or RSSM, is the "
                 "backbone of the Dreamer agent. It splits the latent "
                 "state into two parts: a deterministic path and a "
                 "stochastic path — and together they form a predict "
                 "and update loop."
        ) as tracker:
            title = Text(
                "RSSM = Predict / Update",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── RSSM Architecture Diagram ─────────────────────────────────
        with self.voiceover(
            text="Here's the architecture. The GRU computes a deterministic "
                 "hidden state h_t from the previous state and action. "
                 "The prior network predicts the stochastic state s_t from "
                 "h_t alone — this is the prediction step. The posterior "
                 "network incorporates the observation o_t to refine s_t — "
                 "this is the update step."
        ) as tracker:
            diagram = RSSMDiagram(width=10.0)
            diagram.next_to(title, DOWN, buff=0.6)

            self.play(
                FadeIn(diagram.blocks, shift=RIGHT * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.play(
                Create(diagram.arrows),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── KF Annotations ────────────────────────────────────────────
        with self.voiceover(
            text="Let's annotate this with Kalman filter labels. The prior "
                 "is the prediction: p of s_t given h_t. The posterior is "
                 "the update: q of s_t given h_t and the observation. The "
                 "KL divergence between them measures how much information "
                 "the observation provided — just like the innovation in "
                 "a Kalman filter."
        ) as tracker:
            # Add KF labels next to diagram blocks
            predict_label = Text(
                "= KF Predict", color=COLOR_PREDICTION,
                font_size=SMALL_FONT_SIZE,
            )
            update_label = Text(
                "= KF Update", color=COLOR_POSTERIOR,
                font_size=SMALL_FONT_SIZE,
            )

            # Position near the prior and posterior blocks in the diagram
            predict_label.next_to(diagram.blocks[1], RIGHT, buff=0.3)
            update_label.next_to(diagram.blocks[2], RIGHT, buff=0.3)

            self.play(
                FadeIn(predict_label, shift=LEFT * 0.2),
                FadeIn(update_label, shift=LEFT * 0.2),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── RSSM-KF Correspondence Table ──────────────────────────────
        with self.voiceover(
            text="The correspondence is precise. The GRU is the learned "
                 "state transition — replacing F. The prior network is the "
                 "prediction step. The posterior network is the measurement "
                 "update. And the KL loss trains both to stay consistent — "
                 "like the innovation covariance in a well-tuned filter."
        ) as tracker:
            self.play(
                FadeOut(diagram), FadeOut(predict_label), FadeOut(update_label),
                run_time=FAST_ANIM,
            )

            table = ComparisonTable(
                headers=["RSSM", "Kalman Filter", "Role"],
                rows=[
                    ["GRU (h_t)",       "F (transition)",   "State dynamics"],
                    ["Prior p(s|h)",    "Predict step",     "Forecast state"],
                    ["Posterior q(s|h,o)", "Update step",   "Incorporate obs"],
                    ["KL(q || p)",      "Innovation",       "Info from obs"],
                    ["Reconstruction",  "Likelihood",       "Observation fit"],
                ],
                row_colors=[
                    COLOR_FILTER_TF, COLOR_PREDICTION, COLOR_POSTERIOR,
                    COLOR_HIGHLIGHT, COLOR_MEASUREMENT,
                ],
                title="RSSM-KF Correspondence",
                width=10.0,
                font_size=18,
            )
            table.next_to(title, DOWN, buff=0.6)

            self.play(FadeIn(table.bg), run_time=FAST_ANIM)
            for anim in table.animate_rows():
                self.play(anim, run_time=0.4)

        self.wait(PAUSE_MEDIUM)

        # ── Live simulation: prior vs posterior ───────────────────────
        with self.voiceover(
            text="Let's see this in action with our teaching RSSM. At each "
                 "step, the prior — in red — predicts the stochastic state. "
                 "Then the observation arrives and the posterior — in gold — "
                 "refines it. The posterior is always tighter, just like a "
                 "Kalman update."
        ) as tracker:
            self.play(FadeOut(table), run_time=FAST_ANIM)

            # Run RSSM simulation
            rssm = RSSMSim(h_dim=4, s_dim=2, seed=42)
            n_steps = 12
            observations = [
                0.5 * np.array([np.sin(0.3 * t), np.cos(0.3 * t)])
                + 0.1 * np.random.default_rng(t).normal(size=2)
                for t in range(n_steps)
            ]
            results = rssm.run(observations)

            # Plot prior vs posterior means (s_dim=2, show first component)
            prior_means = [m[0] for m in results["s_prior_means"]]
            post_means = [m[0] for m in results["s_post_means"]]

            # Create axes
            axes = Axes(
                x_range=[0, n_steps, 2],
                y_range=[-1.5, 1.5, 0.5],
                x_length=8, y_length=3.5,
                axis_config={"color": SLATE, "stroke_width": 1.5},
                tips=False,
            )
            axes.next_to(title, DOWN, buff=0.8)

            x_label = Text("Time step", color=SLATE, font_size=16)
            x_label.next_to(axes, DOWN, buff=0.2)
            y_label = Text("s_t[0]", color=SLATE, font_size=16)
            y_label.next_to(axes, LEFT, buff=0.2)

            self.play(FadeIn(axes), FadeIn(x_label), FadeIn(y_label),
                      run_time=FAST_ANIM)

            # Animate dots step by step
            prior_dots = VGroup()
            post_dots = VGroup()
            for t in range(n_steps):
                pd = Dot(
                    axes.c2p(t, prior_means[t]),
                    radius=0.06, color=COLOR_PREDICTION,
                )
                qd = Dot(
                    axes.c2p(t, post_means[t]),
                    radius=0.06, color=COLOR_POSTERIOR,
                )
                prior_dots.add(pd)
                post_dots.add(qd)
                self.play(FadeIn(pd), FadeIn(qd), run_time=0.15)

            # Connect with lines
            prior_line = VMobject(color=COLOR_PREDICTION, stroke_width=1.5)
            prior_line.set_points_smoothly(
                [axes.c2p(t, prior_means[t]) for t in range(n_steps)]
            )
            post_line = VMobject(color=COLOR_POSTERIOR, stroke_width=1.5)
            post_line.set_points_smoothly(
                [axes.c2p(t, post_means[t]) for t in range(n_steps)]
            )
            self.play(Create(prior_line), Create(post_line), run_time=FAST_ANIM)

            # Legend
            legend = VGroup(
                VGroup(
                    Dot(radius=0.05, color=COLOR_PREDICTION),
                    Text("Prior (predict)", color=COLOR_PREDICTION,
                         font_size=16),
                ).arrange(RIGHT, buff=0.1),
                VGroup(
                    Dot(radius=0.05, color=COLOR_POSTERIOR),
                    Text("Posterior (update)", color=COLOR_POSTERIOR,
                         font_size=16),
                ).arrange(RIGHT, buff=0.1),
            ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            legend.to_edge(RIGHT, buff=0.3).shift(DOWN * 0.5)

            self.play(FadeIn(legend), run_time=FAST_ANIM)

        self.wait(PAUSE_MEDIUM)

        # ── Key insight ───────────────────────────────────────────────
        with self.voiceover(
            text="The key insight: the RSSM isn't just inspired by Kalman "
                 "filtering — it IS Kalman filtering with a learned model. "
                 "The GRU replaces the transition matrix, and variational "
                 "inference replaces the closed-form Kalman gain."
        ) as tracker:
            insight = Text(
                "RSSM = KF with learned dynamics + variational inference",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            )
            insight.to_edge(DOWN, buff=0.4)
            self.play(FadeIn(insight, scale=0.9), run_time=NORMAL_ANIM)

        note = make_observation_note(
            "Hafner et al. (2019, ICML): PlaNet/RSSM architecture.\n"
            "Hafner et al. (2020, ICLR): Dreamer builds on RSSM."
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
