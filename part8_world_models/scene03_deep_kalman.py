"""Part 8, Scene 3: Deep Kalman Filters.

Data: Conceptual diagrams (GraphicalModel plate notation)

Deep Kalman Filters and Deep Variational Bayes Filters learn both the
transition and emission models as neural networks, trained end-to-end
via the ELBO = reconstruction + KL. The graphical model is a state-space
model with learned parameters.

Papers:
- Krishnan et al. (2017, AAAI) — Deep Kalman Filters
- Karl et al. (2017, ICLR) — DVBF: Deep Variational Bayes Filters
- Kingma & Welling (2014, ICLR) — VAE framework
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.rssm_diagram import GraphicalModel
from kalman_manim.mobjects.comparison_table import ComparisonTable
from kalman_manim.mobjects.observation_note import make_observation_note
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneDeepKalman(VoiceoverScene, MovingCameraScene):
    """Deep Kalman Filters: ELBO = reconstruction + KL.

    Visual: GraphicalModel plate notation + ELBO decomposition + comparison.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="The Deep Kalman Filter takes the state-space model and "
                 "makes every component learnable. The transition, the "
                 "emission, and the inference — all are neural networks "
                 "trained end-to-end."
        ) as tracker:
            title = Text(
                "Deep Kalman Filters",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Graphical Model (plate notation) ──────────────────────────
        with self.voiceover(
            text="Here's the graphical model. Latent states z_t — shown in "
                 "red — form a Markov chain. Each z_t generates an observation "
                 "x_t — in blue. The plate notation indicates this repeats "
                 "over T time steps. This is exactly a state-space model."
        ) as tracker:
            gm = GraphicalModel(n_steps=4)
            gm.scale(0.85)
            gm.next_to(title, DOWN, buff=0.8)

            self.play(FadeIn(gm, shift=UP * 0.2), run_time=NORMAL_ANIM)

            # Label the components
            z_label = Text(
                "Latent states (hidden)", color=COLOR_PREDICTION,
                font_size=SMALL_FONT_SIZE,
            )
            z_label.next_to(gm, LEFT, buff=0.3).shift(UP * 0.5)

            x_label = Text(
                "Observations (visible)", color=COLOR_MEASUREMENT,
                font_size=SMALL_FONT_SIZE,
            )
            x_label.next_to(gm, LEFT, buff=0.3).shift(DOWN * 0.3)

            self.play(FadeIn(z_label), FadeIn(x_label), run_time=FAST_ANIM)

        self.wait(PAUSE_MEDIUM)

        # ── The three learned components ──────────────────────────────
        with self.voiceover(
            text="Three neural networks replace the hand-crafted matrices. "
                 "The transition network models p of z_t given z_{t-1} — "
                 "replacing F. The emission network models p of x_t given "
                 "z_t — replacing H. And the inference network approximates "
                 "the posterior q of z_t given x — this is the encoder."
        ) as tracker:
            self.play(
                FadeOut(gm), FadeOut(z_label), FadeOut(x_label),
                run_time=FAST_ANIM,
            )

            # Three component boxes
            transition_box = RoundedRectangle(
                corner_radius=0.1, width=3.2, height=0.8,
                color=COLOR_PREDICTION, fill_opacity=0.2, stroke_width=2,
            )
            trans_label = Text(
                "Transition: p(z_t | z_{t-1})",
                color=COLOR_PREDICTION, font_size=18,
            )
            trans_label.move_to(transition_box)
            trans_note = Text("replaces F", color=SLATE, font_size=14)
            trans_note.next_to(transition_box, RIGHT, buff=0.2)
            trans_block = VGroup(transition_box, trans_label, trans_note)

            emission_box = RoundedRectangle(
                corner_radius=0.1, width=3.2, height=0.8,
                color=COLOR_MEASUREMENT, fill_opacity=0.2, stroke_width=2,
            )
            emis_label = Text(
                "Emission: p(x_t | z_t)",
                color=COLOR_MEASUREMENT, font_size=18,
            )
            emis_label.move_to(emission_box)
            emis_note = Text("replaces H", color=SLATE, font_size=14)
            emis_note.next_to(emission_box, RIGHT, buff=0.2)
            emis_block = VGroup(emission_box, emis_label, emis_note)

            inference_box = RoundedRectangle(
                corner_radius=0.1, width=3.2, height=0.8,
                color=COLOR_POSTERIOR, fill_opacity=0.2, stroke_width=2,
            )
            inf_label = Text(
                "Inference: q(z_t | x_{1:T})",
                color=COLOR_POSTERIOR, font_size=18,
            )
            inf_label.move_to(inference_box)
            inf_note = Text("replaces K gain", color=SLATE, font_size=14)
            inf_note.next_to(inference_box, RIGHT, buff=0.2)
            inf_block = VGroup(inference_box, inf_label, inf_note)

            components = VGroup(trans_block, emis_block, inf_block)
            components.arrange(DOWN, buff=0.4)
            components.next_to(title, DOWN, buff=0.7)

            for comp in components:
                self.play(FadeIn(comp, shift=RIGHT * 0.2), run_time=0.5)

        self.wait(PAUSE_MEDIUM)

        # ── ELBO decomposition ────────────────────────────────────────
        with self.voiceover(
            text="Training uses the evidence lower bound — the ELBO. It "
                 "decomposes into two terms. Reconstruction: how well can "
                 "the model explain the observations from the latent states? "
                 "And KL: how close is the approximate posterior to the prior? "
                 "Maximize reconstruction, minimize KL divergence."
        ) as tracker:
            self.play(FadeOut(components), run_time=FAST_ANIM)

            elbo_title = Text(
                "ELBO = Reconstruction - KL",
                color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
            )
            elbo_title.next_to(title, DOWN, buff=0.8)

            recon_text = Text(
                "Reconstruction: E_q[ log p(x_t | z_t) ]",
                color=COLOR_MEASUREMENT, font_size=BODY_FONT_SIZE,
            )
            recon_explain = Text(
                "How well do latent states explain observations?",
                color=SLATE, font_size=SMALL_FONT_SIZE,
            )

            kl_text = Text(
                "KL: KL( q(z_t | x) || p(z_t | z_{t-1}) )",
                color=COLOR_PREDICTION, font_size=BODY_FONT_SIZE,
            )
            kl_explain = Text(
                "How close is the posterior to the prior?",
                color=SLATE, font_size=SMALL_FONT_SIZE,
            )

            elbo_group = VGroup(
                elbo_title,
                VGroup(recon_text, recon_explain).arrange(DOWN, buff=0.1),
                VGroup(kl_text, kl_explain).arrange(DOWN, buff=0.1),
            ).arrange(DOWN, buff=0.5)
            elbo_group.next_to(title, DOWN, buff=0.6)

            self.play(FadeIn(elbo_title, shift=UP * 0.1), run_time=NORMAL_ANIM)
            self.play(
                FadeIn(recon_text), FadeIn(recon_explain),
                run_time=NORMAL_ANIM,
            )
            self.play(
                FadeIn(kl_text), FadeIn(kl_explain),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── KF connection ─────────────────────────────────────────────
        with self.voiceover(
            text="The connection to Kalman filtering: the reconstruction "
                 "term is the log-likelihood — same as the KF innovation "
                 "likelihood. The KL term regularizes the posterior toward "
                 "the prior — analogous to the prediction step constraining "
                 "the update. The ELBO objective trains the system to be a "
                 "good state estimator."
        ) as tracker:
            kf_connection = VGroup(
                Text("Reconstruction  ~  Innovation likelihood",
                     color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
                Text("KL regularizer  ~  Prediction constraint",
                     color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
                Text("Maximize ELBO  ~  Optimal filtering",
                     color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE),
            ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            kf_connection.to_edge(DOWN, buff=0.4)

            self.play(
                FadeIn(kf_connection, shift=UP * 0.2),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── DKF vs DVBF comparison ────────────────────────────────────
        with self.voiceover(
            text="Two key papers. Krishnan and colleagues at AAAI 2017 "
                 "proposed the Deep Kalman Filter, using an RNN encoder for "
                 "inference. Karl and colleagues at ICLR 2017 proposed the "
                 "Deep Variational Bayes Filter, which enforces the Markov "
                 "structure more strictly. Both achieve the same goal: "
                 "learned state estimation via variational inference."
        ) as tracker:
            self.play(
                *[FadeOut(mob) for mob in self.mobjects if mob is not title],
                run_time=FAST_ANIM,
            )

            table = ComparisonTable(
                headers=["Model", "Inference", "Key Idea"],
                rows=[
                    ["DKF (Krishnan 2017)", "RNN encoder",
                     "Bidirectional smoothing"],
                    ["DVBF (Karl 2017)",    "Structured q",
                     "Enforce Markov property"],
                    ["VRNN (Chung 2015)",   "VAE per step",
                     "Variational RNN"],
                ],
                row_colors=[
                    COLOR_FILTER_TF, COLOR_SSM, COLOR_FILTER_KALMANNET,
                ],
                title="Deep State-Space Models",
                width=10.0,
                font_size=18,
            )
            table.next_to(title, DOWN, buff=0.6)

            self.play(FadeIn(table.bg), run_time=FAST_ANIM)
            for anim in table.animate_rows():
                self.play(anim, run_time=0.4)

        self.wait(PAUSE_MEDIUM)

        # ── Takeaway ──────────────────────────────────────────────────
        with self.voiceover(
            text="The takeaway: the Deep Kalman Filter family shows that "
                 "variational inference can replace the closed-form Kalman "
                 "equations. You lose optimality guarantees, but gain the "
                 "ability to handle arbitrary nonlinear dynamics and "
                 "high-dimensional observations like images."
        ) as tracker:
            takeaway = Text(
                "Trade optimality for generality",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            )
            takeaway.to_edge(DOWN, buff=0.4)
            self.play(FadeIn(takeaway, scale=0.9), run_time=NORMAL_ANIM)

        note = make_observation_note(
            "Krishnan et al. (2017, AAAI): Deep Kalman Filters.\n"
            "Karl et al. (2017, ICLR): DVBF with structured inference."
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
