"""Part 6, Scene 5: SSM-Attention Duality.

Data: Conceptual diagrams

The state-space model (SSM) renaissance: S4, Mamba, and the SSD duality.
Shows that the S4 state equation IS a Kalman-style state-space model,
and that Mamba's selectivity = input-dependent dynamics.

Papers:
- Gu et al. (2022, ICLR) — S4: Structured State Spaces
- Gu & Dao (2024, ICLR) — Mamba: Selective State Spaces
- Dao & Gu (2024, ICML) — SSD: State Space Duality
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.architecture import SSMDiagram
from kalman_manim.mobjects.comparison_table import ComparisonTable
from kalman_manim.mobjects.observation_note import make_observation_note
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneSSMDuality(VoiceoverScene, MovingCameraScene):
    """SSM-Attention Duality: S4 = KF state-space, Mamba adds selectivity.

    Visual: SSM diagram + KF correspondence table + duality equation.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="There's a deeper connection between Kalman filtering and "
                 "modern sequence models. The state-space model renaissance "
                 "— S4, Mamba, and their descendants — is built on the same "
                 "mathematical foundation."
        ) as tracker:
            title = Text(
                "The SSM-Attention Duality",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── S4 equations ───────────────────────────────────────────────
        with self.voiceover(
            text="S4's core equations look familiar: a hidden state h "
                 "evolves as A times h plus B times the input. The output "
                 "is C times h plus D times the input. This IS a "
                 "state-space model — exactly the same structure as our "
                 "Kalman filter."
        ) as tracker:
            s4_eq1 = Text(
                "h'(t) = A h(t) + B x(t)",
                color=COLOR_SSM, font_size=BODY_FONT_SIZE,
            )
            s4_eq2 = Text(
                "y(t) = C h(t) + D x(t)",
                color=COLOR_SSM, font_size=BODY_FONT_SIZE,
            )
            s4_eqs = VGroup(s4_eq1, s4_eq2).arrange(DOWN, buff=0.3)
            s4_eqs.next_to(title, DOWN, buff=0.8)

            s4_label = Text(
                "S4 (Gu et al. 2022)", color=SLATE,
                font_size=SMALL_FONT_SIZE,
            )
            s4_label.next_to(s4_eqs, UP, buff=0.2)

            self.play(
                Write(s4_eq1), FadeIn(s4_label),
                run_time=NORMAL_ANIM,
            )
            self.play(Write(s4_eq2), run_time=NORMAL_ANIM)

        self.wait(PAUSE_SHORT)

        # ── KF correspondence ──────────────────────────────────────────
        with self.voiceover(
            text="The correspondence is direct: A and B are the state "
                 "transition, C is the observation model, and D is a "
                 "feedthrough term. The only difference is that S4 learns "
                 "these matrices from data, while the Kalman filter "
                 "requires them to be specified."
        ) as tracker:
            self.play(FadeOut(s4_eqs), FadeOut(s4_label), run_time=FAST_ANIM)

            table = ComparisonTable(
                headers=["SSM", "Kalman Filter", "Role"],
                rows=[
                    ["A", "F", "State transition"],
                    ["B", "B", "Input matrix"],
                    ["C", "H", "Observation model"],
                    ["D", "0", "Feedthrough"],
                ],
                row_colors=[COLOR_SSM, COLOR_SSM, COLOR_SSM, COLOR_SSM],
                title="SSM ↔ KF Correspondence",
                width=8.0,
            )
            table.next_to(title, DOWN, buff=0.7)

            self.play(FadeIn(table.bg), run_time=FAST_ANIM)
            for anim in table.animate_rows():
                self.play(anim, run_time=0.4)

        self.wait(PAUSE_MEDIUM)

        # ── Mamba: selective SSM ───────────────────────────────────────
        with self.voiceover(
            text="Mamba's key innovation: make A, B, and C depend on the "
                 "input. This selectivity means the model can choose what "
                 "to remember and what to forget — like an adaptive Kalman "
                 "gain that varies with the observation."
        ) as tracker:
            self.play(FadeOut(table), run_time=FAST_ANIM)

            diagram = SSMDiagram(width=9.0)
            diagram.next_to(title, DOWN, buff=0.8)

            self.play(
                FadeIn(diagram.blocks, shift=RIGHT * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.play(
                Create(diagram.arrows),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── SSD duality theorem ────────────────────────────────────────
        with self.voiceover(
            text="The SSD duality theorem from Dao and Gu proves that "
                 "structured state-space models and attention are "
                 "mathematically equivalent under certain conditions. "
                 "The SSM computes the same thing as quadratic attention — "
                 "just more efficiently."
        ) as tracker:
            self.play(FadeOut(diagram), run_time=FAST_ANIM)

            duality = VGroup(
                Text("State Space Duality (SSD):", color=COLOR_SSM,
                     font_size=BODY_FONT_SIZE),
                Text(
                    "SSM(A,B,C) ≡ Attention(Q,K,V)",
                    color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
                ),
                Text(
                    "SSMs = linear attention with structured masks",
                    color=SLATE, font_size=SMALL_FONT_SIZE,
                ),
            ).arrange(DOWN, buff=0.3)
            duality.next_to(title, DOWN, buff=1.0)

            self.play(FadeIn(duality, shift=UP * 0.2), run_time=NORMAL_ANIM)

        self.wait(PAUSE_MEDIUM)

        # ── Grand connection ───────────────────────────────────────────
        with self.voiceover(
            text="So the thread connecting everything: Kalman filtering, "
                 "state-space models, and transformer attention are all "
                 "different views of the same underlying computation — "
                 "optimal state estimation from sequential observations."
        ) as tracker:
            grand = Text(
                "KF ↔ SSM ↔ Attention: same computation, different views",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            )
            grand.to_edge(DOWN, buff=0.4)
            self.play(FadeIn(grand, scale=0.9), run_time=NORMAL_ANIM)

        note = make_observation_note(
            "Dao & Gu (2024, ICML): SSD duality theorem.\n"
            "Cole et al. (2025, NeurIPS): depth separation result."
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
