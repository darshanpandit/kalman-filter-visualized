"""Part 6, Scene 4: KalmanNet — Hybrid Approach.

Data: Precomputed .npz from Revach et al. (2022) and Mehrfard et al. (2024)

KalmanNet keeps the predict/update structure but replaces the Kalman gain
computation with a learned GRU. Shows the architecture, then the radar
failure case where KalmanNet underperforms classical IMM.

Papers:
- Revach et al. (2022, IEEE TSP) — KalmanNet architecture
- Mehrfard et al. (2024) — RadarScenes evaluation showing failure
- Shen et al. (2025) — KalmanFormer extension
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.architecture import KalmanNetDiagram
from kalman_manim.mobjects.comparison_table import ComparisonTable
from kalman_manim.mobjects.observation_note import make_observation_note
from models.kalmannet_stub import load_kalmannet_results
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneKalmanNet(VoiceoverScene, MovingCameraScene):
    """KalmanNet: hybrid model-based + learned Kalman gain.

    Visual: Architecture diagram + RadarScenes failure table.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="KalmanNet takes a different approach: keep the classical "
                 "predict-update structure, but replace the Kalman gain "
                 "computation with a learned recurrent network."
        ) as tracker:
            title = Text(
                "KalmanNet: Hybrid Approach",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Architecture diagram ───────────────────────────────────────
        with self.voiceover(
            text="The architecture: prediction uses the known dynamics model "
                 "F times x. Then the innovation — the difference between "
                 "measurement and prediction — feeds into a GRU. The GRU "
                 "outputs the Kalman gain K, which drives the update step. "
                 "The GRU learns to adapt the gain based on context."
        ) as tracker:
            diagram = KalmanNetDiagram(width=10.0)
            diagram.next_to(title, DOWN, buff=0.8)

            self.play(
                FadeIn(diagram.blocks, shift=RIGHT * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.play(
                Create(diagram.arrows),
                FadeIn(diagram.note),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── Advantage: model mismatch ──────────────────────────────────
        with self.voiceover(
            text="The advantage: when the true noise statistics don't match "
                 "your assumptions — model mismatch — the GRU can learn to "
                 "compensate. On synthetic benchmarks, KalmanNet often "
                 "outperforms the classical KF."
        ) as tracker:
            advantage = Text(
                "Handles model mismatch by learning the gain",
                color=COLOR_FILTER_KALMANNET, font_size=BODY_FONT_SIZE,
            )
            advantage.to_edge(DOWN, buff=0.5)
            self.play(FadeIn(advantage, shift=UP * 0.2), run_time=NORMAL_ANIM)

        self.wait(PAUSE_SHORT)

        # ── Radar failure ──────────────────────────────────────────────
        with self.voiceover(
            text="But here's the catch. Mehrfard and colleagues tested "
                 "KalmanNet on real radar data — the RadarScenes dataset — "
                 "and it failed. The classical IMM filter beat KalmanNet "
                 "on both position and velocity RMSE."
        ) as tracker:
            self.play(
                FadeOut(diagram), FadeOut(advantage),
                run_time=FAST_ANIM,
            )

            results = load_kalmannet_results()
            table = ComparisonTable(
                headers=["Method", "Pos RMSE", "Vel RMSE"],
                rows=[
                    ["KalmanNet", "1.23", "2.98"],
                    ["IMM",       "1.08", "1.28"],
                ],
                row_colors=[COLOR_FILTER_KALMANNET, COLOR_FILTER_IMM],
                title="RadarScenes (Mehrfard et al. 2024)",
                highlight_best=[1, 2],
                width=7.0,
            )
            table.next_to(title, DOWN, buff=0.8)

            self.play(FadeIn(table.bg), run_time=FAST_ANIM)
            for anim in table.animate_rows():
                self.play(anim, run_time=0.5)

        self.wait(PAUSE_MEDIUM)

        # ── Safety conclusion ──────────────────────────────────────────
        with self.voiceover(
            text="The paper's conclusion is sobering: quote, 'the current "
                 "lack of reliability makes KalmanNet unsuited for "
                 "safety-critical applications.' The learned gain doesn't "
                 "generalize well to out-of-distribution real-world data."
        ) as tracker:
            conclusion = Text(
                '"Unsuited for safety-critical applications"',
                color=SWISS_RED, font_size=BODY_FONT_SIZE,
            )
            conclusion.to_edge(DOWN, buff=0.4)

            source = Text(
                "— Mehrfard et al. (2024)",
                color=SLATE, font_size=SMALL_FONT_SIZE,
            )
            source.next_to(conclusion, DOWN, buff=0.15)

            self.play(
                FadeIn(conclusion, scale=0.9),
                FadeIn(source),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_LONG)

        note = make_observation_note(
            "Revach et al. (2022, IEEE TSP): KalmanNet architecture.\n"
            "Shen et al. (2025): KalmanFormer extends with attention."
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_MEDIUM)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
