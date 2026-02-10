"""Part 7, Scene 4: Social Prediction — learned trajectory forecasting.

Data: Published ADE/FDE results from ETH/UCY benchmarks

From filtering (estimating present) to prediction (forecasting future).
Social models learn interaction patterns from data.

Papers:
- Alahi et al. (2016) Social LSTM — CVPR
- Gupta et al. (2018) Social GAN — CVPR
- Salzmann et al. (2020) Trajectron++ — ECCV
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.comparison_table import ComparisonTable
from kalman_manim.mobjects.prediction_fan import PredictionFan
from kalman_manim.mobjects.observation_note import make_observation_note
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneSocial(VoiceoverScene, MovingCameraScene):
    """Social prediction: learned trajectory forecasting.

    Visual: Prediction fan + ADE/FDE results table.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="Filtering estimates the present. Prediction forecasts "
                 "the future. Social prediction models learn how pedestrians "
                 "interact — avoiding collisions, forming groups, following "
                 "social conventions."
        ) as tracker:
            title = Text(
                "Social Trajectory Prediction",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Prediction fan visualization ───────────────────────────────
        with self.voiceover(
            text="Given 8 observed steps, the model predicts 12 future "
                 "steps. But the future is uncertain — so we get a fan of "
                 "possible trajectories, not a single line."
        ) as tracker:
            # Generate a simple fan
            rng = np.random.default_rng(42)
            origin = np.array([0.0, 0.0])
            n_samples = 15
            trajs = []
            for _ in range(n_samples):
                pts = []
                pos = origin.copy()
                vel = np.array([0.3, 0.0]) + rng.normal(0, 0.05, 2)
                for t in range(12):
                    vel += rng.normal(0, 0.03, 2)
                    pos = pos + vel
                    pts.append(pos.copy())
                trajs.append(np.array(pts))

            fan = PredictionFan(
                origin=origin, trajectories=trajs,
                color=COLOR_SOCIAL, opacity=0.4,
            )
            fan.shift(LEFT * 2 + DOWN * 0.5)

            # Observed path (straight)
            obs_pts = [np.array([-4 + i * 0.5, 0, 0]) + fan.get_center()
                       for i in range(8)]
            obs_line = VMobject()
            obs_line.set_points_smoothly(obs_pts)
            obs_line.set_color(COLOR_MEASUREMENT).set_stroke(width=2.5)

            obs_label = Text("observed (8 steps)", color=COLOR_MEASUREMENT,
                             font_size=14)
            obs_label.next_to(obs_line, UP, buff=0.15)
            pred_label = Text("predicted (12 steps)", color=COLOR_SOCIAL,
                              font_size=14)
            pred_label.next_to(fan, DOWN, buff=0.2)

            self.play(Create(obs_line), FadeIn(obs_label), run_time=NORMAL_ANIM)
            for anim in fan.animate_fan():
                self.play(anim, run_time=0.3)
            self.play(FadeIn(pred_label), run_time=FAST_ANIM)

        self.wait(PAUSE_MEDIUM)

        # ── ADE/FDE results ────────────────────────────────────────────
        with self.voiceover(
            text="The ETH-UCY benchmark shows dramatic progress. Linear "
                 "prediction gives 1.3 meters average displacement error. "
                 "Social LSTM cut it to 0.72. Social GAN to 0.58. "
                 "Trajectron++ to 0.43. And AgentFormer — a transformer "
                 "model — reaches 0.23 meters."
        ) as tracker:
            self.play(
                FadeOut(fan), FadeOut(obs_line),
                FadeOut(obs_label), FadeOut(pred_label),
                run_time=FAST_ANIM,
            )

            table = ComparisonTable(
                headers=["Method", "Year", "ADE (m)", "FDE (m)"],
                rows=[
                    ["Linear",       "--",   "1.33", "2.94"],
                    ["S-LSTM",       "2016", "0.72", "1.48"],
                    ["S-GAN",        "2018", "0.58", "1.18"],
                    ["Trajectron++", "2020", "0.43", "0.86"],
                    ["AgentFormer",  "2021", "0.23", "0.39"],
                ],
                row_colors=[SLATE, COLOR_SOCIAL, COLOR_SOCIAL,
                            COLOR_SOCIAL, COLOR_FILTER_TF],
                title="ETH/UCY Trajectory Prediction (best-of-20)",
                highlight_best=[2, 3],
                width=9.0,
            )
            table.next_to(title, DOWN, buff=0.7)

            self.play(FadeIn(table.bg), run_time=FAST_ANIM)
            for anim in table.animate_rows():
                self.play(anim, run_time=0.4)

        self.wait(PAUSE_MEDIUM)

        # ── Key insight ────────────────────────────────────────────────
        with self.voiceover(
            text="The key difference from classical filtering: these models "
                 "learn social forces — how pedestrians repel each other, "
                 "form groups, and navigate shared spaces — entirely from data."
        ) as tracker:
            insight = Text(
                "Social forces learned from data, not hand-crafted",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            )
            insight.to_edge(DOWN, buff=0.4)
            self.play(FadeIn(insight, scale=0.9), run_time=NORMAL_ANIM)

        note = make_observation_note(
            "ADE = Avg Displacement Error, FDE = Final Displacement Error\n"
            "Best-of-20 samples, meters. Standard ETH/UCY splits."
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
