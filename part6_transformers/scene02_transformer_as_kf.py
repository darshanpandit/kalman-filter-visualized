"""Part 6, Scene 2: Attention IS Filtering — Nadaraya-Watson connection.

Data: Synthetic 1D linear system

Split screen: left shows KF predict/update, right shows attention
heatmap. The Nadaraya-Watson kernel estimator from Goel & Bartlett (2024)
demonstrates that softmax attention over past observations recovers
a form of Kalman smoothing.

Papers:
- Goel & Bartlett (2024, L4DC) Theorem 1
- Akram & Vikalo (2024, TMLR) scaling table
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.attention_heatmap import AttentionHeatmap
from kalman_manim.mobjects.comparison_table import ComparisonTable
from kalman_manim.mobjects.observation_note import make_observation_note
from models.transformer_kf import NWKalmanEstimator
from filters.kalman import KalmanFilter
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneTransformerAsKF(VoiceoverScene, MovingCameraScene):
    """Attention IS filtering: Nadaraya-Watson connection.

    Visual: KF estimates vs NW estimates on 1D system + attention heatmap.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Generate 1D data ───────────────────────────────────────────
        rng = np.random.default_rng(42)
        T = 30
        F, H, Q, R = 1.0, 1.0, 0.1, 1.0
        true_states = np.zeros(T + 1)
        observations = np.zeros(T)
        true_states[0] = 0.0
        for t in range(T):
            true_states[t + 1] = F * true_states[t] + rng.normal(0, np.sqrt(Q))
            observations[t] = H * true_states[t + 1] + rng.normal(0, np.sqrt(R))

        # Run KF
        kf = KalmanFilter(
            F=np.array([[F]]), H=np.array([[H]]),
            Q=np.array([[Q]]), R=np.array([[R]]),
            x0=np.array([0.0]), P0=np.array([[1.0]]),
        )
        kf_results = kf.run([np.array([z]) for z in observations])
        kf_estimates = np.array([x[0] for x in kf_results["x_estimates"]])

        # Run NW estimator
        nw = NWKalmanEstimator(bandwidth=1.5)
        nw_estimates = nw.estimate(observations)
        attn_weights = nw.attention_weights(observations)

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="Here's the key insight from Goel and Bartlett: a single "
                 "attention layer performing Nadaraya-Watson kernel regression "
                 "can approximate Kalman filtering."
        ) as tracker:
            title = Text(
                "Attention IS Filtering",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Left: KF vs NW on 1D system ───────────────────────────────
        with self.voiceover(
            text="On a simple one-dimensional linear system, watch how "
                 "the Nadaraya-Watson estimator — which is just softmax "
                 "attention over past observations — closely tracks the "
                 "Kalman filter output."
        ) as tracker:
            # Create axes
            axes = Axes(
                x_range=[0, T, 5],
                y_range=[
                    float(min(observations.min(), true_states.min()) - 1),
                    float(max(observations.max(), true_states.max()) + 1),
                    1.0,
                ],
                x_length=5.5,
                y_length=3.5,
                axis_config={"color": CREAM, "include_tip": False},
            )
            axes.to_edge(LEFT, buff=0.5).shift(DOWN * 0.5)

            x_lab = Text("time", font_size=14, color=SLATE)
            x_lab.next_to(axes.x_axis, DOWN, buff=0.15)
            y_lab = Text("state", font_size=14, color=SLATE)
            y_lab.next_to(axes.y_axis, LEFT, buff=0.15).rotate(PI / 2)

            # Observations as dots
            obs_dots = VGroup(*[
                Dot(axes.c2p(t + 1, observations[t]),
                    radius=0.03, color=COLOR_MEASUREMENT)
                for t in range(T)
            ])

            # KF line
            kf_points = [axes.c2p(t + 1, kf_estimates[t]) for t in range(T)]
            kf_line = VMobject()
            kf_line.set_points_smoothly(kf_points)
            kf_line.set_color(COLOR_PREDICTION).set_stroke(width=2.5)

            # NW line
            nw_points = [axes.c2p(t + 1, nw_estimates[t]) for t in range(T)]
            nw_line = VMobject()
            nw_line.set_points_smoothly(nw_points)
            nw_line.set_color(COLOR_FILTER_TF).set_stroke(width=2.5)

            # Legend
            legend = VGroup(
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=COLOR_PREDICTION, stroke_width=2.5),
                    Text("KF", color=COLOR_PREDICTION, font_size=14),
                ).arrange(RIGHT, buff=0.1),
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=COLOR_FILTER_TF, stroke_width=2.5),
                    Text("NW Attention", color=COLOR_FILTER_TF, font_size=14),
                ).arrange(RIGHT, buff=0.1),
                VGroup(
                    Dot(radius=0.03, color=COLOR_MEASUREMENT),
                    Text("Observations", color=COLOR_MEASUREMENT, font_size=14),
                ).arrange(RIGHT, buff=0.1),
            ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            legend.next_to(axes, DOWN, buff=0.3)

            self.play(
                FadeIn(axes), FadeIn(x_lab), FadeIn(y_lab),
                FadeIn(obs_dots),
                run_time=NORMAL_ANIM,
            )
            self.play(Create(kf_line), run_time=NORMAL_ANIM)
            self.play(Create(nw_line), run_time=NORMAL_ANIM)
            self.play(FadeIn(legend), run_time=FAST_ANIM)

        self.wait(PAUSE_SHORT)

        # ── Right: Attention heatmap ───────────────────────────────────
        with self.voiceover(
            text="The attention heatmap shows which past observations each "
                 "time step attends to. The causal mask ensures we only look "
                 "backward — just like a real-time filter."
        ) as tracker:
            heatmap = AttentionHeatmap(
                attn_weights[:15, :15],
                cell_size=0.25,
                max_display=15,
            )
            heatmap.to_edge(RIGHT, buff=0.5).shift(DOWN * 0.3)

            hm_title = Text(
                "Attention Weights", color=COLOR_FILTER_TF,
                font_size=SMALL_FONT_SIZE,
            )
            hm_title.next_to(heatmap, UP, buff=0.3)

            self.play(FadeIn(hm_title), run_time=FAST_ANIM)
            for anim in heatmap.animate_rows():
                self.play(anim)

        self.wait(PAUSE_MEDIUM)

        # ── Scaling table ──────────────────────────────────────────────
        with self.voiceover(
            text="Akram and Vikalo showed that with more transformer layers, "
                 "the learned filter converges toward the optimal — matching "
                 "the EKF and particle filter with just 8 layers."
        ) as tracker:
            self.play(
                *[FadeOut(mob) for mob in self.mobjects
                  if mob is not title],
                run_time=FAST_ANIM,
            )

            table = ComparisonTable(
                headers=["TF Layers", "vs EKF", "vs PF"],
                rows=[
                    ["1",  "1.028", "0.990"],
                    ["4",  "0.147", "0.124"],
                    ["8",  "0.053", "0.034"],
                    ["16", "0.053", "0.034"],
                ],
                row_colors=[COLOR_FILTER_TF] * 4,
                title="MSPD (lower = closer to optimal)",
                highlight_best=[1, 2],
                width=7.0,
            )
            table.next_to(title, DOWN, buff=0.8)

            self.play(FadeIn(table.bg), run_time=FAST_ANIM)
            for anim in table.animate_rows():
                self.play(anim, run_time=0.4)

        note = make_observation_note(
            "MSPD = Mean Squared Prediction Difference.\n"
            "Akram & Vikalo (2024, TMLR)"
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
