"""Part 5, Scene 5: Quantitative Recommendations.

Data: precomputed sweep_results.npz

Decision guide backed by benchmark data:
- At what nonlinearity does KF degrade?
- Accuracy vs computation trade-off curve.
- Updated decision flowchart.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.observation_note import make_observation_note
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

_SWEEP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "benchmark_results", "sweep_results.npz"
)
_TIMING_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "benchmark_results", "timing_results.npz"
)

COLORS = {
    "KF": COLOR_FILTER_KF, "EKF": COLOR_FILTER_EKF,
    "UKF": COLOR_FILTER_UKF, "PF": COLOR_FILTER_PF,
}


class SceneRecommendations(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # Load data
        sweep = np.load(_SWEEP_PATH, allow_pickle=True)
        timing = np.load(_TIMING_PATH, allow_pickle=True)

        turn_rates = sweep["turn_rates"]
        filter_names = [str(n) for n in sweep["filter_names"]]
        mean_rmse = sweep["mean_rmse"]  # (R, F)
        timing_mean = timing["mean_s"]

        # ── Title ─────────────────────────────────────────────────────────
        with self.voiceover(text="Let's turn our benchmark data into practical recommendations. When should you use each filter?") as tracker:
            title = Text("Which Filter Should You Use?", color=COLOR_TEXT,
                          font_size=TITLE_FONT_SIZE)
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Quantitative thresholds ───────────────────────────────────────
        # Find where KF RMSE exceeds UKF by >20%
        kf_idx = filter_names.index("KF")
        ukf_idx = filter_names.index("UKF")
        ratio = mean_rmse[:, kf_idx] / np.maximum(mean_rmse[:, ukf_idx], 1e-6)
        threshold_mask = ratio > 1.2
        if threshold_mask.any():
            crossover_rate = turn_rates[np.argmax(threshold_mask)]
        else:
            crossover_rate = turn_rates[-1]

        with self.voiceover(text=f"From our sweep data: below a turn rate of about {crossover_rate:.2f} radians per second, the KF stays within twenty percent of the UKF. Above that, the gap widens rapidly. If your system has mild nonlinearity, the simple KF is a perfectly good choice.") as tracker:
            threshold_text = VGroup(
                Text("KF is fine when:", color=COLOR_TEXT,
                     font_size=BODY_FONT_SIZE),
                Text(
                    f"\u03c9 < {crossover_rate:.2f} rad/s",
                    color=COLOR_FILTER_KF, font_size=EQUATION_FONT_SIZE,
                ),
                Text(f"(within 20% of UKF)", color=SLATE,
                     font_size=SMALL_FONT_SIZE),
            ).arrange(DOWN, buff=0.2)
            threshold_text.next_to(title, DOWN, buff=0.8)
            self.play(FadeIn(threshold_text, shift=UP * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        self.play(FadeOut(threshold_text), run_time=FAST_ANIM)

        # ── Decision flowchart ────────────────────────────────────────────
        with self.voiceover(text="Here's the updated decision flowchart, backed by data. Start with the KF. If your model is nonlinear, switch to EKF for mild cases or UKF for stronger nonlinearity. If your noise is non-Gaussian or multimodal, use the Particle Filter. Each step up costs more computation.") as tracker:
            rules = VGroup(
                VGroup(
                    Text("Linear + Gaussian", color=COLOR_FILTER_KF,
                          font_size=BODY_FONT_SIZE),
                    Text("\u2192", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
                    Text("KF", color=COLOR_FILTER_KF, font_size=BODY_FONT_SIZE),
                    Text(f"({timing_mean[kf_idx]*1000:.1f}ms)", color=SLATE,
                         font_size=SMALL_FONT_SIZE),
                ).arrange(RIGHT, buff=0.25),
                VGroup(
                    Text("Mildly nonlinear", color=COLOR_FILTER_EKF,
                          font_size=BODY_FONT_SIZE),
                    Text("\u2192", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
                    Text("EKF", color=COLOR_FILTER_EKF, font_size=BODY_FONT_SIZE),
                    Text(f"({timing_mean[filter_names.index('EKF')]*1000:.1f}ms)",
                         color=SLATE, font_size=SMALL_FONT_SIZE),
                ).arrange(RIGHT, buff=0.25),
                VGroup(
                    Text("Strongly nonlinear", color=COLOR_FILTER_UKF,
                          font_size=BODY_FONT_SIZE),
                    Text("\u2192", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
                    Text("UKF", color=COLOR_FILTER_UKF, font_size=BODY_FONT_SIZE),
                    Text(f"({timing_mean[ukf_idx]*1000:.1f}ms)", color=SLATE,
                         font_size=SMALL_FONT_SIZE),
                ).arrange(RIGHT, buff=0.25),
                VGroup(
                    Text("Non-Gaussian / multimodal", color=COLOR_FILTER_PF,
                          font_size=BODY_FONT_SIZE),
                    Text("\u2192", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
                    Text("PF", color=COLOR_FILTER_PF, font_size=BODY_FONT_SIZE),
                    Text(f"({timing_mean[filter_names.index('PF')]*1000:.1f}ms)",
                         color=SLATE, font_size=SMALL_FONT_SIZE),
                ).arrange(RIGHT, buff=0.25),
            ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
            rules.next_to(title, DOWN, buff=LARGE_BUFF)

            for rule in rules:
                self.play(FadeIn(rule, shift=RIGHT * 0.3), run_time=NORMAL_ANIM)
                self.wait(PAUSE_SHORT)

            self.wait(PAUSE_MEDIUM)

        # ── Closing ───────────────────────────────────────────────────────
        with self.voiceover(text="The simplest filter that meets your accuracy needs is the right one. Don't over-engineer. Now you have the data to make that decision quantitatively. Thanks for watching the complete Kalman Filter series!") as tracker:
            note = make_observation_note(
                "Timing measured on 60-step trajectories.\n"
                "PF: 300 particles.",
            )
            self.play(FadeIn(note), run_time=FAST_ANIM)
            self.wait(PAUSE_MEDIUM)

            closing = Text(
                "The simplest filter that works is the right one.",
                color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
            )
            closing.to_edge(DOWN, buff=0.5)
            self.play(FadeIn(closing, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
