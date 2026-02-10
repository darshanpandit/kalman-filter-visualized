"""Part 5, Scene 2: RMSE vs Nonlinearity Sweep (flagship).

Data: precomputed sweep_results.npz (25 turn_rates × 50 trials × 4 filters)

Animated line chart showing how each filter's RMSE changes as trajectory
nonlinearity increases. Confidence bands show trial-to-trial variation.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.charts import RMSELineChart
from kalman_manim.mobjects.observation_note import make_observation_note
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

_RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "benchmark_results", "sweep_results.npz"
)

COLORS = {
    "KF": COLOR_FILTER_KF, "EKF": COLOR_FILTER_EKF,
    "UKF": COLOR_FILTER_UKF, "PF": COLOR_FILTER_PF,
}


class SceneSweepRMSE(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # Load precomputed data
        results = np.load(_RESULTS_PATH, allow_pickle=True)
        turn_rates = results["turn_rates"]
        filter_names = list(results["filter_names"])
        mean_rmse = results["mean_rmse"]  # (R, F)
        std_rmse = results["std_rmse"]    # (R, F)

        with self.voiceover(text="Let's sweep across nonlinearity levels. We generate fifty trajectories at each turn rate, from zero (perfectly linear) to zero point five (strongly curved), and measure the RMSE for all four filters.") as tracker:
            title = Text("RMSE vs Nonlinearity", color=COLOR_TEXT,
                          font_size=TITLE_FONT_SIZE)
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # Build chart data
        y_data = {}
        y_std = {}
        for fi, fname in enumerate(filter_names):
            y_data[fname] = mean_rmse[:, fi]
            y_std[fname] = std_rmse[:, fi]

        chart = RMSELineChart(
            x_values=turn_rates,
            y_data=y_data,
            y_std=y_std,
            colors=COLORS,
            x_label="Turn Rate (rad/s)",
            y_label="Position RMSE",
        )
        chart.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(chart.axes), run_time=NORMAL_ANIM)

        # Animate each line sequentially
        voiceovers = {
            "KF": "The linear Kalman Filter in red. At low nonlinearity it performs well, but as curvature increases, its error grows rapidly.",
            "EKF": "The Extended Kalman Filter in orange. Its Jacobian-based linearization keeps errors lower on curved paths.",
            "UKF": "The Unscented Kalman Filter in teal. Sigma points capture nonlinearity better than first-order Jacobians.",
            "PF": "And the Particle Filter in gold. With enough particles, it matches or beats the UKF across all regimes.",
        }

        for fname in filter_names:
            with self.voiceover(text=voiceovers[fname]) as tracker:
                anims = chart.animate_line(fname)
                self.play(*anims, run_time=SLOW_ANIM)
                self.wait(PAUSE_SHORT)

        self.wait(PAUSE_MEDIUM)

        # Legend
        legend = VGroup()
        for fname in filter_names:
            item = VGroup(
                Line(ORIGIN, RIGHT * 0.4, color=COLORS[fname], stroke_width=3),
                Text(fname, color=COLORS[fname], font_size=SMALL_FONT_SIZE),
            ).arrange(RIGHT, buff=0.1)
            legend.add(item)
        legend.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        legend.to_corner(UR, buff=0.4).set_z_index(10)

        with self.voiceover(text="The key takeaway: at low nonlinearity, all filters perform similarly. As curvature increases, the linear KF falls behind first, then the EKF, while UKF and PF remain robust. These results are averaged over fifty trials per data point, so the trend is statistically reliable.") as tracker:
            self.play(FadeIn(legend), run_time=FAST_ANIM)

            note = make_observation_note(
                "50 trials per turn rate.\n"
                "Shaded bands show ±1 std dev.",
            )
            self.play(FadeIn(note), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG * 2)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
