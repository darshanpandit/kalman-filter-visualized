"""Part 5, Scene 3: Corpus Dashboard — Bar charts, histograms, timing.

Data: precomputed corpus_results.npz and timing_results.npz

Sequential dashboard: bar chart of average RMSE, error distribution
histograms, and computation time comparison.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.charts import FilterBarChart, ErrorHistogram
from kalman_manim.mobjects.observation_note import make_observation_note
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

_CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "benchmark_results", "corpus_results.npz"
)
_TIMING_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "benchmark_results", "timing_results.npz"
)

COLORS = {
    "KF": COLOR_FILTER_KF, "EKF": COLOR_FILTER_EKF,
    "UKF": COLOR_FILTER_UKF, "PF": COLOR_FILTER_PF,
}


class SceneCorpusDashboard(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Load data ─────────────────────────────────────────────────────
        corpus = np.load(_CORPUS_PATH, allow_pickle=True)
        timing = np.load(_TIMING_PATH, allow_pickle=True)

        filter_names = list(corpus["filter_names"])
        rmse_all = corpus["rmse"]     # (N_traj, N_filters)
        regimes = corpus["regimes"]

        timing_names = list(timing["filter_names"])
        timing_mean = timing["mean_s"]
        timing_std = timing["std_s"]

        # ── Title ─────────────────────────────────────────────────────────
        with self.voiceover(text="Now let's see the full picture. We ran all four filters on a mixed corpus of synthetic and real trajectories — linear, pedestrian, curved, and sharp turns.") as tracker:
            title = Text("Corpus Dashboard", color=COLOR_TEXT,
                          font_size=TITLE_FONT_SIZE)
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

            n_traj = rmse_all.shape[0]
            subtitle = Text(
                f"{n_traj} trajectories across 5 regimes",
                color=SLATE, font_size=SMALL_FONT_SIZE,
            )
            subtitle.next_to(title, DOWN, buff=0.15)
            self.play(FadeIn(subtitle), run_time=FAST_ANIM)

        # ── Bar chart: Average RMSE ───────────────────────────────────────
        # Compute mean RMSE per filter (ignoring NaN)
        mean_rmse = np.nanmean(rmse_all, axis=0)
        std_rmse = np.nanstd(rmse_all, axis=0)

        bar_chart = FilterBarChart(
            filter_names=filter_names,
            values=mean_rmse,
            errors=std_rmse,
            colors=COLORS,
            title="Average RMSE (all regimes)",
            width=5.5,
            height=3.0,
        )
        bar_chart.move_to(LEFT * 3.2 + DOWN * 0.5)

        with self.voiceover(text="Averaged across the entire corpus, the standard KF has the highest error. The EKF improves significantly, and the UKF and Particle Filter are neck and neck at the lowest error.") as tracker:
            self.play(FadeIn(bar_chart.axes), run_time=FAST_ANIM)
            anims = bar_chart.animate_bars(run_time=NORMAL_ANIM)
            self.play(*anims, run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Histograms ────────────────────────────────────────────────────
        # Build per-filter RMSE arrays for histogram
        hist_data = {}
        for fi, fname in enumerate(filter_names):
            vals = rmse_all[:, fi]
            hist_data[fname] = vals[~np.isnan(vals)]

        histogram = ErrorHistogram(
            data=hist_data,
            colors=COLORS,
            n_bins=20,
            width=5.5,
            height=3.0,
        )
        histogram.move_to(RIGHT * 3.2 + DOWN * 0.5)

        with self.voiceover(text="The error distributions tell a richer story. The KF distribution has a long right tail — occasionally very bad on curved paths. The UKF and PF distributions are tighter and shifted left.") as tracker:
            self.play(FadeIn(histogram.axes), run_time=FAST_ANIM)
            for fname in filter_names:
                anims = histogram.animate_histogram(fname)
                self.play(*anims, run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # ── Fade and show timing ──────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects
              if mob is not title and mob is not subtitle],
            run_time=NORMAL_ANIM,
        )

        # Timing bar chart
        timing_chart = FilterBarChart(
            filter_names=filter_names,
            values=timing_mean * 1000,  # convert to ms
            errors=timing_std * 1000,
            colors=COLORS,
            title="Computation Time (ms per trajectory)",
            width=7.0,
            height=3.5,
        )
        timing_chart.next_to(subtitle, DOWN, buff=0.6)

        with self.voiceover(text="But accuracy isn't free. The KF and EKF are the fastest — just matrix operations. The UKF is about three times slower due to sigma point propagation. And the Particle Filter, with three hundred particles, is roughly ten times slower than the KF. For real-time applications, this trade-off matters.") as tracker:
            self.play(FadeIn(timing_chart.axes), run_time=FAST_ANIM)
            anims = timing_chart.animate_bars()
            self.play(*anims, run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

            note = make_observation_note(
                "PF cost scales with particle count.\n"
                "300 particles used here.",
            )
            self.play(FadeIn(note), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
