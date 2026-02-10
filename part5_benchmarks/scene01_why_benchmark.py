"""Part 5, Scene 1: Why Benchmark? — One trajectory is misleading.

Data: synthetic (3 seeds showing different filter rankings)

Shows the same trajectory type with different random seeds,
where the filter ranking changes. Motivates corpus-level evaluation.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.observation_note import make_observation_note
from kalman_manim.data.generators import generate_nonlinear_trajectory
from benchmarks.configs import make_all_filters, FILTER_NAMES
from benchmarks.metrics import position_rmse
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneWhyBenchmark(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        colors = {
            "KF": COLOR_FILTER_KF, "EKF": COLOR_FILTER_EKF,
            "UKF": COLOR_FILTER_UKF, "PF": COLOR_FILTER_PF,
        }

        with self.voiceover(text="So far, we've compared filters on single trajectories. But a single trial can be misleading. Watch what happens with three different random seeds.") as tracker:
            title = Text("Why Benchmark?", color=COLOR_TEXT,
                          font_size=TITLE_FONT_SIZE)
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # Run 3 seeds and show how rankings change
        seeds = [10, 25, 77]
        panels = VGroup()

        for idx, seed in enumerate(seeds):
            data = generate_nonlinear_trajectory(
                n_steps=40, dt=0.5, turn_rate=0.2,
                measurement_noise_std=0.5, seed=seed,
            )
            true_states = data["true_states"]
            meas = data["measurements"]
            x0 = true_states[0, :4]

            filters = make_all_filters(dt=0.5, x0=x0, pf_seed=seed)
            errors = {}
            for name in FILTER_NAMES:
                res = filters[name].run(meas)
                ts = true_states[:, :4]
                errors[name] = position_rmse(res["x_estimates"], ts)

            # Sort by RMSE
            ranked = sorted(errors.items(), key=lambda kv: kv[1])

            # Build ranking panel
            panel_title = Text(f"Seed {seed}", color=COLOR_TEXT,
                               font_size=BODY_FONT_SIZE)
            rows = VGroup()
            for rank, (name, rmse) in enumerate(ranked, 1):
                row = VGroup(
                    Text(f"#{rank}", color=SLATE, font_size=SMALL_FONT_SIZE),
                    Text(name, color=colors[name], font_size=SMALL_FONT_SIZE),
                    Text(f"{rmse:.3f}", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
                ).arrange(RIGHT, buff=0.2)
                rows.add(row)
            rows.arrange(DOWN, buff=0.1, aligned_edge=LEFT)

            panel = VGroup(panel_title, rows).arrange(DOWN, buff=0.2)
            panels.add(panel)

        panels.arrange(RIGHT, buff=0.8)
        panels.next_to(title, DOWN, buff=LARGE_BUFF)

        with self.voiceover(text="With seed ten, the UKF wins. With seed twenty-five, the EKF is best. With seed seventy-seven, the particle filter takes the lead. The ranking depends on the specific noise realization!") as tracker:
            for panel in panels:
                self.play(FadeIn(panel, shift=UP * 0.3), run_time=NORMAL_ANIM)
                self.wait(PAUSE_SHORT)

        self.wait(PAUSE_MEDIUM)

        # Highlight the problem
        with self.voiceover(text="One trajectory proves nothing. We need hundreds of trials across varying conditions to draw reliable conclusions. That's what statistical benchmarking gives us.") as tracker:
            problem = Text(
                "One trajectory proves nothing.",
                color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
            )
            problem.to_edge(DOWN, buff=0.5)
            self.play(FadeIn(problem, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

            solution = Text(
                "We need hundreds of trials.",
                color=TEAL, font_size=HEADING_FONT_SIZE,
            )
            solution.to_edge(DOWN, buff=0.5)
            self.play(
                FadeOut(problem),
                FadeIn(solution, shift=UP * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
