"""Scene 1: Hook — 'Where is the pedestrian?'

Shows noisy LBS/GPS pings of a pedestrian walking, then reveals the true path
and teases the Kalman-filtered result.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.data.generators import generate_pedestrian_trajectory
from kalman_manim.mobjects.trajectory import PedestrianPath
from filters.kalman import KalmanFilter


class SceneHook(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Generate data ───────────────────────────────────────────────
        data = generate_pedestrian_trajectory(
            n_steps=60, dt=0.5, speed=0.8,
            process_noise_std=0.15, measurement_noise_std=0.6,
            turn_probability=0.08, seed=42,
        )
        true_states = data["true_states"]
        measurements = data["measurements"]

        # Run KF to get filtered estimates
        dt = data["dt"]
        F = np.array([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1,  0],
                       [0, 0, 0,  1]])
        H = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]])
        Q = 0.1 * np.eye(4)
        R = 0.36 * np.eye(2)
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R,
                          x0=np.array([0, 0, 0.8, 0]),
                          P0=np.eye(4))
        results = kf.run(measurements)
        estimates = np.array([x[:2] for x in results["x_estimates"]])

        # Scale to fit Manim scene coordinates (roughly ±5)
        scale = 3.0 / max(
            np.ptp(true_states[:, 0]), np.ptp(true_states[:, 1]), 1
        )
        true_pos = true_states[:, :2] * scale
        meas_scaled = measurements * scale
        est_scaled = estimates * scale

        # Center everything
        center = true_pos.mean(axis=0)
        true_pos -= center
        meas_scaled -= center
        est_scaled -= center

        # ── Title ───────────────────────────────────────────────────────
        title = Text(
            "Where is the pedestrian?",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
        self.wait(PAUSE_SHORT)

        # ── Noisy measurements appear one by one ────────────────────────
        meas_dots = VGroup()
        for i, m in enumerate(meas_scaled):
            dot = Dot(
                np.array([m[0], m[1], 0]),
                radius=MEASUREMENT_DOT_RADIUS,
                color=COLOR_MEASUREMENT,
                fill_opacity=0.8,
            )
            meas_dots.add(dot)

        # Reveal in batches for pacing
        batch_size = 8
        for start in range(0, len(meas_dots), batch_size):
            batch = meas_dots[start:start + batch_size]
            self.play(
                LaggedStart(
                    *[FadeIn(d, scale=1.5) for d in batch],
                    lag_ratio=0.15,
                ),
                run_time=0.8,
            )
        self.wait(PAUSE_MEDIUM)

        # ── Subtitle ───────────────────────────────────────────────────
        subtitle = Text(
            "Your phone says you're here... but are you really?",
            color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
        )
        subtitle.next_to(title, DOWN, buff=SMALL_BUFF)
        self.play(FadeIn(subtitle), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG)

        # ── Reveal true path ────────────────────────────────────────────
        true_points = [np.array([p[0], p[1], 0]) for p in true_pos]
        true_path = VMobject()
        true_path.set_points_smoothly(true_points)
        true_path.set_color(COLOR_TRUE_PATH)
        true_path.set_stroke(width=2, opacity=0.8)
        true_path_dashed = DashedVMobject(true_path, num_dashes=40)

        self.play(Create(true_path_dashed), run_time=SLOW_ANIM)
        self.wait(PAUSE_MEDIUM)

        # ── Question ───────────────────────────────────────────────────
        question = Text(
            "Can we recover the true trajectory from noisy measurements?",
            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
        )
        question.to_edge(DOWN, buff=0.5)
        self.play(
            FadeOut(subtitle),
            FadeIn(question, shift=UP * 0.2),
            run_time=NORMAL_ANIM,
        )
        self.wait(PAUSE_LONG)

        # ── Tease: Kalman-filtered result ───────────────────────────────
        est_points = [np.array([p[0], p[1], 0]) for p in est_scaled]
        est_path = VMobject()
        est_path.set_points_smoothly(est_points)
        est_path.set_color(COLOR_POSTERIOR)
        est_path.set_stroke(width=3)

        self.play(Create(est_path), run_time=SLOW_ANIM)
        self.wait(PAUSE_SHORT)

        answer = Text(
            "Yes — with a Kalman Filter.",
            color=COLOR_POSTERIOR, font_size=HEADING_FONT_SIZE,
        )
        answer.to_edge(DOWN, buff=0.5)
        self.play(
            FadeOut(question),
            FadeIn(answer, shift=UP * 0.2),
            run_time=NORMAL_ANIM,
        )
        self.wait(PAUSE_LONG * 2)

        # ── Fade out ───────────────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=NORMAL_ANIM,
        )
