"""Part 2, Scene 1: Why the Linear KF Fails

Shows a pedestrian taking curved turns. Applies linear KF → it diverges.
Motivates the need for handling nonlinearity.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.data.generators import generate_nonlinear_trajectory
from filters.kalman import KalmanFilter


class SceneEKFMotivation(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────────
        title = Text("When Linearity Breaks", color=COLOR_TEXT,
                      font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3).set_z_index(10)
        self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Generate curved trajectory ──────────────────────────────────
        data = generate_nonlinear_trajectory(
            n_steps=50, dt=0.5, turn_rate=0.2, speed=0.8,
            process_noise_std=0.05, measurement_noise_std=0.4, seed=10,
        )
        true_states = data["true_states"]
        measurements = data["measurements"]
        dt = data["dt"]

        # ── Run LINEAR KF (will fail on curved path) ───────────────────
        F = np.array([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1,  0],
                       [0, 0, 0,  1]])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        Q = 0.1 * np.eye(4)
        R = 0.16 * np.eye(2)
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R,
                          x0=np.array([0, 0, 0.8, 0]), P0=np.eye(4))
        results = kf.run(measurements)
        kf_estimates = np.array([x[:2] for x in results["x_estimates"]])

        # ── Scale and center ────────────────────────────────────────────
        all_pos = np.vstack([true_states[:, :2], measurements, kf_estimates])
        scale = 4.0 / max(np.ptp(all_pos[:, 0]), np.ptp(all_pos[:, 1]), 1)
        center = true_states[:, :2].mean(axis=0)

        def to_s(xy):
            s = (xy - center) * scale
            return np.array([s[0], s[1], 0])

        # ── Draw true path ──────────────────────────────────────────────
        true_pts = [to_s(true_states[i, :2]) for i in range(len(true_states))]
        true_path = DashedVMobject(
            VMobject().set_points_smoothly(true_pts), num_dashes=50)
        true_path.set_color(COLOR_TRUE_PATH).set_stroke(width=1.5, opacity=0.7)
        self.play(Create(true_path), run_time=NORMAL_ANIM)

        # ── Measurements ────────────────────────────────────────────────
        meas_dots = VGroup(*[
            Dot(to_s(m), radius=MEASUREMENT_DOT_RADIUS, color=COLOR_MEASUREMENT,
                fill_opacity=0.5)
            for m in measurements
        ])
        self.play(FadeIn(meas_dots, lag_ratio=0.02), run_time=NORMAL_ANIM)

        # ── Linear KF result (it lags and cuts corners) ─────────────────
        kf_pts = [to_s(kf_estimates[i]) for i in range(len(kf_estimates))]
        kf_path = VMobject().set_points_smoothly(kf_pts)
        kf_path.set_color(COLOR_PREDICTION).set_stroke(width=3)

        kf_label = Text("Linear KF", color=COLOR_PREDICTION,
                         font_size=SMALL_FONT_SIZE)
        kf_label.next_to(kf_path.get_end(), RIGHT, buff=0.2)

        self.play(Create(kf_path), FadeIn(kf_label), run_time=SLOW_ANIM)
        self.wait(PAUSE_MEDIUM)

        # ── Highlight the failure ───────────────────────────────────────
        fail_text = Text(
            "The linear KF can't handle curves!\n"
            "It assumes straight-line motion.",
            color=COLOR_PREDICTION, font_size=BODY_FONT_SIZE,
            line_spacing=1.2,
        )
        fail_text.to_edge(DOWN, buff=0.4).set_z_index(10)
        self.play(FadeIn(fail_text), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG)

        # ── Solution teaser ─────────────────────────────────────────────
        solution = Text(
            "Solution: Linearize around the current estimate",
            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
        )
        solution.to_edge(DOWN, buff=0.4).set_z_index(10)
        self.play(FadeOut(fail_text), FadeIn(solution), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
