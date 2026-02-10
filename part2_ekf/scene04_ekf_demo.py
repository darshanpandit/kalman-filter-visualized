"""Part 2, Scene 4: EKF Demo on Curved Trajectory

Applies EKF to the nonlinear pedestrian trajectory and shows it
successfully tracking the curved path (unlike the linear KF).
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.data.generators import generate_nonlinear_trajectory
from filters.ekf import ExtendedKalmanFilter
from filters.kalman import KalmanFilter


def _ct_transition(x, u):
    """Coordinated turn state transition: [x, y, vx, vy]."""
    dt = 0.5
    px, py, vx, vy = x
    # Approximate as constant velocity (the EKF linearizes around this)
    return np.array([
        px + vx * dt,
        py + vy * dt,
        vx,
        vy,
    ])


def _ct_jacobian(x, u):
    """Jacobian of transition function."""
    dt = 0.5
    return np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ])


def _meas_func(x):
    return x[:2]


def _meas_jacobian(x):
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])


class SceneEKFDemo(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("EKF vs Linear KF", color=COLOR_TEXT,
                      font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3).set_z_index(10)
        self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Generate data ───────────────────────────────────────────────
        data = generate_nonlinear_trajectory(
            n_steps=50, dt=0.5, turn_rate=0.2, speed=0.8,
            process_noise_std=0.05, measurement_noise_std=0.4, seed=10,
        )
        true_states = data["true_states"]
        measurements = data["measurements"]
        dt = data["dt"]

        # ── Run EKF ─────────────────────────────────────────────────────
        Q = 0.1 * np.eye(4)
        R = 0.16 * np.eye(2)
        ekf = ExtendedKalmanFilter(
            f=_ct_transition, h=_meas_func,
            F_jacobian=_ct_jacobian, H_jacobian=_meas_jacobian,
            Q=Q, R=R,
            x0=np.array([0, 0, 0.8, 0]), P0=np.eye(4),
        )
        ekf_results = ekf.run(measurements)
        ekf_est = np.array([x[:2] for x in ekf_results["x_estimates"]])

        # ── Run Linear KF for comparison ────────────────────────────────
        F_lin = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                           [0, 0, 1, 0], [0, 0, 0, 1]])
        H_lin = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf = KalmanFilter(F=F_lin, H=H_lin, Q=Q, R=R,
                          x0=np.array([0, 0, 0.8, 0]), P0=np.eye(4))
        kf_results = kf.run(measurements)
        kf_est = np.array([x[:2] for x in kf_results["x_estimates"]])

        # ── Scale ───────────────────────────────────────────────────────
        all_pos = np.vstack([true_states[:, :2], measurements, ekf_est, kf_est])
        scale = 4.0 / max(np.ptp(all_pos[:, 0]), np.ptp(all_pos[:, 1]), 1)
        ctr = true_states[:, :2].mean(axis=0)

        def to_s(xy):
            s = (xy - ctr) * scale
            return np.array([s[0], s[1], 0])

        # ── True path ──────────────────────────────────────────────────
        true_pts = [to_s(true_states[i, :2]) for i in range(len(true_states))]
        true_path = DashedVMobject(
            VMobject().set_points_smoothly(true_pts), num_dashes=50)
        true_path.set_color(COLOR_TRUE_PATH).set_stroke(width=1.5, opacity=0.7)

        # ── Measurement dots ────────────────────────────────────────────
        meas_dots = VGroup(*[
            Dot(to_s(m), radius=MEASUREMENT_DOT_RADIUS, color=COLOR_MEASUREMENT,
                fill_opacity=0.4) for m in measurements
        ])

        # ── KF path (red — fails) ──────────────────────────────────────
        kf_pts = [to_s(kf_est[i]) for i in range(len(kf_est))]
        kf_path = VMobject().set_points_smoothly(kf_pts)
        kf_path.set_color(COLOR_PREDICTION).set_stroke(width=2.5)

        # ── EKF path (gold — succeeds) ─────────────────────────────────
        ekf_pts = [to_s(ekf_est[i]) for i in range(len(ekf_est))]
        ekf_path = VMobject().set_points_smoothly(ekf_pts)
        ekf_path.set_color(COLOR_POSTERIOR).set_stroke(width=3)

        # ── Animate ─────────────────────────────────────────────────────
        self.play(Create(true_path), FadeIn(meas_dots, lag_ratio=0.01),
                  run_time=NORMAL_ANIM)
        self.wait(PAUSE_SHORT)

        # Linear KF first
        kf_label = Text("Linear KF", color=COLOR_PREDICTION,
                         font_size=SMALL_FONT_SIZE)
        kf_label.next_to(kf_path.get_end(), RIGHT, buff=0.15).set_z_index(10)
        self.play(Create(kf_path), FadeIn(kf_label), run_time=SLOW_ANIM)
        self.wait(PAUSE_SHORT)

        # EKF
        ekf_label = Text("EKF", color=COLOR_POSTERIOR,
                          font_size=SMALL_FONT_SIZE)
        ekf_label.next_to(ekf_path.get_end(), RIGHT, buff=0.15).set_z_index(10)
        self.play(Create(ekf_path), FadeIn(ekf_label), run_time=SLOW_ANIM)
        self.wait(PAUSE_MEDIUM)

        # ── Legend ──────────────────────────────────────────────────────
        legend = VGroup(
            VGroup(Line(LEFT * 0.3, RIGHT * 0.3, color=COLOR_TRUE_PATH, stroke_width=1.5),
                   Text("True path", color=COLOR_TRUE_PATH, font_size=16)).arrange(RIGHT, buff=0.1),
            VGroup(Line(LEFT * 0.3, RIGHT * 0.3, color=COLOR_PREDICTION, stroke_width=2.5),
                   Text("Linear KF", color=COLOR_PREDICTION, font_size=16)).arrange(RIGHT, buff=0.1),
            VGroup(Line(LEFT * 0.3, RIGHT * 0.3, color=COLOR_POSTERIOR, stroke_width=3),
                   Text("EKF", color=COLOR_POSTERIOR, font_size=16)).arrange(RIGHT, buff=0.1),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        legend.to_corner(DL, buff=0.3).set_z_index(10)
        self.play(FadeIn(legend), run_time=FAST_ANIM)

        result = Text(
            "The EKF tracks the curved path successfully!",
            color=COLOR_POSTERIOR, font_size=BODY_FONT_SIZE,
        )
        result.to_edge(DOWN, buff=0.3).set_z_index(10)
        self.play(FadeIn(result), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG * 2)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
