"""Scene 7: Demo — Pedestrian Trajectory Filtering

Applies the Kalman Filter to a simulated pedestrian trajectory.
Shows true path (white dashed), noisy measurements (blue dots),
and filtered estimate (gold) with breathing covariance ellipse.
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.gaussian_ellipse import GaussianEllipse
from kalman_manim.data.generators import generate_pedestrian_trajectory
from filters.kalman import KalmanFilter


class SceneDemoTrajectory(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Generate trajectory ─────────────────────────────────────────
        data = generate_pedestrian_trajectory(
            n_steps=40, dt=0.5, speed=0.6,
            process_noise_std=0.1, measurement_noise_std=0.5,
            turn_probability=0.1, seed=7,
        )
        true_states = data["true_states"]
        measurements = data["measurements"]
        dt = data["dt"]

        # ── Run Kalman Filter ───────────────────────────────────────────
        F = np.array([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1,  0],
                       [0, 0, 0,  1]])
        H = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]])
        q = 0.05
        Q = q * np.array([[dt**3/3, 0, dt**2/2, 0],
                           [0, dt**3/3, 0, dt**2/2],
                           [dt**2/2, 0, dt, 0],
                           [0, dt**2/2, 0, dt]])
        R = 0.25 * np.eye(2)

        kf = KalmanFilter(
            F=F, H=H, Q=Q, R=R,
            x0=np.array([0, 0, 0.6, 0]),
            P0=np.diag([1, 1, 0.5, 0.5]),
        )
        results = kf.run(measurements)

        # ── Scale and center ────────────────────────────────────────────
        all_pos = np.vstack([true_states[:, :2], measurements])
        scale = 4.0 / max(np.ptp(all_pos[:, 0]), np.ptp(all_pos[:, 1]), 1)
        center = all_pos.mean(axis=0)

        def to_scene(xy):
            s = (xy - center) * scale
            return np.array([s[0], s[1], 0])

        # ── Title ───────────────────────────────────────────────────────
        title = Text("Kalman Filter in Action", color=COLOR_TEXT,
                      font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3)
        title.set_z_index(10)

        # ── Legend ──────────────────────────────────────────────────────
        legend_items = VGroup(
            VGroup(
                Dot(color=COLOR_TRUE_PATH, radius=0.04),
                Text("True path", color=COLOR_TRUE_PATH, font_size=16),
            ).arrange(RIGHT, buff=0.1),
            VGroup(
                Dot(color=COLOR_MEASUREMENT, radius=0.04),
                Text("Measurement", color=COLOR_MEASUREMENT, font_size=16),
            ).arrange(RIGHT, buff=0.1),
            VGroup(
                Dot(color=COLOR_POSTERIOR, radius=0.04),
                Text("KF estimate", color=COLOR_POSTERIOR, font_size=16),
            ).arrange(RIGHT, buff=0.1),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        legend_items.to_corner(UL, buff=0.3)
        legend_items.set_z_index(10)

        with self.voiceover(text="Let's watch the Kalman Filter track a pedestrian in real time. White is the true path, blue dots are GPS measurements, and gold is the filter's estimate.") as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.play(FadeIn(legend_items), run_time=FAST_ANIM)

        # ── Animate step by step ────────────────────────────────────────
        # Start with initial position
        pos_est = to_scene(results["x_estimates"][0][:2])
        P_2d = results["P_estimates"][0][:2, :2]

        # Current estimate dot + ellipse
        est_dot = Dot(pos_est, color=COLOR_POSTERIOR, radius=DOT_RADIUS_MEDIUM)
        est_ellipse = GaussianEllipse(
            mean=results["x_estimates"][0][:2],
            cov=P_2d * scale**2,  # scale covariance to scene units
            color=COLOR_PREDICTION,
            n_sigma=2,
            fill_opacity=0.15,
        )
        est_ellipse.move_to(pos_est)

        # Trails
        true_trail = VMobject(color=COLOR_TRUE_PATH, stroke_width=1.5, stroke_opacity=0.6)
        est_trail = VMobject(color=COLOR_POSTERIOR, stroke_width=2.5)

        self.add(true_trail, est_trail, est_ellipse, est_dot)

        # True path points accumulated
        true_points = [to_scene(true_states[0, :2])]
        est_points = [pos_est]

        with self.voiceover(text="Each cycle, a new measurement arrives. The filter predicts, compares to the measurement, and updates. Watch the uncertainty ellipse breathe — growing during prediction, shrinking during update.") as tracker:
            pass  # Voice plays while loop runs

        for k in range(len(measurements)):
            # True position at step k+1
            true_pt = to_scene(true_states[k + 1, :2])
            true_points.append(true_pt)

            # Measurement
            meas_pt = to_scene(measurements[k])
            meas_dot = Dot(meas_pt, color=COLOR_MEASUREMENT,
                           radius=MEASUREMENT_DOT_RADIUS, fill_opacity=0.6)

            # Estimate
            est_pt = to_scene(results["x_estimates"][k][:2])
            est_points.append(est_pt)

            P_2d_k = results["P_estimates"][k][:2, :2]
            new_ellipse = GaussianEllipse(
                mean=results["x_estimates"][k][:2],
                cov=P_2d_k * scale**2,
                color=COLOR_PREDICTION,
                n_sigma=2,
                fill_opacity=0.15,
            )
            new_ellipse.move_to(est_pt)

            # Update trails
            if len(true_points) >= 2:
                new_true_trail = VMobject(color=COLOR_TRUE_PATH,
                                          stroke_width=1.5, stroke_opacity=0.6)
                new_true_trail.set_points_smoothly(true_points)

                new_est_trail = VMobject(color=COLOR_POSTERIOR, stroke_width=2.5)
                new_est_trail.set_points_smoothly(est_points)
            else:
                new_true_trail = true_trail
                new_est_trail = est_trail

            anims = [
                FadeIn(meas_dot, scale=1.2),
                Transform(est_ellipse, new_ellipse),
                est_dot.animate.move_to(est_pt),
                Transform(true_trail, new_true_trail),
                Transform(est_trail, new_est_trail),
            ]

            # Camera gently follows the action
            if k % 5 == 0:
                anims.append(
                    self.camera.frame.animate.move_to(
                        est_pt + UP * 1.5
                    )
                )

            self.play(*anims, run_time=0.25)

        # ── Zoom out to see full picture ────────────────────────────────
        with self.voiceover(text="Let's zoom out and see the full picture.") as tracker:
            self.play(
                self.camera.frame.animate.move_to(ORIGIN).set(width=14),
                run_time=SLOW_ANIM,
            )
            self.wait(PAUSE_LONG)

        # ── Summary text ────────────────────────────────────────────────
        summary = Text(
            "The Kalman Filter recovers a smooth trajectory\n"
            "from noisy measurements",
            color=COLOR_POSTERIOR, font_size=BODY_FONT_SIZE,
            line_spacing=1.2,
        )
        summary.to_edge(DOWN, buff=0.3)
        summary.set_z_index(10)

        with self.voiceover(text="From noisy, scattered measurements, the Kalman Filter has reconstructed a smooth, accurate trajectory. This is optimal Bayesian inference in action.") as tracker:
            self.play(FadeIn(summary), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
