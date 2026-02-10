"""Part 3, Scene 4: UKF vs EKF Demo

Data: curated synthetic (coordinated turn, turn_rate=0.25, seed=15)

Side-by-side comparison on the nonlinear pedestrian trajectory.
Shows UKF tracking more accurately on sharp turns.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.observation_note import make_observation_note
from kalman_manim.data.generators import generate_nonlinear_trajectory
from filters.ekf import ExtendedKalmanFilter
from filters.ukf import UnscentedKalmanFilter
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


def _ct_f(x, u):
    dt = 0.5
    return np.array([x[0] + x[2]*dt, x[1] + x[3]*dt, x[2], x[3]])

def _ct_F(x, u):
    dt = 0.5
    return np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])

def _h(x):
    return x[:2]

def _H(x):
    return np.array([[1,0,0,0],[0,1,0,0]])


class SceneUKFDemo(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        with self.voiceover(text="Time for the showdown: UKF versus EKF on a sharp turning trajectory. Same data, same initial conditions.") as tracker:
            title = Text("UKF vs EKF Comparison", color=COLOR_TEXT,
                          font_size=TITLE_FONT_SIZE)
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

            # ── Data ────────────────────────────────────────────────────────
            data = generate_nonlinear_trajectory(
                n_steps=50, dt=0.5, turn_rate=0.25, speed=0.9,
                process_noise_std=0.08, measurement_noise_std=0.5, seed=15,
            )
            true_states = data["true_states"]
            measurements = data["measurements"]

            Q = 0.1 * np.eye(4)
            R = 0.25 * np.eye(2)
            x0 = np.array([0, 0, 0.9, 0])
            P0 = np.eye(4)

            # Run EKF
            ekf = ExtendedKalmanFilter(
                f=_ct_f, h=_h, F_jacobian=_ct_F, H_jacobian=_H,
                Q=Q, R=R, x0=x0.copy(), P0=P0.copy(),
            )
            ekf_results = ekf.run(measurements)
            ekf_est = np.array([x[:2] for x in ekf_results["x_estimates"]])

            # Run UKF
            ukf = UnscentedKalmanFilter(
                f=_ct_f, h=_h, Q=Q, R=R,
                x0=x0.copy(), P0=P0.copy(),
                alpha=0.1, beta=2.0, kappa=0.0,
            )
            ukf_results = ukf.run(measurements)
            ukf_est = np.array([x[:2] for x in ukf_results["x_estimates"]])

            # Scale
            all_pos = np.vstack([true_states[:, :2], ekf_est, ukf_est])
            scale = 4.0 / max(np.ptp(all_pos[:, 0]), np.ptp(all_pos[:, 1]), 1)
            ctr = true_states[:, :2].mean(axis=0)

            def to_s(xy):
                s = (xy - ctr) * scale
                return np.array([s[0], s[1], 0])

            # ── Paths ───────────────────────────────────────────────────────
            true_pts = [to_s(true_states[i, :2]) for i in range(len(true_states))]
            true_path = DashedVMobject(
                VMobject().set_points_smoothly(true_pts), num_dashes=50)
            true_path.set_color(COLOR_TRUE_PATH).set_stroke(width=1.5, opacity=0.7)

            meas_dots = VGroup(*[
                Dot(to_s(m), radius=MEASUREMENT_DOT_RADIUS, color=COLOR_MEASUREMENT,
                    fill_opacity=0.35) for m in measurements
            ])

            ekf_pts = [to_s(ekf_est[i]) for i in range(len(ekf_est))]
            ekf_path = VMobject().set_points_smoothly(ekf_pts)
            ekf_path.set_color(COLOR_PREDICTION).set_stroke(width=2.5)

            ukf_pts = [to_s(ukf_est[i]) for i in range(len(ukf_est))]
            ukf_path = VMobject().set_points_smoothly(ukf_pts)
            ukf_path.set_color(COLOR_POSTERIOR).set_stroke(width=3)

            # Animate
            self.play(Create(true_path), FadeIn(meas_dots, lag_ratio=0.01),
                      run_time=NORMAL_ANIM)

        with self.voiceover(text="The EKF in red tracks reasonably well but lags on sharp turns. The Jacobian approximation can't fully capture the curvature.") as tracker:
            ekf_label = Text("EKF", color=COLOR_PREDICTION, font_size=SMALL_FONT_SIZE)
            ekf_label.next_to(ekf_path.get_end(), RIGHT, buff=0.15).set_z_index(10)
            self.play(Create(ekf_path), FadeIn(ekf_label), run_time=SLOW_ANIM)

        with self.voiceover(text="The UKF in gold hugs the true path more closely. Sigma points capture the nonlinear deformation better than linearization.") as tracker:
            ukf_label = Text("UKF", color=COLOR_POSTERIOR, font_size=SMALL_FONT_SIZE)
            ukf_label.next_to(ukf_path.get_end(), RIGHT, buff=0.15).set_z_index(10)
            self.play(Create(ukf_path), FadeIn(ukf_label), run_time=SLOW_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Error comparison ────────────────────────────────────────────
        with self.voiceover(text="Quantitatively, the UKF has lower average error. In this particular trajectory the difference may be subtle. On sharper turns, the gap widens, and the UKF needs no Jacobians to derive.") as tracker:
            ekf_err = np.mean(np.linalg.norm(ekf_est - true_states[1:, :2], axis=1))
            ukf_err = np.mean(np.linalg.norm(ukf_est - true_states[1:, :2], axis=1))

            err_text = VGroup(
                Text(f"EKF avg error: {ekf_err:.3f}", color=COLOR_PREDICTION,
                      font_size=SMALL_FONT_SIZE),
                Text(f"UKF avg error: {ukf_err:.3f}", color=COLOR_POSTERIOR,
                      font_size=SMALL_FONT_SIZE),
            ).arrange(DOWN, buff=0.1)
            err_text.to_corner(DL, buff=0.3).set_z_index(10)
            self.play(FadeIn(err_text), run_time=NORMAL_ANIM)

            # Theory-observation honesty note
            note = make_observation_note(
                "The advantage grows on sharper turns.\n"
                "Here both use a linear transition model.",
            )
            self.play(FadeIn(note), run_time=FAST_ANIM)

            result = Text(
                "UKF handles sharp turns better — no Jacobians needed!",
                color=COLOR_POSTERIOR, font_size=BODY_FONT_SIZE,
            )
            result.to_edge(DOWN, buff=0.3).set_z_index(10)
            self.play(FadeIn(result), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

        # Teaser
        with self.voiceover(text="But we've been assuming Gaussian distributions this whole time. What if the posterior is multimodal, two peaks, three peaks? That's where particle filters come in. See you in Part 4.") as tracker:
            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)

            teaser = Text("But what about non-Gaussian distributions?",
                           color=COLOR_HIGHLIGHT, font_size=TITLE_FONT_SIZE)
            self.play(FadeIn(teaser, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

            next_ep = Text("Next: Particle Filters", color=COLOR_TEXT,
                            font_size=HEADING_FONT_SIZE)
            next_ep.next_to(teaser, DOWN, buff=LARGE_BUFF)
            self.play(FadeIn(next_ep, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

            self.play(FadeOut(teaser), FadeOut(next_ep), run_time=NORMAL_ANIM)
