"""Part 4, Scene 4: Grand Comparison — KF vs EKF vs UKF vs PF

Data: real-world (ETH eth, pedestrian #171, first 50 steps)

All four filters applied to the same real-world trajectory.
Side-by-side paths with error metrics. Decision flowchart.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.observation_note import make_observation_note
from kalman_manim.data.loader import load_eth_trajectory
from filters.kalman import KalmanFilter
from filters.ekf import ExtendedKalmanFilter
from filters.ukf import UnscentedKalmanFilter
from filters.particle import ParticleFilter
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


def _f(x, u):
    dt = 0.5
    return np.array([x[0]+x[2]*dt, x[1]+x[3]*dt, x[2], x[3]])

def _F_jac(x, u):
    dt = 0.5
    return np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])

def _h(x):
    return x[:2]

def _H_jac(x):
    return np.array([[1,0,0,0],[0,1,0,0]])

def _pf_f(x, u, noise):
    dt = 0.5
    return np.array([x[0]+x[2]*dt+noise[0], x[1]+x[3]*dt+noise[1],
                      x[2]+noise[2], x[3]+noise[3]])


class SceneGrandComparison(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        with self.voiceover(text="The grand finale: all four filters on a real pedestrian trajectory from ETH Zurich. Let's see how they compare.") as tracker:
            title = Text("Grand Comparison", color=COLOR_TEXT,
                          font_size=TITLE_FONT_SIZE)
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

            # ── Generate data ───────────────────────────────────────────────
            data = load_eth_trajectory(
                sequence="eth", pedestrian_id=171,
                measurement_noise_std=0.45, max_steps=50, seed=20,
            )
            true_states = data["true_states"]
            meas = data["measurements"]
            dt = data["dt"]

            Q4 = 0.08 * np.eye(4)
            R2 = 0.2 * np.eye(2)
            x0 = data["true_states"][0]
            P0 = np.eye(4)

            # ── Run all four filters ────────────────────────────────────────
            # Linear KF
            F_lin = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
            H_lin = np.array([[1,0,0,0],[0,1,0,0]])
            kf = KalmanFilter(F=F_lin, H=H_lin, Q=Q4, R=R2, x0=x0.copy(), P0=P0.copy())
            kf_res = kf.run(meas)
            kf_est = np.array([x[:2] for x in kf_res["x_estimates"]])

            # EKF
            ekf = ExtendedKalmanFilter(f=_f, h=_h, F_jacobian=_F_jac, H_jacobian=_H_jac,
                                        Q=Q4, R=R2, x0=x0.copy(), P0=P0.copy())
            ekf_res = ekf.run(meas)
            ekf_est = np.array([x[:2] for x in ekf_res["x_estimates"]])

            # UKF
            ukf = UnscentedKalmanFilter(f=_f, h=_h, Q=Q4, R=R2,
                                         x0=x0.copy(), P0=P0.copy())
            ukf_res = ukf.run(meas)
            ukf_est = np.array([x[:2] for x in ukf_res["x_estimates"]])

            # PF
            Q_pf = np.diag([0.02, 0.02, 0.04, 0.04])
            pf = ParticleFilter(f=_pf_f, h=_h, Q=Q_pf, R=R2, n_particles=300,
                                 x0=x0.copy(), P0=P0.copy(), seed=42)
            pf_res = pf.run(meas)
            pf_est = np.array([x[:2] for x in pf_res["x_estimates"]])

            # ── Scale ───────────────────────────────────────────────────────
            all_pos = np.vstack([true_states[:, :2], kf_est, ekf_est, ukf_est, pf_est])
            scale = 4.0 / max(np.ptp(all_pos[:, 0]), np.ptp(all_pos[:, 1]), 1)
            ctr = true_states[:, :2].mean(axis=0)

            def to_s(xy):
                s = (xy - ctr) * scale
                return np.array([s[0], s[1], 0])

            # ── Draw paths ──────────────────────────────────────────────────
            # True path
            true_pts = [to_s(true_states[i, :2]) for i in range(len(true_states))]
            true_path = DashedVMobject(
                VMobject().set_points_smoothly(true_pts), num_dashes=50)
            true_path.set_color(COLOR_TRUE_PATH).set_stroke(width=1.5, opacity=0.7)
            self.play(Create(true_path), run_time=NORMAL_ANIM)

        # Filter paths — distinct comparison colors
        colors = {
            "KF": COLOR_FILTER_KF,
            "EKF": COLOR_FILTER_EKF,
            "UKF": COLOR_FILTER_UKF,
            "PF": COLOR_FILTER_PF,
        }
        estimates = {"KF": kf_est, "EKF": ekf_est, "UKF": ukf_est, "PF": pf_est}

        paths = {}

        with self.voiceover(text="The linear KF in red fails on curves, it assumes straight-line motion.") as tracker:
            name = "KF"
            est = estimates[name]
            pts = [to_s(est[i]) for i in range(len(est))]
            path = VMobject().set_points_smoothly(pts)
            path.set_color(colors[name]).set_stroke(width=2.5)
            paths[name] = path
            label = Text(name, color=colors[name], font_size=SMALL_FONT_SIZE)
            label.next_to(path.get_end(), RIGHT, buff=0.1).set_z_index(10)
            self.play(Create(path), FadeIn(label), run_time=NORMAL_ANIM)

        with self.voiceover(text="The EKF in orange improves significantly with its Jacobian-based linearization.") as tracker:
            name = "EKF"
            est = estimates[name]
            pts = [to_s(est[i]) for i in range(len(est))]
            path = VMobject().set_points_smoothly(pts)
            path.set_color(colors[name]).set_stroke(width=2.5)
            paths[name] = path
            label = Text(name, color=colors[name], font_size=SMALL_FONT_SIZE)
            label.next_to(path.get_end(), RIGHT, buff=0.1).set_z_index(10)
            self.play(Create(path), FadeIn(label), run_time=NORMAL_ANIM)

        with self.voiceover(text="The UKF in teal tracks even more accurately using sigma points.") as tracker:
            name = "UKF"
            est = estimates[name]
            pts = [to_s(est[i]) for i in range(len(est))]
            path = VMobject().set_points_smoothly(pts)
            path.set_color(colors[name]).set_stroke(width=2.5)
            paths[name] = path
            label = Text(name, color=colors[name], font_size=SMALL_FONT_SIZE)
            label.next_to(path.get_end(), RIGHT, buff=0.1).set_z_index(10)
            self.play(Create(path), FadeIn(label), run_time=NORMAL_ANIM)

        with self.voiceover(text="And the particle filter in gold matches the UKF, with no distributional assumptions.") as tracker:
            name = "PF"
            est = estimates[name]
            pts = [to_s(est[i]) for i in range(len(est))]
            path = VMobject().set_points_smoothly(pts)
            path.set_color(colors[name]).set_stroke(width=2.5)
            paths[name] = path
            label = Text(name, color=colors[name], font_size=SMALL_FONT_SIZE)
            label.next_to(path.get_end(), RIGHT, buff=0.1).set_z_index(10)
            self.play(Create(path), FadeIn(label), run_time=NORMAL_ANIM)

        self.wait(PAUSE_MEDIUM)

        # ── Error table ─────────────────────────────────────────────────
        with self.voiceover(text="Here are the average errors on this real trajectory. The linear KF is worst, then EKF, then UKF and PF neck and neck. Differences depend on trajectory characteristics.") as tracker:
            true_pos = true_states[1:, :2]
            errors = {}
            for name, est in estimates.items():
                errors[name] = np.mean(np.linalg.norm(est - true_pos, axis=1))

            table_items = VGroup()
            for name in ["KF", "EKF", "UKF", "PF"]:
                row = VGroup(
                    Text(f"{name}:", color=colors[name], font_size=SMALL_FONT_SIZE),
                    Text(f"{errors[name]:.3f}", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
                ).arrange(RIGHT, buff=0.3)
                table_items.add(row)

            table = VGroup(
                Text("Avg Error", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
                *table_items,
            ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            table.to_corner(DL, buff=0.3).set_z_index(10)

            bg = SurroundingRectangle(table, color=SLATE, fill_color=BG_COLOR,
                                       fill_opacity=0.9, buff=0.15)
            self.play(FadeIn(bg), FadeIn(table), run_time=NORMAL_ANIM)

            # Theory-observation honesty note
            note = make_observation_note(
                "Differences depend on trajectory characteristics.\n"
                "Sharper turns amplify filter differences.",
            )
            self.play(FadeIn(note), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # ── Fade to decision guide ──────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)

        with self.voiceover(text="Which filter to use? Linear plus Gaussian: standard KF. Mildly nonlinear: EKF. Strongly nonlinear: UKF. Non-Gaussian or multimodal: Particle Filter.") as tracker:
            guide_title = Text("Which Filter to Use?", color=COLOR_TEXT,
                                font_size=TITLE_FONT_SIZE)
            guide_title.to_edge(UP, buff=0.4)
            self.play(Write(guide_title), run_time=NORMAL_ANIM)

            rules = VGroup(
                VGroup(
                    Text("Linear + Gaussian", color=COLOR_FILTER_KF,
                          font_size=BODY_FONT_SIZE),
                    MathTex(r"\rightarrow", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
                    Text("Standard KF", color=COLOR_FILTER_KF,
                          font_size=BODY_FONT_SIZE),
                ).arrange(RIGHT, buff=0.3),
                VGroup(
                    Text("Mildly nonlinear", color=COLOR_FILTER_EKF,
                          font_size=BODY_FONT_SIZE),
                    MathTex(r"\rightarrow", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
                    Text("EKF", color=COLOR_FILTER_EKF, font_size=BODY_FONT_SIZE),
                ).arrange(RIGHT, buff=0.3),
                VGroup(
                    Text("Strongly nonlinear", color=COLOR_FILTER_UKF,
                          font_size=BODY_FONT_SIZE),
                    MathTex(r"\rightarrow", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
                    Text("UKF", color=COLOR_FILTER_UKF, font_size=BODY_FONT_SIZE),
                ).arrange(RIGHT, buff=0.3),
                VGroup(
                    Text("Non-Gaussian / multimodal", color=COLOR_FILTER_PF,
                          font_size=BODY_FONT_SIZE),
                    MathTex(r"\rightarrow", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
                    Text("Particle Filter", color=COLOR_FILTER_PF,
                          font_size=BODY_FONT_SIZE),
                ).arrange(RIGHT, buff=0.3),
            ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
            rules.next_to(guide_title, DOWN, buff=LARGE_BUFF)

            for rule in rules:
                self.play(FadeIn(rule, shift=RIGHT * 0.3), run_time=NORMAL_ANIM)
                self.wait(PAUSE_SHORT)

            self.wait(PAUSE_MEDIUM)

        # ── Computational cost ────────────────────────────────────────
        with self.voiceover(text="Computational cost matters too: KF and EKF are fast, UKF is about three times more, and the particle filter scales with the number of particles. Now you understand the Kalman Filter family!") as tracker:
            cost_note = VGroup(
                Text("Computational cost:", color=SLATE, font_size=SMALL_FONT_SIZE),
                Text("KF/EKF: O(n³)  |  UKF: ~3× KF  |  PF: O(N·n), N = particles",
                     color=SLATE, font_size=SMALL_FONT_SIZE - 2),
            ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            cost_note.to_edge(DOWN, buff=0.3)
            self.play(FadeIn(cost_note), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

            # ── Closing ─────────────────────────────────────────────────────
            closing = Text(
                "Now you understand the Kalman Filter family!",
                color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
            )
            closing.to_edge(DOWN, buff=0.5)
            self.play(FadeIn(closing, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
