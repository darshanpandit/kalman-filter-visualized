"""Part 4, Scene 3: Particle Filter Demo

Applies PF to a pedestrian trajectory. Shows the particle cloud
converging around the true path.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.particle_cloud import ParticleCloud
from kalman_manim.data.generators import generate_pedestrian_trajectory
from filters.particle import ParticleFilter
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


def _pf_transition(x, u, noise):
    dt = 0.5
    return np.array([
        x[0] + x[2] * dt + noise[0],
        x[1] + x[3] * dt + noise[1],
        x[2] + noise[2],
        x[3] + noise[3],
    ])


def _pf_measurement(x):
    return x[:2]


class ScenePFDemo(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        with self.voiceover(text="Here's the particle filter tracking a pedestrian with 200 particles. The teal cloud represents the state distribution.") as tracker:
            title = Text("Particle Filter in Action", color=COLOR_TEXT,
                          font_size=TITLE_FONT_SIZE)
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

            # ── Generate data ───────────────────────────────────────────────
            data = generate_pedestrian_trajectory(
                n_steps=30, dt=0.5, speed=0.6,
                process_noise_std=0.1, measurement_noise_std=0.5,
                turn_probability=0.1, seed=7,
            )
            true_states = data["true_states"]
            measurements = data["measurements"]

            # ── Run PF ──────────────────────────────────────────────────────
            Q = np.diag([0.02, 0.02, 0.05, 0.05])
            R = 0.25 * np.eye(2)
            pf = ParticleFilter(
                f=_pf_transition, h=_pf_measurement,
                Q=Q, R=R, n_particles=200,
                x0=np.array([0, 0, 0.6, 0]),
                P0=np.diag([0.5, 0.5, 0.3, 0.3]),
                seed=42,
            )
            results = pf.run(measurements)

            # Scale
            all_pos = np.vstack([true_states[:, :2], measurements])
            scale = 4.0 / max(np.ptp(all_pos[:, 0]), np.ptp(all_pos[:, 1]), 1)
            ctr = all_pos.mean(axis=0)

            def to_s(xy):
                s = (xy - ctr) * scale
                return np.array([s[0], s[1], 0])

            # ── Legend ──────────────────────────────────────────────────────
            legend = VGroup(
                VGroup(Dot(color=COLOR_TRUE_PATH, radius=0.04),
                       Text("True", color=COLOR_TRUE_PATH, font_size=16)).arrange(RIGHT, buff=0.1),
                VGroup(Dot(color=COLOR_MEASUREMENT, radius=0.04),
                       Text("Measurement", color=COLOR_MEASUREMENT, font_size=16)).arrange(RIGHT, buff=0.1),
                VGroup(Dot(color=COLOR_PROCESS_NOISE, radius=0.04),
                       Text("Particles", color=COLOR_PROCESS_NOISE, font_size=16)).arrange(RIGHT, buff=0.1),
                VGroup(Dot(color=COLOR_POSTERIOR, radius=0.04),
                       Text("PF estimate", color=COLOR_POSTERIOR, font_size=16)).arrange(RIGHT, buff=0.1),
            ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            legend.to_corner(UL, buff=0.3).set_z_index(10)
            self.play(FadeIn(legend), run_time=FAST_ANIM)

        # ── Animate step by step ────────────────────────────────────────
        with self.voiceover(text="Each step, the particle cloud moves during prediction, then concentrates when a measurement arrives. Watch the cloud breathe, spreading and converging, spreading and converging.") as tracker:
            est_trail_pts = []
            true_trail_pts = [to_s(true_states[0, :2])]
            true_trail = VMobject(color=COLOR_TRUE_PATH, stroke_width=1.5, stroke_opacity=0.6)
            est_trail = VMobject(color=COLOR_POSTERIOR, stroke_width=2.5)
            prev_cloud_mob = None

            self.add(true_trail, est_trail)

        for k in range(len(measurements)):
            # True position
            true_pt = to_s(true_states[k + 1, :2])
            true_trail_pts.append(true_pt)

            # Measurement
            meas_pt = to_s(measurements[k])
            meas_dot = Dot(meas_pt, radius=MEASUREMENT_DOT_RADIUS,
                           color=COLOR_MEASUREMENT, fill_opacity=0.5)

            # Particles from results
            particles_2d = results["particles_history"][k][:, :2]
            particles_scaled = (particles_2d - ctr) * scale
            weights = results["weights_history"][k]

            cloud_mob = ParticleCloud(
                particles=particles_scaled,
                weights=weights,
                color=COLOR_PROCESS_NOISE,
                max_particles_shown=150,
            )

            # Estimate
            est_pt = to_s(results["x_estimates"][k][:2])
            est_trail_pts.append(est_pt)

            # Update trails
            anims = [FadeIn(meas_dot, scale=1.2)]

            if prev_cloud_mob is not None:
                anims.append(FadeOut(prev_cloud_mob))
            anims.append(FadeIn(cloud_mob))

            if len(true_trail_pts) >= 2:
                new_true = VMobject(color=COLOR_TRUE_PATH, stroke_width=1.5, stroke_opacity=0.6)
                new_true.set_points_smoothly(true_trail_pts)
                anims.append(Transform(true_trail, new_true))

            if len(est_trail_pts) >= 2:
                new_est = VMobject(color=COLOR_POSTERIOR, stroke_width=2.5)
                new_est.set_points_smoothly(est_trail_pts)
                anims.append(Transform(est_trail, new_est))

            self.play(*anims, run_time=0.35)
            prev_cloud_mob = cloud_mob

        # Zoom out
        with self.voiceover(text="From noisy measurements, the particle filter has reconstructed the trajectory. No Gaussian assumption needed, particles can represent any distribution.") as tracker:
            self.play(
                self.camera.frame.animate.move_to(ORIGIN).set(width=14),
                run_time=SLOW_ANIM,
            )

            result = Text(
                "Particles converge to the true trajectory!",
                color=COLOR_POSTERIOR, font_size=BODY_FONT_SIZE,
            )
            result.to_edge(DOWN, buff=0.3).set_z_index(10)
            self.play(FadeIn(result), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
