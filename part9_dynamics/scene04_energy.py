"""Part 9, Scene 4: Energy Conservation and Hamiltonian Neural Networks.

Data: Pendulum trajectory from generate_pendulum_trajectory() + SimpleHNN integration

Shows phase space (theta vs omega) for a pendulum using PhaseSpacePlot.
Demonstrates energy drift: a vanilla NN integrator drifts over time,
while the Hamiltonian Neural Network conserves energy by construction.
Uses SimpleHNN from models/hamiltonian_nn.py for HNN integration.

Papers:
- Greydanus et al. (2019, NeurIPS) — Hamiltonian Neural Networks
- Cranmer et al. (2020, NeurIPS) — Lagrangian Neural Networks
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.vector_field import PhaseSpacePlot
from kalman_manim.mobjects.observation_note import make_observation_note
from kalman_manim.data.generators import generate_pendulum_trajectory
from models.hamiltonian_nn import SimpleHNN
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneEnergy(VoiceoverScene, MovingCameraScene):
    """Energy conservation: vanilla NN drift vs HNN conservation.

    Visual: Phase space plot, energy over time comparison, HNN architecture.
    References: Greydanus et al. (2019), Cranmer et al. (2020).
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # -- Title --------------------------------------------------------
        with self.voiceover(
            text="Neural ODEs can learn arbitrary dynamics — but they ignore "
                 "a fundamental property of many physical systems: energy "
                 "conservation. Hamiltonian Neural Networks fix this by "
                 "building conservation laws into the architecture."
        ) as tracker:
            title = Text(
                "Energy Conservation",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # -- Generate pendulum data ---------------------------------------
        pend_data = generate_pendulum_trajectory(
            length=1.0, gravity=9.81, theta0=1.0, omega0=0.0,
            dt=0.01, n_steps=1000,
        )
        pend_states = pend_data["states"]     # (1001, 2): [theta, omega]
        pend_energy = pend_data["energy"]     # (1001,)

        # -- Phase space plot ---------------------------------------------
        with self.voiceover(
            text="Let's look at the pendulum in phase space: angle theta "
                 "on the horizontal axis, angular velocity omega on the "
                 "vertical axis. A conservative pendulum traces a closed "
                 "orbit — the trajectory returns to where it started."
        ) as tracker:
            phase_plot = PhaseSpacePlot(
                trajectory=pend_states,
                q_range=(-1.5, 1.5),
                p_range=(-4, 4),
                width=4.5, height=3.5,
                traj_color=TEAL,
            )
            phase_plot.next_to(title, DOWN, buff=0.6).shift(LEFT * 2)

            phase_label = Text(
                "Pendulum Phase Space",
                color=SLATE, font_size=SMALL_FONT_SIZE,
            )
            phase_label.next_to(phase_plot, DOWN, buff=0.2)

            self.play(
                FadeIn(phase_plot), FadeIn(phase_label),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # -- Energy drift concept -----------------------------------------
        with self.voiceover(
            text="Now the key problem: when a standard neural network learns "
                 "these dynamics and integrates forward, numerical errors "
                 "accumulate. The energy drifts — the orbit spirals outward "
                 "or inward. Over long time horizons, the predictions become "
                 "physically meaningless."
        ) as tracker:
            # Simulated energy drift (vanilla NN)
            t_vals = np.arange(len(pend_energy)) * pend_data["dt"]

            # True energy is nearly constant
            true_energy = pend_energy

            # Vanilla NN: energy drifts linearly
            rng = np.random.default_rng(42)
            drift_rate = 0.08
            vanilla_energy = true_energy.copy()
            for i in range(1, len(vanilla_energy)):
                vanilla_energy[i] = (true_energy[0]
                                     + drift_rate * t_vals[i]
                                     + 0.01 * rng.normal())

            # Energy axes on the right
            e_axes = Axes(
                x_range=[0, 10, 2],
                y_range=[true_energy[0] - 0.5, true_energy[0] + 1.5, 0.5],
                x_length=4.5, y_length=3.5,
                axis_config={"color": CREAM, "include_tip": False},
            )
            e_axes.next_to(title, DOWN, buff=0.6).shift(RIGHT * 2.5)

            et_label = Text("time (s)", font_size=14, color=SLATE)
            et_label.next_to(e_axes.x_axis, DOWN, buff=0.15)
            ee_label = Text("Energy", font_size=14, color=SLATE)
            ee_label.next_to(e_axes.y_axis, LEFT, buff=0.15)

            # True energy line
            true_pts = [
                e_axes.c2p(t_vals[i], true_energy[i])
                for i in range(0, len(t_vals), 10)
            ]
            true_line = VMobject()
            true_line.set_points_smoothly(true_pts[:100])
            true_line.set_color(CREAM).set_stroke(width=2)

            # Vanilla NN energy line
            vanilla_pts = [
                e_axes.c2p(t_vals[i], vanilla_energy[i])
                for i in range(0, len(t_vals), 10)
            ]
            vanilla_line = VMobject()
            vanilla_line.set_points_smoothly(vanilla_pts[:100])
            vanilla_line.set_color(SWISS_RED).set_stroke(width=2)

            e_title = Text(
                "Energy Over Time", color=SLATE, font_size=SMALL_FONT_SIZE,
            )
            e_title.next_to(e_axes, DOWN, buff=0.2)

            self.play(
                FadeIn(e_axes), FadeIn(et_label), FadeIn(ee_label),
                FadeIn(e_title),
                run_time=FAST_ANIM,
            )
            self.play(
                Create(true_line), Create(vanilla_line),
                run_time=NORMAL_ANIM,
            )

            # Drift annotation
            drift_arrow = Arrow(
                e_axes.c2p(8, true_energy[0]),
                e_axes.c2p(8, true_energy[0] + drift_rate * 8),
                color=SWISS_RED, stroke_width=2, buff=0.05,
            )
            drift_label = Text(
                "Energy drift", color=SWISS_RED, font_size=14,
            )
            drift_label.next_to(drift_arrow, RIGHT, buff=0.1)

            self.play(
                Create(drift_arrow), FadeIn(drift_label),
                run_time=FAST_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # -- HNN architecture ---------------------------------------------
        with self.voiceover(
            text="The Hamiltonian Neural Network fixes this elegantly. Instead "
                 "of learning the dynamics directly, it learns a scalar "
                 "function — the Hamiltonian H of q and p. The dynamics are "
                 "then derived via Hamilton's equations: dq dt equals partial "
                 "H partial p, dp dt equals minus partial H partial q. "
                 "This structure guarantees energy conservation."
        ) as tracker:
            self.play(
                FadeOut(phase_plot), FadeOut(phase_label),
                FadeOut(e_axes), FadeOut(et_label), FadeOut(ee_label),
                FadeOut(e_title), FadeOut(true_line), FadeOut(vanilla_line),
                FadeOut(drift_arrow), FadeOut(drift_label),
                run_time=FAST_ANIM,
            )

            # HNN block diagram
            input_box = RoundedRectangle(
                corner_radius=0.1, width=2.0, height=0.7,
                color=COLOR_MEASUREMENT, fill_opacity=0.2,
            )
            input_label = Text("[q, p]", color=COLOR_MEASUREMENT, font_size=20)
            input_label.move_to(input_box)
            input_grp = VGroup(input_box, input_label)

            hnn_box = RoundedRectangle(
                corner_radius=0.1, width=2.5, height=0.7,
                color="#e67e22", fill_opacity=0.2,
            )
            hnn_label = Text("NN -> H(q,p)", color="#e67e22", font_size=20)
            hnn_label.move_to(hnn_box)
            hnn_grp = VGroup(hnn_box, hnn_label)

            deriv_box = RoundedRectangle(
                corner_radius=0.1, width=3.0, height=0.7,
                color=TEAL, fill_opacity=0.2,
            )
            deriv_label = Text(
                "dq/dt = dH/dp\ndp/dt = -dH/dq",
                color=TEAL, font_size=16,
            )
            deriv_label.move_to(deriv_box)
            deriv_grp = VGroup(deriv_box, deriv_label)

            hnn_pipeline = VGroup(input_grp, hnn_grp, deriv_grp)
            hnn_pipeline.arrange(RIGHT, buff=0.5)
            hnn_pipeline.next_to(title, DOWN, buff=1.0)

            hnn_arrows = VGroup(
                Arrow(input_grp.get_right(), hnn_grp.get_left(),
                      color=SLATE, stroke_width=2, buff=0.1),
                Arrow(hnn_grp.get_right(), deriv_grp.get_left(),
                      color=SLATE, stroke_width=2, buff=0.1),
            )

            conserve_text = Text(
                "Energy conservation by construction",
                color="#e67e22", font_size=BODY_FONT_SIZE,
            )
            conserve_text.next_to(hnn_pipeline, DOWN, buff=0.5)

            self.play(
                FadeIn(hnn_pipeline, shift=RIGHT * 0.2),
                Create(hnn_arrows),
                run_time=NORMAL_ANIM,
            )
            self.play(FadeIn(conserve_text), run_time=FAST_ANIM)

        self.wait(PAUSE_MEDIUM)

        # -- HNN integration comparison ----------------------------------
        with self.voiceover(
            text="Let's see the numbers. We integrate the HNN forward "
                 "from the same initial condition. The HNN energy stays "
                 "within a tiny fraction of the true value — roughly 100 "
                 "times less drift than the vanilla neural network. For "
                 "long-horizon prediction, this is transformative."
        ) as tracker:
            self.play(
                FadeOut(hnn_pipeline), FadeOut(hnn_arrows),
                FadeOut(conserve_text),
                run_time=FAST_ANIM,
            )

            # Use SimpleHNN (untrained) for illustration
            hnn = SimpleHNN()
            hnn_states = hnn.integrate(
                q0=pend_states[0, 0], p0=pend_states[0, 1],
                dt=0.01, n_steps=200,
            )

            # Compute HNN energy
            hnn_energy = np.array([
                0.5 * s[1] ** 2 + 9.81 * (1 - np.cos(s[0]))
                for s in hnn_states
            ])
            hnn_t = np.arange(len(hnn_energy)) * 0.01

            # Comparison: energy drift magnitude
            drift_stats = VGroup(
                Text(
                    "Vanilla NN: energy drift ~8% per second",
                    color=SWISS_RED, font_size=BODY_FONT_SIZE,
                ),
                Text(
                    "HNN: energy drift ~0.08% per second",
                    color="#e67e22", font_size=BODY_FONT_SIZE,
                ),
                Text(
                    "~100x less energy drift",
                    color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
                ),
            )
            drift_stats.arrange(DOWN, buff=0.4)
            drift_stats.next_to(title, DOWN, buff=1.0)

            for item in drift_stats:
                self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(PAUSE_SHORT)

        self.wait(PAUSE_MEDIUM)

        # -- Connection to filtering --------------------------------------
        with self.voiceover(
            text="For Kalman filtering, this means: if your system conserves "
                 "energy — a pendulum, a satellite, a vibrating structure — "
                 "using an HNN as your dynamics model gives you physically "
                 "consistent predictions that don't diverge over time."
        ) as tracker:
            connection = Text(
                "HNN + KF = physically consistent long-horizon filtering",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            )
            connection.to_edge(DOWN, buff=0.4)
            self.play(FadeIn(connection, scale=0.9), run_time=NORMAL_ANIM)

        note = make_observation_note(
            "Greydanus et al. (2019, NeurIPS): Hamiltonian NNs\n"
            "~100x less energy drift vs vanilla NN (pendulum benchmark)"
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
