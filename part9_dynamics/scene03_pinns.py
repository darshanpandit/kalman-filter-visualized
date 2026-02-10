"""Part 9, Scene 3: Physics-Informed Neural Networks (PINNs).

Data: Pendulum trajectory from generate_pendulum_trajectory()

Shows the PINN loss decomposition: data loss + physics residual.
Demonstrates extrapolation advantage on pendulum dynamics —
a vanilla NN only interpolates, but a PINN respects Newton's laws
and extrapolates beyond the training window.

Papers:
- Raissi et al. (2019, J. Computational Physics) — PINNs
- Karniadakis et al. (2021, Nature Reviews Physics) — Physics-informed ML
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.observation_note import make_observation_note
from kalman_manim.data.generators import generate_pendulum_trajectory
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class ScenePINNs(VoiceoverScene, MovingCameraScene):
    """Physics-Informed Neural Networks: embedding equations into the loss.

    Visual: Loss decomposition, pendulum analytical vs learned trajectories,
    extrapolation comparison.
    References: Raissi et al. (2019), Karniadakis et al. (2021).
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # -- Title --------------------------------------------------------
        with self.voiceover(
            text="Neural ODEs learn dynamics purely from data. But what if "
                 "you know some of the physics? Physics-Informed Neural "
                 "Networks — PINNs — embed the governing equations directly "
                 "into the loss function."
        ) as tracker:
            title = Text(
                "Physics-Informed Neural Networks",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # -- Loss decomposition -------------------------------------------
        with self.voiceover(
            text="The PINN loss has two parts. First, a data loss: how well "
                 "does the network fit observed data points? Second, a physics "
                 "residual: does the network's output satisfy the governing "
                 "differential equation? The physics term acts as a powerful "
                 "regularizer."
        ) as tracker:
            # Data loss box
            data_box = RoundedRectangle(
                corner_radius=0.1, width=3.5, height=1.0,
                color=COLOR_MEASUREMENT, fill_opacity=0.2,
            )
            data_label = Text(
                "Data Loss\nsum(y_pred - y_obs)^2",
                color=COLOR_MEASUREMENT, font_size=18,
            )
            data_label.move_to(data_box)
            data_grp = VGroup(data_box, data_label)

            # Physics residual box
            phys_box = RoundedRectangle(
                corner_radius=0.1, width=3.5, height=1.0,
                color=TEAL, fill_opacity=0.2,
            )
            phys_label = Text(
                "Physics Residual\nsum(d2y/dt2 + g/L sin(y))^2",
                color=TEAL, font_size=18,
            )
            phys_label.move_to(phys_box)
            phys_grp = VGroup(phys_box, phys_label)

            # Plus sign and total
            plus = Text("+", color=CREAM, font_size=HEADING_FONT_SIZE)

            total_box = RoundedRectangle(
                corner_radius=0.1, width=3.0, height=0.8,
                color=COLOR_HIGHLIGHT, fill_opacity=0.2,
            )
            total_label = Text(
                "PINN Loss", color=COLOR_HIGHLIGHT, font_size=20,
            )
            total_label.move_to(total_box)
            total_grp = VGroup(total_box, total_label)

            loss_row = VGroup(data_grp, plus, phys_grp)
            loss_row.arrange(RIGHT, buff=0.3)
            loss_row.next_to(title, DOWN, buff=0.7)

            equals = Text("=", color=CREAM, font_size=HEADING_FONT_SIZE)
            total_grp.next_to(loss_row, DOWN, buff=0.3)

            self.play(
                FadeIn(data_grp, shift=LEFT * 0.3),
                FadeIn(plus),
                FadeIn(phys_grp, shift=RIGHT * 0.3),
                run_time=NORMAL_ANIM,
            )
            self.play(FadeIn(total_grp, shift=UP * 0.2), run_time=FAST_ANIM)

        self.wait(PAUSE_MEDIUM)

        # -- Pendulum trajectory ------------------------------------------
        with self.voiceover(
            text="Let's see this on a pendulum. The analytical solution "
                 "oscillates periodically. We'll train on just the first "
                 "half — the training window — and see how each approach "
                 "extrapolates beyond it."
        ) as tracker:
            self.play(
                FadeOut(data_grp), FadeOut(plus),
                FadeOut(phys_grp), FadeOut(total_grp),
                run_time=FAST_ANIM,
            )

            # Generate pendulum data
            pend_data = generate_pendulum_trajectory(
                length=1.0, gravity=9.81, theta0=1.0, omega0=0.0,
                dt=0.01, n_steps=1000,
            )
            pend_states = pend_data["states"]  # (1001, 2)
            pend_dt = pend_data["dt"]
            t_vals = np.arange(len(pend_states)) * pend_dt
            theta_vals = pend_states[:, 0]

            # Axes
            axes = Axes(
                x_range=[0, 10, 2],
                y_range=[-1.5, 1.5, 0.5],
                x_length=8.0, y_length=3.5,
                axis_config={"color": CREAM, "include_tip": False},
            )
            axes.next_to(title, DOWN, buff=0.7)

            t_label = Text("time (s)", font_size=14, color=SLATE)
            t_label.next_to(axes.x_axis, DOWN, buff=0.15)
            theta_label = Text("theta (rad)", font_size=14, color=SLATE)
            theta_label.next_to(axes.y_axis, LEFT, buff=0.15)

            # Analytical curve (ground truth)
            analytical_points = [
                axes.c2p(t_vals[i], theta_vals[i])
                for i in range(0, len(t_vals), 5)
            ]
            analytical_line = VMobject()
            analytical_line.set_points_smoothly(analytical_points[:200])
            analytical_line.set_color(CREAM).set_stroke(width=2)

            # Training window shading
            train_end = 5.0  # seconds
            train_region = Rectangle(
                width=axes.c2p(train_end, 0)[0] - axes.c2p(0, 0)[0],
                height=3.5,
                fill_color=DARK_SLATE, fill_opacity=0.3,
                stroke_width=0,
            )
            train_region.move_to(axes.c2p(train_end / 2, 0))

            train_label = Text(
                "Training window", color=SLATE, font_size=14,
            )
            train_label.next_to(train_region, UP, buff=0.1)

            self.play(
                FadeIn(axes), FadeIn(t_label), FadeIn(theta_label),
                FadeIn(train_region), FadeIn(train_label),
                run_time=FAST_ANIM,
            )
            self.play(Create(analytical_line), run_time=NORMAL_ANIM)

        self.wait(PAUSE_SHORT)

        # -- Vanilla NN vs PINN extrapolation -----------------------------
        with self.voiceover(
            text="A vanilla neural network — shown in red — fits the training "
                 "data well, but diverges outside the training window. The "
                 "PINN — in gold — stays close to the true solution because "
                 "the physics residual penalizes any trajectory that violates "
                 "Newton's second law."
        ) as tracker:
            # Simulated vanilla NN: matches in training, diverges outside
            # (decaying oscillation to show extrapolation failure)
            rng = np.random.default_rng(42)
            vanilla_theta = theta_vals.copy()
            # After training window, add growing error
            train_idx = int(train_end / pend_dt)
            for i in range(train_idx, len(vanilla_theta)):
                decay = np.exp(-0.3 * (t_vals[i] - train_end))
                vanilla_theta[i] = theta_vals[i] * decay + 0.15 * rng.normal()

            vanilla_points = [
                axes.c2p(t_vals[i], vanilla_theta[i])
                for i in range(0, len(t_vals), 5)
            ]
            vanilla_line = VMobject()
            vanilla_line.set_points_smoothly(vanilla_points[:200])
            vanilla_line.set_color(SWISS_RED).set_stroke(width=2, opacity=0.8)

            # Simulated PINN: stays close throughout
            pinn_theta = theta_vals.copy()
            for i in range(train_idx, len(pinn_theta)):
                pinn_theta[i] = theta_vals[i] + 0.02 * rng.normal()

            pinn_points = [
                axes.c2p(t_vals[i], pinn_theta[i])
                for i in range(0, len(t_vals), 5)
            ]
            pinn_line = VMobject()
            pinn_line.set_points_smoothly(pinn_points[:200])
            pinn_line.set_color(COLOR_HIGHLIGHT).set_stroke(width=2, opacity=0.8)

            # Legend
            legend_items = VGroup(
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=CREAM, stroke_width=2),
                    Text("Analytical", color=CREAM, font_size=14),
                ).arrange(RIGHT, buff=0.15),
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=SWISS_RED, stroke_width=2),
                    Text("Vanilla NN", color=SWISS_RED, font_size=14),
                ).arrange(RIGHT, buff=0.15),
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=COLOR_HIGHLIGHT, stroke_width=2),
                    Text("PINN", color=COLOR_HIGHLIGHT, font_size=14),
                ).arrange(RIGHT, buff=0.15),
            )
            legend_items.arrange(RIGHT, buff=0.5)
            legend_items.next_to(axes, DOWN, buff=0.5)

            self.play(
                Create(vanilla_line), Create(pinn_line),
                FadeIn(legend_items),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # -- Extrapolation highlight --------------------------------------
        with self.voiceover(
            text="This is the extrapolation advantage. By encoding known "
                 "physics into the loss, the PINN generalizes far beyond "
                 "its training data. For filtering applications, this means "
                 "more robust predictions in regimes where you have limited "
                 "observations."
        ) as tracker:
            # Highlight extrapolation region
            extrap_region = Rectangle(
                width=axes.c2p(10, 0)[0] - axes.c2p(train_end, 0)[0],
                height=3.5,
                color=COLOR_HIGHLIGHT, fill_opacity=0.08,
                stroke_color=COLOR_HIGHLIGHT, stroke_width=1,
            )
            extrap_center_x = (axes.c2p(train_end, 0)[0] + axes.c2p(10, 0)[0]) / 2
            extrap_region.move_to(
                np.array([extrap_center_x, axes.c2p(0, 0)[1], 0])
            )

            extrap_label = Text(
                "Extrapolation region",
                color=COLOR_HIGHLIGHT, font_size=14,
            )
            extrap_label.next_to(extrap_region, UP, buff=0.1)

            self.play(
                FadeIn(extrap_region), FadeIn(extrap_label),
                run_time=FAST_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # -- PINN for filtering -------------------------------------------
        with self.voiceover(
            text="In the filtering context, PINNs let you combine sparse "
                 "sensor data with known differential equations. You don't "
                 "need a perfect model — just a partial one. The network "
                 "fills in the gaps."
        ) as tracker:
            filter_text = Text(
                "PINN for filtering: sparse data + partial physics = robust prediction",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            )
            filter_text.to_edge(DOWN, buff=0.3)
            self.play(FadeIn(filter_text, shift=UP * 0.2), run_time=NORMAL_ANIM)

        note = make_observation_note(
            "Raissi et al. (2019, J. Comp. Physics): PINNs\n"
            "Loss = data fit + physics residual, enables extrapolation"
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
