"""Part 9, Scene 2: Neural ODEs.

Data: Lorenz attractor trajectory from generate_lorenz_trajectory()

Visualizes the Lorenz attractor projected onto the x-z plane, then
shows the vector field concept: a neural network learns the velocity
field that governs the system. Introduces the ODE solver as a
differentiable black box.

Papers:
- Chen et al. (2018, NeurIPS) — Neural Ordinary Differential Equations
- Rubanova et al. (2019, NeurIPS) — Latent ODEs for Irregularly-Sampled Time Series
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.vector_field import VectorFieldPlot
from kalman_manim.mobjects.observation_note import make_observation_note
from kalman_manim.data.generators import generate_lorenz_trajectory
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneNeuralODEs(VoiceoverScene, MovingCameraScene):
    """Neural ODEs: learning continuous dynamics from data.

    Visual: Lorenz attractor trajectory (x-z projection), vector field,
    ODE solver block diagram.
    References: Chen et al. (2018), Rubanova et al. (2019).
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # -- Title --------------------------------------------------------
        with self.voiceover(
            text="Neural ODEs are the foundational idea: instead of specifying "
                 "the differential equation, let a neural network learn it. "
                 "Let's see this on the Lorenz attractor — one of the most "
                 "famous chaotic systems."
        ) as tracker:
            title = Text(
                "Neural ODEs",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # -- Generate Lorenz trajectory -----------------------------------
        lorenz_data = generate_lorenz_trajectory(
            sigma=10.0, rho=28.0, beta=8.0 / 3.0,
            dt=0.01, n_steps=3000, initial_state=np.array([1.0, 1.0, 1.0]),
        )
        states = lorenz_data["states"]  # (3001, 3)

        # Project to x-z plane
        x_vals = states[:, 0]
        z_vals = states[:, 2]

        # -- Lorenz attractor plot ----------------------------------------
        with self.voiceover(
            text="Here's the Lorenz attractor projected onto the x-z plane. "
                 "The trajectory is chaotic — tiny changes in initial "
                 "conditions lead to wildly different paths. This is exactly "
                 "the kind of system where hand-crafting a prediction model "
                 "is extremely difficult."
        ) as tracker:
            # Build axes
            axes = Axes(
                x_range=[-25, 25, 10],
                y_range=[0, 55, 10],
                x_length=6.0, y_length=4.0,
                axis_config={"color": CREAM, "include_tip": False},
            )
            axes.next_to(title, DOWN, buff=0.6)

            x_label = Text("x", font_size=16, color=SLATE)
            x_label.next_to(axes.x_axis, DOWN, buff=0.15)
            z_label = Text("z", font_size=16, color=SLATE)
            z_label.next_to(axes.y_axis, LEFT, buff=0.15)

            # Build trajectory line (subsample for performance)
            step = 3
            points = [
                axes.c2p(x_vals[i], z_vals[i])
                for i in range(0, len(x_vals), step)
            ]
            traj_line = VMobject()
            traj_line.set_points_smoothly(points[:500])
            traj_line.set_color(TEAL).set_stroke(width=1.5)

            lorenz_label = Text(
                "Lorenz Attractor (x-z projection)",
                color=SLATE, font_size=SMALL_FONT_SIZE,
            )
            lorenz_label.next_to(axes, DOWN, buff=0.3)

            self.play(
                FadeIn(axes), FadeIn(x_label), FadeIn(z_label),
                run_time=FAST_ANIM,
            )
            self.play(
                Create(traj_line, run_time=2.0),
                FadeIn(lorenz_label),
            )

        self.wait(PAUSE_MEDIUM)

        # -- Vector field concept -----------------------------------------
        with self.voiceover(
            text="A Neural ODE learns the velocity field: at every point in "
                 "state space, the network outputs the direction and speed "
                 "of change. The ODE solver then integrates this field to "
                 "produce trajectories."
        ) as tracker:
            self.play(
                FadeOut(axes), FadeOut(traj_line),
                FadeOut(x_label), FadeOut(z_label),
                FadeOut(lorenz_label),
                run_time=FAST_ANIM,
            )

            # Simple 2D vector field to illustrate the concept
            def spiral_field(x, y):
                """Spiral sink for illustration."""
                return (-0.3 * x - 0.5 * y, 0.5 * x - 0.3 * y)

            vf = VectorFieldPlot(
                func=spiral_field,
                x_range=(-3, 3),
                y_range=(-3, 3),
                n_arrows=8,
                width=4.5, height=4.5,
                color=TEAL,
            )
            vf.next_to(title, DOWN, buff=0.6).shift(LEFT * 2)

            vf_label = Text(
                "Learned velocity field: f_theta(x)",
                color=TEAL, font_size=SMALL_FONT_SIZE,
            )
            vf_label.next_to(vf, DOWN, buff=0.2)

            self.play(
                FadeIn(vf.axes), FadeIn(vf_label),
                run_time=FAST_ANIM,
            )
            self.play(FadeIn(vf.arrows_group, lag_ratio=0.05), run_time=NORMAL_ANIM)

        self.wait(PAUSE_SHORT)

        # -- ODE solver block diagram -------------------------------------
        with self.voiceover(
            text="The ODE solver is the key differentiable component. Given "
                 "an initial state and the learned vector field, it outputs "
                 "the state at any future time. Crucially, Chen et al. showed "
                 "that you can backpropagate through the solver using the "
                 "adjoint method — with constant memory cost."
        ) as tracker:
            # Block diagram on the right
            input_box = RoundedRectangle(
                corner_radius=0.1, width=1.8, height=0.7,
                color=COLOR_MEASUREMENT, fill_opacity=0.2,
            )
            input_label = Text("x(t_0)", color=COLOR_MEASUREMENT, font_size=18)
            input_label.move_to(input_box)
            input_grp = VGroup(input_box, input_label)

            net_box = RoundedRectangle(
                corner_radius=0.1, width=1.8, height=0.7,
                color=TEAL, fill_opacity=0.2,
            )
            net_label = Text("f_theta", color=TEAL, font_size=18)
            net_label.move_to(net_box)
            net_grp = VGroup(net_box, net_label)

            solver_box = RoundedRectangle(
                corner_radius=0.1, width=1.8, height=0.7,
                color=COLOR_HIGHLIGHT, fill_opacity=0.2,
            )
            solver_label = Text("ODE Solve", color=COLOR_HIGHLIGHT, font_size=18)
            solver_label.move_to(solver_box)
            solver_grp = VGroup(solver_box, solver_label)

            output_box = RoundedRectangle(
                corner_radius=0.1, width=1.8, height=0.7,
                color=COLOR_POSTERIOR, fill_opacity=0.2,
            )
            output_label = Text("x(t_1)", color=COLOR_POSTERIOR, font_size=18)
            output_label.move_to(output_box)
            output_grp = VGroup(output_box, output_label)

            blocks = VGroup(input_grp, net_grp, solver_grp, output_grp)
            blocks.arrange(DOWN, buff=0.3)
            blocks.next_to(title, DOWN, buff=0.5).shift(RIGHT * 2.5)

            arrows = VGroup(
                Arrow(input_grp.get_bottom(), net_grp.get_top(),
                      color=SLATE, stroke_width=2, buff=0.1),
                Arrow(net_grp.get_bottom(), solver_grp.get_top(),
                      color=SLATE, stroke_width=2, buff=0.1),
                Arrow(solver_grp.get_bottom(), output_grp.get_top(),
                      color=SLATE, stroke_width=2, buff=0.1),
            )

            adjoint_label = Text(
                "Backprop via adjoint method\n(O(1) memory)",
                color=SLATE, font_size=14,
            )
            adjoint_label.next_to(blocks, RIGHT, buff=0.3)

            self.play(
                FadeIn(blocks, shift=DOWN * 0.2),
                Create(arrows),
                FadeIn(adjoint_label),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # -- Latent ODEs --------------------------------------------------
        with self.voiceover(
            text="Rubanova et al. extended this to latent ODEs: the dynamics "
                 "happen in a learned latent space, with an encoder-decoder "
                 "wrapping the ODE. This handles irregularly sampled data — "
                 "exactly the scenario in GPS or LBS tracking where pings "
                 "arrive at uneven intervals."
        ) as tracker:
            latent_text = Text(
                "Latent ODE: encode -> ODE solve in latent space -> decode",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            )
            latent_text.to_edge(DOWN, buff=0.5)

            irregular_text = Text(
                "Handles irregular timestamps (GPS, LBS pings)",
                color=SLATE, font_size=SMALL_FONT_SIZE,
            )
            irregular_text.next_to(latent_text, DOWN, buff=0.2)

            self.play(
                FadeIn(latent_text, shift=UP * 0.2),
                FadeIn(irregular_text),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        note = make_observation_note(
            "Chen et al. (2018, NeurIPS Best Paper): Neural ODEs\n"
            "Rubanova et al. (2019): Latent ODEs for irregular time series"
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
