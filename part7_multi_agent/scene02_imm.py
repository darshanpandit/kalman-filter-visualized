"""Part 7, Scene 2: IMM Filter — multiple models mixing.

Data: Synthetic mode-switching trajectory

The IMM filter runs CV and CT models in parallel, with mode
probabilities adapting over time.

Papers:
- Blom & Bar-Shalom (1988)
- Li & Jilkov (2005) survey
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.mode_probability import ModeProbabilityBar
from kalman_manim.mobjects.observation_note import make_observation_note
from kalman_manim.data.generators import generate_mode_switching_trajectory
from filters.kalman import KalmanFilter
from filters.imm import IMMFilter
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneIMM(VoiceoverScene, MovingCameraScene):
    """IMM filter: two models mixing based on data.

    Visual: Trajectory + mode probability bar evolving.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # Generate data
        data = generate_mode_switching_trajectory(n_steps=80, seed=42)
        true_states = data["true_states"]
        measurements = data["measurements"]
        true_modes = data["true_modes"]
        dt = data["dt"]

        # Set up IMM with CV and CT sub-filters
        x0 = true_states[0]
        F_cv = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                          [0, 0, 1, 0], [0, 0, 0, 1]])
        omega = 0.15
        cos_w, sin_w = np.cos(omega * dt), np.sin(omega * dt)
        F_ct = np.array([
            [1, 0, sin_w / omega, -(1 - cos_w) / omega],
            [0, 1, (1 - cos_w) / omega, sin_w / omega],
            [0, 0, cos_w, -sin_w],
            [0, 0, sin_w, cos_w],
        ])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        Q_cv = 0.01 * np.eye(4)
        Q_ct = 0.05 * np.eye(4)
        R = 0.25 * np.eye(2)

        cv_filter = KalmanFilter(F=F_cv, H=H, Q=Q_cv, R=R, x0=x0.copy(), P0=np.eye(4))
        ct_filter = KalmanFilter(F=F_ct, H=H, Q=Q_ct, R=R, x0=x0.copy(), P0=np.eye(4))
        Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMFilter(filters=[cv_filter, ct_filter], transition_matrix=Pi)

        # Run IMM
        results = imm.run(measurements)

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="The Interacting Multiple Model filter runs two Kalman "
                 "filters in parallel — one for straight-line motion, one "
                 "for turns — and mixes their outputs based on which model "
                 "fits the data better."
        ) as tracker:
            title = Text(
                "IMM: Multiple Models",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Trajectory with mode coloring ──────────────────────────────
        with self.voiceover(
            text="This pedestrian alternates between walking straight and "
                 "turning. The color shows which model is active — red for "
                 "constant velocity, purple for coordinated turn."
        ) as tracker:
            # Build axes
            pos = true_states[:, :2]
            x_min, x_max = pos[:, 0].min() - 1, pos[:, 0].max() + 1
            y_min, y_max = pos[:, 1].min() - 1, pos[:, 1].max() + 1

            axes = Axes(
                x_range=[x_min, x_max, (x_max - x_min) / 5],
                y_range=[y_min, y_max, (y_max - y_min) / 5],
                x_length=7, y_length=3.5,
                axis_config={"color": CREAM, "include_tip": False},
            )
            axes.shift(DOWN * 0.3)

            # True path colored by mode
            for k in range(len(true_modes) - 1):
                color = COLOR_FILTER_KF if true_modes[k] == 0 else COLOR_FILTER_IMM
                p1 = axes.c2p(*pos[k + 1])
                p2 = axes.c2p(*pos[k + 2])
                seg = Line(p1, p2, color=color, stroke_width=2)
                self.add(seg)

            self.play(FadeIn(axes), run_time=FAST_ANIM)

        self.wait(PAUSE_SHORT)

        # ── Mode probability evolution ─────────────────────────────────
        with self.voiceover(
            text="Watch the mode probability bar: during straight segments, "
                 "the CV model dominates. When the pedestrian starts turning, "
                 "the CT model probability surges. The IMM adapts within "
                 "about five steps."
        ) as tracker:
            bar = ModeProbabilityBar(
                model_names=["CV", "CT"],
                colors=[COLOR_FILTER_KF, COLOR_FILTER_IMM],
                width=5.0,
            )
            bar.to_edge(DOWN, buff=0.5)

            self.play(FadeIn(bar), run_time=FAST_ANIM)

            # Animate probability changes at key moments
            key_steps = [10, 25, 45, 65]
            for step in key_steps:
                mu = results["model_probabilities"][step]
                new_bar = ModeProbabilityBar(
                    model_names=["CV", "CT"],
                    colors=[COLOR_FILTER_KF, COLOR_FILTER_IMM],
                    width=5.0,
                )
                new_bar.set_probabilities(mu)
                new_bar.to_edge(DOWN, buff=0.5)

                step_label = Text(
                    f"Step {step}: CV={mu[0]:.2f}, CT={mu[1]:.2f}",
                    color=SLATE, font_size=14,
                )
                step_label.next_to(new_bar, DOWN, buff=0.15)

                self.play(
                    Transform(bar, new_bar),
                    FadeIn(step_label),
                    run_time=0.7,
                )
                self.wait(0.3)
                self.play(FadeOut(step_label), run_time=0.2)

        self.wait(PAUSE_MEDIUM)

        note = make_observation_note(
            "IMM: Blom & Bar-Shalom (1988)\n"
            "Converges to correct model within ~5 steps"
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
