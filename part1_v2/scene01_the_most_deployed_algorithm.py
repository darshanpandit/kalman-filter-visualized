"""Scene 1: The Most Deployed Algorithm — 'You use it. You just don't know it.'

Two-voice format (Narrator + Skeptic) using Azure TTS with style/prosody variations.
Data: real-world (ETH eth, pedestrian #171)

Shows noisy GPS pings, reveals true path, teases Kalman-filtered result.

References:
  Pellegrini et al. (2009) — ETH Zurich pedestrian dataset
  Kalman (1960) — A new approach to linear filtering and prediction problems

Requires: pip install "manim-voiceover[azure]"
          Set AZURE_SUBSCRIPTION_KEY and AZURE_SERVICE_REGION in .env
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.azure import AzureService
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.data.loader import load_eth_trajectory
from kalman_manim.mobjects.observation_note import make_observation_note
from filters.kalman import KalmanFilter


class SceneTheMostDeployedAlgorithm(VoiceoverScene, MovingCameraScene):
    """The Most Deployed Algorithm — 'You use it. You just don't know it.'"""

    def construct(self):
        # ── Voice setup ───────────────────────────────────────────────
        # One AzureService per (voice, style) combination
        narrator = AzureService(
            voice="en-US-JennyNeural", style="chat",
        )
        narrator_newscast = AzureService(
            voice="en-US-JennyNeural", style="newscast",
        )
        narrator_hopeful = AzureService(
            voice="en-US-JennyNeural", style="hopeful",
        )
        skeptic = AzureService(
            voice="en-US-TonyNeural", style="friendly",
        )
        self.set_speech_service(narrator)
        self.camera.background_color = BG_COLOR

        # ── Data setup ────────────────────────────────────────────────
        data = load_eth_trajectory(
            sequence="eth", pedestrian_id=171,
            measurement_noise_std=0.6, max_steps=60, seed=42,
        )
        true_states = data["true_states"]
        measurements = data["measurements"]

        # Run KF to get filtered estimates
        dt = data["dt"]
        F = np.array([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1,  0],
                       [0, 0, 0,  1]])
        H = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]])
        Q = 0.1 * np.eye(4)
        R = 0.36 * np.eye(2)
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R,
                          x0=true_states[0], P0=np.eye(4))
        results = kf.run(measurements)
        estimates = np.array([x[:2] for x in results["x_estimates"]])

        # Scale to fit Manim scene coordinates (roughly ±3)
        scale = 3.0 / max(
            np.ptp(true_states[:, 0]), np.ptp(true_states[:, 1]), 1
        )
        true_pos = true_states[:, :2] * scale
        meas_scaled = measurements * scale
        est_scaled = estimates * scale

        # Center everything
        center = true_pos.mean(axis=0)
        true_pos -= center
        meas_scaled -= center
        est_scaled -= center

        # ── Beat 1: Cold open ─────────────────────────────────────────
        wrong_pos = true_pos[0] + np.array([1.2, -0.8])
        gps_dot = Dot(
            np.array([wrong_pos[0], wrong_pos[1], 0]),
            radius=DOT_RADIUS_LARGE,
            color=COLOR_MEASUREMENT,
            fill_opacity=0.9,
        )

        self.set_speech_service(narrator)
        with self.voiceover(
            text=(
                "According to your GPS, you're standing in the middle "
                "of a parking lot. You're not. You're on the sidewalk, "
                "ten meters away."
            ),
        ) as tracker:
            self.play(FadeIn(gps_dot, scale=2), run_time=FAST_ANIM)
            self.play(
                Flash(gps_dot, color=COLOR_MEASUREMENT, flash_radius=0.4),
                run_time=0.6,
            )

        with self.voiceover(
            text="Your phone is lying to you.",
            prosody={"rate": "-10%"},
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        self.play(FadeOut(gps_dot), run_time=FAST_ANIM)

        # ── Beat 2: The noisy reality ─────────────────────────────────
        meas_dots = VGroup()
        for m in meas_scaled:
            dot = Dot(
                np.array([m[0], m[1], 0]),
                radius=MEASUREMENT_DOT_RADIUS,
                color=COLOR_MEASUREMENT,
                fill_opacity=0.8,
            )
            meas_dots.add(dot)

        with self.voiceover(
            text=(
                "These are real GPS coordinates from the ETH Zurich "
                "pedestrian dataset. Sixty readings."
            ),
        ) as tracker:
            batch_size = 8
            for start in range(0, len(meas_dots), batch_size):
                batch = meas_dots[start : start + batch_size]
                self.play(
                    LaggedStart(
                        *[FadeIn(d, scale=1.5) for d in batch],
                        lag_ratio=0.15,
                    ),
                    run_time=0.8,
                )

        with self.voiceover(
            text="Every single one of them is wrong.",
            prosody={"rate": "-15%"},
        ) as tracker:
            self.wait(PAUSE_SHORT)

        self.set_speech_service(skeptic)
        with self.voiceover(text="How wrong?") as tracker:
            self.wait(PAUSE_SHORT)

        self.set_speech_service(narrator)
        with self.voiceover(
            text=(
                "Wrong enough that if you connected the dots, "
                "you'd think this person was lost."
            ),
        ) as tracker:
            self.wait(PAUSE_SHORT)

        # ── Beat 3: The true path ─────────────────────────────────────
        true_points = [np.array([p[0], p[1], 0]) for p in true_pos]
        true_path = VMobject()
        true_path.set_points_smoothly(true_points)
        true_path.set_color(COLOR_TRUE_PATH)
        true_path.set_stroke(width=2, opacity=0.8)
        true_path_dashed = DashedVMobject(true_path, num_dashes=40)

        with self.voiceover(
            text=(
                "Here's where they actually walked. Smooth. Predictable. "
                "And completely invisible to the sensor. All we have are "
                "those blue dots. The question is, can we recover this path?"
            ),
        ) as tracker:
            self.play(Create(true_path_dashed), run_time=SLOW_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Beat 4: The reveal ────────────────────────────────────────
        est_points = [np.array([p[0], p[1], 0]) for p in est_scaled]
        est_path = VMobject()
        est_path.set_points_smoothly(est_points)
        est_path.set_color(COLOR_POSTERIOR)
        est_path.set_stroke(width=3)

        self.set_speech_service(skeptic)
        with self.voiceover(text="So you average them.") as tracker:
            self.wait(PAUSE_SHORT)

        self.set_speech_service(narrator)
        with self.voiceover(
            text=(
                "No. Averaging uses the data but ignores the physics. "
                "Pedestrians don't teleport between readings. Instead, "
                "you combine two imperfect things, a prediction from a "
                "model that's too simple, and a measurement from a sensor "
                "that's too noisy, and you get something better than "
                "either one."
            ),
        ) as tracker:
            self.play(Create(est_path), run_time=SLOW_ANIM)
            self.wait(PAUSE_LONG)

        # ── Beat 5: The name ──────────────────────────────────────────
        title = Text(
            "The Kalman Filter",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3)
        title.set_z_index(10)

        cite = Text(
            "Kalman, 1960",
            color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
        )
        cite.next_to(title, DOWN, buff=SMALL_BUFF)

        self.set_speech_service(narrator_newscast)
        with self.voiceover(
            text=(
                "This is the Kalman Filter. Rudolf Kalman, 1960. It "
                "guided Apollo to the moon. It's running right now in "
                "every GPS chip, every autopilot, every phone in this "
                "room. Traffic management systems run it every few "
                "seconds. Power grids use it to estimate ten thousand "
                "bus voltages in real time."
            ),
        ) as tracker:
            self.play(
                FadeIn(title, shift=DOWN * 0.3),
                run_time=NORMAL_ANIM,
            )
            self.play(FadeIn(cite), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        self.set_speech_service(skeptic)
        with self.voiceover(
            text="I've heard of it.",
            prosody={"rate": "-10%"},
        ) as tracker:
            self.wait(PAUSE_SHORT)

        self.set_speech_service(narrator)
        with self.voiceover(text="Then you know the name.") as tracker:
            self.wait(PAUSE_SHORT)

        self.set_speech_service(narrator_hopeful)
        with self.voiceover(
            text="By the end of this series, you'll know the geometry.",
        ) as tracker:
            note = make_observation_note(
                "Kalman (1960): A New Approach to Linear\n"
                "Filtering and Prediction Problems."
            )
            note.to_edge(DOWN, buff=0.4)
            self.play(FadeIn(note, shift=UP * 0.2), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # ── Cleanup ───────────────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=NORMAL_ANIM,
        )
