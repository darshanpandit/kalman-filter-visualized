"""Part 7, Scene 1: The Assignment Problem — tracking multiple targets.

Data: ETH eth, 5-10 simultaneous pedestrians

When multiple targets move, each measurement could belong to any target.
The combinatorial explosion motivates data-association-free approaches.

Papers:
- Fortmann et al. (1983) JPDA
- Bar-Shalom & Fortmann (1988)
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.multi_track import MultiTrackPlot, TRACK_COLORS
from kalman_manim.mobjects.observation_note import make_observation_note
from kalman_manim.data.generators import generate_multi_target_scenario
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneAssignment(VoiceoverScene, MovingCameraScene):
    """The data association problem in multi-target tracking.

    Visual: Multiple tracks with ambiguous measurements.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # Generate multi-target data
        data = generate_multi_target_scenario(
            n_steps=40, n_targets_init=4, birth_step=50, death_step=50,
            clutter_rate=1.0, seed=42,
        )

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="So far, we tracked one pedestrian. But what about a "
                 "crowd? When multiple targets move simultaneously, a "
                 "new challenge emerges: data association."
        ) as tracker:
            title = Text(
                "The Assignment Problem",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Multi-track visualization ──────────────────────────────────
        with self.voiceover(
            text="Here are four pedestrians moving simultaneously. "
                 "Each dot is a noisy measurement — but which measurement "
                 "belongs to which pedestrian? Plus, some dots are clutter."
        ) as tracker:
            tracks = []
            for track in data["true_tracks"][:4]:
                tracks.append(track["states"][:, :2])

            plot = MultiTrackPlot(
                tracks=tracks,
                track_labels=[f"Ped {i+1}" for i in range(4)],
                width=6.0, height=4.0,
            )
            plot.next_to(title, DOWN, buff=0.5).shift(LEFT * 1.5)

            self.play(FadeIn(plot.axes), run_time=FAST_ANIM)
            for line in plot.track_lines:
                self.play(Create(line), run_time=0.5)
            self.play(FadeIn(plot.track_dots), FadeIn(plot.legend),
                      run_time=FAST_ANIM)

        self.wait(PAUSE_SHORT)

        # ── Combinatorial explosion ────────────────────────────────────
        with self.voiceover(
            text="With N targets and M measurements, there are up to "
                 "N-factorial possible assignments. For 10 targets, "
                 "that's over 3 million combinations — per time step."
        ) as tracker:
            factorial_text = VGroup(
                Text("N targets, M measurements:", color=COLOR_TEXT,
                     font_size=BODY_FONT_SIZE),
                Text("N! possible assignments", color=COLOR_HIGHLIGHT,
                     font_size=HEADING_FONT_SIZE),
                Text("10 targets → 3,628,800 combinations",
                     color=SWISS_RED, font_size=BODY_FONT_SIZE),
            ).arrange(DOWN, buff=0.2)
            factorial_text.to_edge(RIGHT, buff=0.5).shift(DOWN * 0.5)

            self.play(FadeIn(factorial_text, shift=LEFT * 0.2),
                      run_time=NORMAL_ANIM)

        self.wait(PAUSE_MEDIUM)

        # ── Solutions preview ──────────────────────────────────────────
        with self.voiceover(
            text="We'll explore three approaches: the IMM filter for "
                 "handling multiple motion models, the PHD filter that "
                 "avoids data association entirely, and learned social "
                 "prediction models."
        ) as tracker:
            approaches = VGroup(
                Text("1. IMM — multiple motion models", color=COLOR_FILTER_IMM,
                     font_size=SMALL_FONT_SIZE),
                Text("2. PHD — no data association needed", color=COLOR_FILTER_PHD,
                     font_size=SMALL_FONT_SIZE),
                Text("3. Social models — learned prediction", color=COLOR_SOCIAL,
                     font_size=SMALL_FONT_SIZE),
            ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            approaches.to_edge(DOWN, buff=0.4)

            self.play(FadeIn(approaches, shift=UP * 0.2), run_time=NORMAL_ANIM)

        note = make_observation_note(
            "JPDA: Joint Probabilistic Data Association\n"
            "(Fortmann et al. 1983; Bar-Shalom & Fortmann 1988)"
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
