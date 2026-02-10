"""Part 6, Scene 1: The Question — Can ML learn to filter?

Data: Recap sweep_results.npz from Part 5

Shows the classical filter pipeline (predict/update with known model),
then poses the question: what if we replace the model with a
sequence-to-sequence transformer? Sets up the narrative for Part 6.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.observation_note import make_observation_note
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneCanMLFilter(VoiceoverScene, MovingCameraScene):
    """Can transformers learn to do Kalman filtering?

    Visual: Classical pipeline vs transformer sequence-to-sequence.
    References: Goel & Bartlett (2024, L4DC).
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="We've benchmarked classical filters — KF, EKF, UKF, "
                 "and particle filters. They all need a hand-crafted "
                 "dynamics model. But what if we skip that step entirely?"
        ) as tracker:
            title = Text(
                "Can Transformers Learn to Filter?",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Classical pipeline ─────────────────────────────────────────
        with self.voiceover(
            text="The classical approach: you specify F, H, Q, R — "
                 "the dynamics model — then run predict and update in a "
                 "loop. The filter is only as good as your model."
        ) as tracker:
            # Classical pipeline: boxes with arrows
            model_box = RoundedRectangle(
                corner_radius=0.1, width=2.2, height=0.8,
                color=COLOR_PREDICTION, fill_opacity=0.2,
            )
            model_label = Text("F, H, Q, R", color=COLOR_PREDICTION, font_size=20)
            model_label.move_to(model_box)
            model = VGroup(model_box, model_label)

            predict_box = RoundedRectangle(
                corner_radius=0.1, width=1.8, height=0.8,
                color=COLOR_PREDICTION, fill_opacity=0.2,
            )
            predict_label = Text("Predict", color=COLOR_PREDICTION, font_size=20)
            predict_label.move_to(predict_box)
            predict = VGroup(predict_box, predict_label)

            update_box = RoundedRectangle(
                corner_radius=0.1, width=1.8, height=0.8,
                color=COLOR_MEASUREMENT, fill_opacity=0.2,
            )
            update_label = Text("Update", color=COLOR_MEASUREMENT, font_size=20)
            update_label.move_to(update_box)
            update = VGroup(update_box, update_label)

            estimate_box = RoundedRectangle(
                corner_radius=0.1, width=1.8, height=0.8,
                color=COLOR_POSTERIOR, fill_opacity=0.2,
            )
            est_label = Text("x\u0302_t", color=COLOR_POSTERIOR, font_size=22)
            est_label.move_to(estimate_box)
            estimate = VGroup(estimate_box, est_label)

            classical = VGroup(model, predict, update, estimate)
            classical.arrange(RIGHT, buff=0.4)
            classical.move_to(UP * 0.5)

            arrows_c = VGroup(
                Arrow(model.get_right(), predict.get_left(), color=SLATE,
                      stroke_width=2, buff=0.1),
                Arrow(predict.get_right(), update.get_left(), color=SLATE,
                      stroke_width=2, buff=0.1),
                Arrow(update.get_right(), estimate.get_left(), color=SLATE,
                      stroke_width=2, buff=0.1),
            )

            c_label = Text(
                "Classical: hand-crafted model",
                color=SLATE, font_size=SMALL_FONT_SIZE,
            )
            c_label.next_to(classical, UP, buff=0.2)

            self.play(
                FadeIn(classical, shift=RIGHT * 0.3),
                Create(arrows_c),
                FadeIn(c_label),
                run_time=NORMAL_ANIM,
            )

        # ── Transformer pipeline ───────────────────────────────────────
        with self.voiceover(
            text="The transformer approach: feed in the raw observation "
                 "sequence, and the model learns to output state estimates "
                 "directly. No hand-crafted dynamics needed."
        ) as tracker:
            obs_box = RoundedRectangle(
                corner_radius=0.1, width=2.5, height=0.8,
                color=COLOR_MEASUREMENT, fill_opacity=0.2,
            )
            obs_label = Text("z_1, z_2, ..., z_T", color=COLOR_MEASUREMENT,
                             font_size=18)
            obs_label.move_to(obs_box)
            obs = VGroup(obs_box, obs_label)

            tf_box = RoundedRectangle(
                corner_radius=0.1, width=2.5, height=0.8,
                color=COLOR_FILTER_TF, fill_opacity=0.2,
            )
            tf_label = Text("Transformer", color=COLOR_FILTER_TF, font_size=20)
            tf_label.move_to(tf_box)
            transformer = VGroup(tf_box, tf_label)

            out_box = RoundedRectangle(
                corner_radius=0.1, width=2.5, height=0.8,
                color=COLOR_POSTERIOR, fill_opacity=0.2,
            )
            out_label = Text("x\u0302_1, x\u0302_2, ..., x\u0302_T",
                             color=COLOR_POSTERIOR, font_size=18)
            out_label.move_to(out_box)
            output = VGroup(out_box, out_label)

            ml_pipeline = VGroup(obs, transformer, output)
            ml_pipeline.arrange(RIGHT, buff=0.6)
            ml_pipeline.move_to(DOWN * 1.5)

            arrows_ml = VGroup(
                Arrow(obs.get_right(), transformer.get_left(),
                      color=SLATE, stroke_width=2, buff=0.1),
                Arrow(transformer.get_right(), output.get_left(),
                      color=SLATE, stroke_width=2, buff=0.1),
            )

            ml_label = Text(
                "ML: learned from data",
                color=COLOR_FILTER_TF, font_size=SMALL_FONT_SIZE,
            )
            ml_label.next_to(ml_pipeline, UP, buff=0.2)

            self.play(
                FadeIn(ml_pipeline, shift=RIGHT * 0.3),
                Create(arrows_ml),
                FadeIn(ml_label),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── The question ───────────────────────────────────────────────
        with self.voiceover(
            text="Recent papers from 2022 to 2024 show this is not just "
                 "possible — transformers can match or even beat classical "
                 "filters. Let's see how."
        ) as tracker:
            question = Text(
                "Can it match a Kalman filter?",
                color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
            )
            question.to_edge(DOWN, buff=0.4)
            self.play(FadeIn(question, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        note = make_observation_note(
            "Goel & Bartlett (2024, L4DC): single attention layer\n"
            "implements Nadaraya-Watson ≈ KF for linear-Gaussian systems"
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
