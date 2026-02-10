"""Part 8, Scene 1: What if We Don't Know F?

Data: Conceptual diagrams

The Kalman filter requires a known dynamics model F. But in many
real-world problems, the dynamics are unknown or too complex to specify.
This motivates learning the dynamics from data: f_theta replaces F.

Papers:
- Kalman (1960) — original KF (known F)
- Ljung (1999) — system identification
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


class SceneUnknownDynamics(VoiceoverScene, MovingCameraScene):
    """What if we don't know F? Motivation for learned dynamics.

    Visual: KF pipeline with F grayed out, then replaced by f_theta.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="Everything we've built so far — Kalman filters, EKF, UKF — "
                 "relies on one critical assumption: we know the dynamics "
                 "model. But what happens when we don't?"
        ) as tracker:
            title = Text(
                "What if We Don't Know F?",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Classical pipeline with F highlighted ─────────────────────
        with self.voiceover(
            text="Recall the classical pipeline. The state transition matrix F "
                 "tells the filter how the state evolves. The observation "
                 "matrix H maps states to measurements. Together with Q and R, "
                 "they define the entire model."
        ) as tracker:
            # F matrix box (highlighted)
            f_box = RoundedRectangle(
                corner_radius=0.1, width=2.0, height=0.8,
                color=COLOR_PREDICTION, fill_opacity=0.3, stroke_width=2,
            )
            f_label = Text("F", color=COLOR_PREDICTION, font_size=HEADING_FONT_SIZE)
            f_label.move_to(f_box)
            f_block = VGroup(f_box, f_label)

            # Predict box
            predict_box = RoundedRectangle(
                corner_radius=0.1, width=2.0, height=0.8,
                color=COLOR_PREDICTION, fill_opacity=0.2, stroke_width=2,
            )
            predict_label = Text("Predict", color=COLOR_PREDICTION, font_size=20)
            predict_label.move_to(predict_box)
            predict_block = VGroup(predict_box, predict_label)

            # Update box
            update_box = RoundedRectangle(
                corner_radius=0.1, width=2.0, height=0.8,
                color=COLOR_MEASUREMENT, fill_opacity=0.2, stroke_width=2,
            )
            update_label = Text("Update", color=COLOR_MEASUREMENT, font_size=20)
            update_label.move_to(update_box)
            update_block = VGroup(update_box, update_label)

            # H, Q, R boxes (smaller)
            h_box = RoundedRectangle(
                corner_radius=0.08, width=1.2, height=0.6,
                color=COLOR_MEASUREMENT, fill_opacity=0.15, stroke_width=1.5,
            )
            h_label = Text("H", color=COLOR_MEASUREMENT, font_size=20)
            h_label.move_to(h_box)
            h_block = VGroup(h_box, h_label)

            qr_box = RoundedRectangle(
                corner_radius=0.08, width=1.6, height=0.6,
                color=COLOR_PROCESS_NOISE, fill_opacity=0.15, stroke_width=1.5,
            )
            qr_label = Text("Q, R", color=COLOR_PROCESS_NOISE, font_size=20)
            qr_label.move_to(qr_box)
            qr_block = VGroup(qr_box, qr_label)

            # Layout
            pipeline = VGroup(f_block, predict_block, update_block)
            pipeline.arrange(RIGHT, buff=0.6)
            pipeline.move_to(UP * 0.3)

            h_block.next_to(update_block, DOWN, buff=0.3)
            qr_block.next_to(pipeline, DOWN, buff=0.8)

            arrows_pipe = VGroup(
                Arrow(f_block.get_right(), predict_block.get_left(),
                      color=SLATE, stroke_width=2, buff=0.1),
                Arrow(predict_block.get_right(), update_block.get_left(),
                      color=SLATE, stroke_width=2, buff=0.1),
            )

            self.play(
                FadeIn(pipeline, shift=RIGHT * 0.3),
                Create(arrows_pipe),
                FadeIn(h_block), FadeIn(qr_block),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── Gray out F — the unknown ──────────────────────────────────
        with self.voiceover(
            text="But for many real-world systems — weather, protein folding, "
                 "robot dynamics — writing down F is impossible. The dynamics "
                 "are too complex, too nonlinear, or simply unknown."
        ) as tracker:
            # Gray out F
            f_box_gray = RoundedRectangle(
                corner_radius=0.1, width=2.0, height=0.8,
                color=SLATE, fill_opacity=0.1, stroke_width=2,
            )
            f_label_gray = Text("F = ???", color=SLATE, font_size=HEADING_FONT_SIZE)
            f_label_gray.move_to(f_box_gray)
            f_block_gray = VGroup(f_box_gray, f_label_gray)
            f_block_gray.move_to(f_block)

            self.play(
                Transform(f_block, f_block_gray),
                run_time=NORMAL_ANIM,
            )

            # Question mark emphasis
            question = Text(
                "Unknown dynamics!",
                color=SWISS_RED, font_size=BODY_FONT_SIZE,
            )
            question.next_to(f_block, UP, buff=0.3)
            self.play(FadeIn(question, scale=0.8), run_time=FAST_ANIM)

        self.wait(PAUSE_MEDIUM)

        # ── Examples of unknown dynamics ──────────────────────────────
        with self.voiceover(
            text="Consider these examples: predicting weather requires modeling "
                 "chaotic fluid dynamics. Robot locomotion involves complex "
                 "contact physics. Game playing requires understanding rules "
                 "you might never see written down."
        ) as tracker:
            examples = VGroup(
                Text("Weather: chaotic fluid dynamics", color=COLOR_TEXT,
                     font_size=SMALL_FONT_SIZE),
                Text("Robotics: contact physics", color=COLOR_TEXT,
                     font_size=SMALL_FONT_SIZE),
                Text("Games: unknown rules", color=COLOR_TEXT,
                     font_size=SMALL_FONT_SIZE),
                Text("Biology: protein dynamics", color=COLOR_TEXT,
                     font_size=SMALL_FONT_SIZE),
            ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            examples.to_edge(RIGHT, buff=0.5).shift(DOWN * 0.5)

            self.play(
                FadeIn(examples, shift=LEFT * 0.2),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_SHORT)

        # ── Replace F with f_theta ────────────────────────────────────
        with self.voiceover(
            text="The solution: replace the hand-crafted F with a learned "
                 "function f theta. A neural network learns the dynamics "
                 "directly from data. This is the core idea behind world "
                 "models — and it connects directly to the Kalman filter."
        ) as tracker:
            self.play(
                FadeOut(question), FadeOut(examples),
                run_time=FAST_ANIM,
            )

            # New f_theta box (highlighted in violet/TF color)
            f_theta_box = RoundedRectangle(
                corner_radius=0.1, width=2.0, height=0.8,
                color=COLOR_FILTER_TF, fill_opacity=0.3, stroke_width=2,
            )
            f_theta_label = Text("f_theta", color=COLOR_FILTER_TF,
                                 font_size=HEADING_FONT_SIZE)
            f_theta_label.move_to(f_theta_box)
            f_theta_block = VGroup(f_theta_box, f_theta_label)
            f_theta_block.move_to(f_block)

            nn_label = Text(
                "Neural network", color=COLOR_FILTER_TF,
                font_size=SMALL_FONT_SIZE,
            )
            nn_label.next_to(f_theta_block, UP, buff=0.2)

            self.play(
                Transform(f_block, f_theta_block),
                FadeIn(nn_label),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── The predict/update loop still works ───────────────────────
        with self.voiceover(
            text="Here's the beautiful part: the predict-update loop remains "
                 "exactly the same. Predict with the learned model, update with "
                 "observations. The structure of Bayesian filtering doesn't "
                 "change — only the dynamics model does."
        ) as tracker:
            # Highlight predict and update
            highlight_rect = SurroundingRectangle(
                VGroup(predict_block, update_block),
                color=COLOR_HIGHLIGHT, buff=0.15, stroke_width=2,
            )
            same_label = Text(
                "Same predict/update loop!",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            )
            same_label.to_edge(DOWN, buff=0.5)

            self.play(
                Create(highlight_rect),
                FadeIn(same_label, shift=UP * 0.2),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── Preview of what's coming ──────────────────────────────────
        with self.voiceover(
            text="In this part, we'll explore the world model family: "
                 "the RSSM architecture from Dreamer, Deep Kalman Filters, "
                 "and MuZero's learned planning model. All are variations "
                 "on the same theme: learn the dynamics, keep the structure."
        ) as tracker:
            self.play(FadeOut(highlight_rect), FadeOut(same_label),
                      run_time=FAST_ANIM)

            preview = VGroup(
                Text("1. RSSM — predict/update with learned dynamics",
                     color=COLOR_FILTER_TF, font_size=SMALL_FONT_SIZE),
                Text("2. Deep Kalman Filters — variational inference",
                     color=COLOR_SSM, font_size=SMALL_FONT_SIZE),
                Text("3. Dreamer — learning by imagining",
                     color=COLOR_FILTER_KALMANNET, font_size=SMALL_FONT_SIZE),
                Text("4. MuZero — learned models for planning",
                     color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE),
            ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            preview.to_edge(DOWN, buff=0.5)

            self.play(
                FadeIn(preview, shift=UP * 0.2),
                run_time=NORMAL_ANIM,
            )

        note = make_observation_note(
            "System identification (Ljung 1999) already learned F from data.\n"
            "World models learn f_theta end-to-end from raw observations."
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
