"""Scene 2: Bayesian Foundations

Introduces Bayes' theorem and connects it to recursive state estimation.
Prior → Likelihood → Posterior, with color-coded terms and full equations.
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *


class SceneBayesFoundations(VoiceoverScene, Scene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────────
        title = Text("Bayesian State Estimation", color=COLOR_TEXT,
                      font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.4)

        # ── Bayes' Theorem ──────────────────────────────────────────────
        # P(x|z) ∝ P(z|x) · P(x)
        bayes = MathTex(
            r"P(", r"\mathbf{x}", r"|", r"\mathbf{z}", r")",
            r"\propto",
            r"P(", r"\mathbf{z}", r"|", r"\mathbf{x}", r")",
            r"\cdot",
            r"P(", r"\mathbf{x}", r")",
            font_size=EQUATION_FONT_SIZE, color=COLOR_EQUATION,
        )
        bayes.next_to(title, DOWN, buff=LARGE_BUFF)

        with self.voiceover(text="At its heart, the Kalman Filter is Bayesian state estimation. Here's Bayes' theorem: the posterior is proportional to the likelihood times the prior.") as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(Write(bayes), run_time=SLOW_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Color-code each term ────────────────────────────────────────
        # Posterior — gold
        posterior_indices = [0, 1, 2, 3, 4]
        # Likelihood — blue
        likelihood_indices = [6, 7, 8, 9, 10]
        # Prior — red
        prior_indices = [12, 13, 14]

        posterior_label = Text("Posterior", color=COLOR_POSTERIOR,
                                font_size=SMALL_FONT_SIZE)
        posterior_label.next_to(bayes[2], UP, buff=0.3)

        likelihood_label = Text("Likelihood", color=COLOR_MEASUREMENT,
                                 font_size=SMALL_FONT_SIZE)
        likelihood_label.next_to(bayes[8], UP, buff=0.3)

        prior_label = Text("Prior", color=COLOR_PREDICTION,
                            font_size=SMALL_FONT_SIZE)
        prior_label.next_to(bayes[13], UP, buff=0.3)

        with self.voiceover(text="The posterior in gold is what we want — our updated belief. The likelihood in blue captures the sensor model. And the prior in red is everything we knew before the measurement.") as tracker:
            self.play(
                *[bayes[i].animate.set_color(COLOR_POSTERIOR) for i in posterior_indices],
                run_time=FAST_ANIM,
            )
            self.play(FadeIn(posterior_label), run_time=FAST_ANIM)

            self.play(
                *[bayes[i].animate.set_color(COLOR_MEASUREMENT) for i in likelihood_indices],
                run_time=FAST_ANIM,
            )
            self.play(FadeIn(likelihood_label), run_time=FAST_ANIM)

            self.play(
                *[bayes[i].animate.set_color(COLOR_PREDICTION) for i in prior_indices],
                run_time=FAST_ANIM,
            )
            self.play(FadeIn(prior_label), run_time=FAST_ANIM)

            self.wait(PAUSE_LONG)

        # ── Recursive form ──────────────────────────────────────────────
        labels_group = VGroup(posterior_label, likelihood_label, prior_label)

        recursive_title = Text("Recursive Bayesian Estimation",
                                color=COLOR_TEXT, font_size=HEADING_FONT_SIZE)
        recursive_title.next_to(bayes, DOWN, buff=LARGE_BUFF)

        recursive_eq = MathTex(
            r"P(\mathbf{x}_k \mid \mathbf{z}_{1:k})",
            r"\propto",
            r"P(\mathbf{z}_k \mid \mathbf{x}_k)",
            r"\cdot",
            r"P(\mathbf{x}_k \mid \mathbf{z}_{1:k-1})",
            font_size=EQUATION_FONT_SIZE, color=COLOR_EQUATION,
        )
        recursive_eq.next_to(recursive_title, DOWN, buff=STANDARD_BUFF)

        # Color code
        recursive_eq[0].set_color(COLOR_POSTERIOR)
        recursive_eq[2].set_color(COLOR_MEASUREMENT)
        recursive_eq[4].set_color(COLOR_PREDICTION)

        with self.voiceover(text="Now the beautiful part: in filtering, this becomes recursive. The posterior at time k depends on the current measurement times the prior from all previous steps.") as tracker:
            self.play(FadeOut(labels_group), run_time=FAST_ANIM)
            self.play(FadeIn(recursive_title, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.play(Write(recursive_eq), run_time=SLOW_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Key insight ─────────────────────────────────────────────────
        insight_box = VGroup()
        insight_text = Text(
            "Today's posterior becomes tomorrow's prior",
            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
        )
        insight_text.next_to(recursive_eq, DOWN, buff=LARGE_BUFF)

        arrow_loop = CurvedArrow(
            recursive_eq[0].get_right() + RIGHT * 0.3,
            recursive_eq[4].get_right() + RIGHT * 0.3 + DOWN * 0.6,
            color=COLOR_HIGHLIGHT,
            stroke_width=2,
        )

        # ── Connection to KF ───────────────────────────────────────────
        kf_note = Text(
            "For linear systems with Gaussian noise,\n"
            "this is the Kalman Filter.",
            color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
            line_spacing=1.3,
        )
        kf_note.to_edge(DOWN, buff=0.5)

        with self.voiceover(text="Today's posterior becomes tomorrow's prior — a continuous loop of prediction and correction. For linear systems with Gaussian noise, this has an exact closed-form solution: the Kalman Filter.") as tracker:
            self.play(
                FadeIn(insight_text, shift=UP * 0.2),
                Create(arrow_loop),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_LONG)
            self.play(FadeIn(kf_note), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

        # ── Fade out ───────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
