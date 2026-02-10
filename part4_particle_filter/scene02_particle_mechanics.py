"""Part 4, Scene 2: Particle Filter Mechanics

Shows the three steps: predict (scatter), weight (by likelihood), resample.
Uses a simple 1D example first for clarity.
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneParticleMechanics(VoiceoverScene, Scene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        with self.voiceover(text="The particle filter algorithm. Step one: represent the distribution as a cloud of particles. Each particle is a hypothesis about the state, and density reflects probability.") as tracker:
            title = Text("Particle Filter Algorithm", color=COLOR_TEXT,
                          font_size=TITLE_FONT_SIZE)
            title.to_edge(UP, buff=0.3)
            self.play(Write(title), run_time=NORMAL_ANIM)

            # ── Number line for 1D demo ─────────────────────────────────────
            line = NumberLine(
                x_range=[-4, 8, 1], length=10,
                include_numbers=True,
                color=COLOR_GRID,
                font_size=18,
            ).shift(DOWN * 0.5)
            self.play(Create(line), run_time=NORMAL_ANIM)

            # ── Step 1: Initial particles (uniform-ish spread) ──────────────
            step1 = Text("1. Particles represent the distribution",
                          color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE)
            step1.next_to(title, DOWN, buff=STANDARD_BUFF)
            self.play(FadeIn(step1), run_time=FAST_ANIM)

            rng = np.random.default_rng(42)
            n_particles = 30
            particles = rng.normal(2.0, 1.5, size=n_particles)

            dots = VGroup()
            for p in particles:
                y_jitter = rng.uniform(0.1, 0.8)
                dot = Dot(
                    line.n2p(p) + UP * y_jitter,
                    radius=0.05, color=COLOR_PROCESS_NOISE, fill_opacity=0.8,
                )
                dots.add(dot)

            self.play(FadeIn(dots, lag_ratio=0.05), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Step 2: Predict (move particles forward with noise) ─────────
        with self.voiceover(text="Step two: predict. Each particle moves according to the process model plus random noise. The cloud shifts and spreads, just like a Gaussian would grow during prediction.") as tracker:
            step2 = Text("2. Predict: move each particle + noise",
                          color=COLOR_PREDICTION, font_size=BODY_FONT_SIZE)
            step2.move_to(step1)
            self.play(FadeOut(step1), FadeIn(step2), run_time=FAST_ANIM)

            # Shift particles right (motion) with noise
            new_particles = particles + 1.5 + rng.normal(0, 0.5, size=n_particles)
            anims = []
            for i, dot in enumerate(dots):
                y_jitter = rng.uniform(0.1, 0.8)
                new_pos = line.n2p(new_particles[i]) + UP * y_jitter
                anims.append(dot.animate.move_to(new_pos))

            self.play(*anims, run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

        # ── Step 3: Measurement arrives ─────────────────────────────────
        with self.voiceover(text="Step three: a measurement arrives. We weight each particle by how consistent it is with the measurement. Close particles get high weight, distant ones get low weight.") as tracker:
            step3 = Text("3. Weight: compare to measurement",
                          color=COLOR_MEASUREMENT, font_size=BODY_FONT_SIZE)
            step3.move_to(step2)
            self.play(FadeOut(step2), FadeIn(step3), run_time=FAST_ANIM)

            # Show measurement
            z = 4.0
            meas_line = DashedLine(
                line.n2p(z) + DOWN * 0.3,
                line.n2p(z) + UP * 1.5,
                color=COLOR_MEASUREMENT, stroke_width=2,
            )
            meas_label = MathTex(r"z", color=COLOR_MEASUREMENT, font_size=BODY_FONT_SIZE)
            meas_label.next_to(meas_line, UP, buff=0.1)
            self.play(Create(meas_line), FadeIn(meas_label), run_time=NORMAL_ANIM)

            # Weight particles by proximity to measurement
            distances = np.abs(new_particles - z)
            weights = np.exp(-0.5 * (distances / 0.8) ** 2)
            weights /= weights.max()

            # Animate opacity change based on weight
            anims = []
            for i, dot in enumerate(dots):
                new_opacity = 0.15 + 0.85 * weights[i]
                new_radius = 0.03 + 0.05 * weights[i]
                anims.append(dot.animate.set_fill(opacity=new_opacity).scale(
                    new_radius / 0.05))

            self.play(*anims, run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Degeneracy warning ─────────────────────────────────────────
        with self.voiceover(text="Without resampling, most particles end up with negligible weight. This degeneracy wastes computation, only a handful of particles actually matter.") as tracker:
            degen_note = Text(
                "Without resampling, weights collapse — most particles become useless.",
                color=SWISS_RED, font_size=SMALL_FONT_SIZE,
            )
            degen_note.to_edge(DOWN, buff=0.3)
            self.play(FadeIn(degen_note), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)
            self.play(FadeOut(degen_note), run_time=FAST_ANIM)

        # ── Step 4: Resample ────────────────────────────────────────────
        with self.voiceover(text="Step four: resample. Duplicate high-weight particles and discard low-weight ones. The particles now cluster around the true state. Predict, weight, resample, that's the full cycle.") as tracker:
            step4 = Text("4. Resample: duplicate high-weight particles",
                          color=COLOR_POSTERIOR, font_size=BODY_FONT_SIZE)
            step4.move_to(step3)
            self.play(FadeOut(step3), FadeIn(step4), run_time=FAST_ANIM)

            # Resample: pick particles proportional to weight
            w_norm = weights / weights.sum()
            indices = rng.choice(n_particles, size=n_particles, p=w_norm)
            resampled = new_particles[indices]

            # Create new dots at resampled positions
            new_dots = VGroup()
            for p in resampled:
                y_jitter = rng.uniform(0.1, 0.8)
                dot = Dot(
                    line.n2p(p) + UP * y_jitter,
                    radius=0.05, color=COLOR_POSTERIOR, fill_opacity=0.8,
                )
                new_dots.add(dot)

            self.play(FadeOut(dots), FadeIn(new_dots, lag_ratio=0.03),
                      run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

            result = Text(
                "Particles cluster around the true state!",
                color=COLOR_POSTERIOR, font_size=BODY_FONT_SIZE,
            )
            result.to_edge(DOWN, buff=0.3)
            self.play(FadeOut(step4), FadeIn(result), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
