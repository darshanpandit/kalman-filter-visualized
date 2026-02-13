"""Scene 03: Walking the Dog — Frechet distance via the dog-walking metaphor.

The mathematical heart of the conflation chapter. Explains Frechet distance
using the classic dog-walking analogy, then contrasts with Hausdorff distance.
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from pandit2019_conflation.data import *


class SceneWalkingTheDog(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ─────────────────────────────────────────────────────
        title = Text(
            "Walking the Dog", color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text="To match road segments between two datasets, we need a way "
                 "to measure how similar two curves are. "
                 "Let me show you the most elegant idea in computational geometry."
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Build two curvy paths ─────────────────────────────────────
        # Person path (red) — upper curve
        person_anchors = [
            np.array([-5.0, 0.8, 0]),
            np.array([-3.0, 1.8, 0]),
            np.array([-0.5, 0.3, 0]),
            np.array([1.5, 1.5, 0]),
            np.array([3.5, 0.5, 0]),
            np.array([5.0, 1.2, 0]),
        ]
        person_path = VMobject()
        person_path.set_points_smoothly(person_anchors)
        person_path.set_color(COLOR_PREDICTION)
        person_path.set_stroke(width=4)

        # Dog path (blue) — lower curve
        dog_anchors = [
            np.array([-5.0, -0.8, 0]),
            np.array([-2.5, -1.6, 0]),
            np.array([-0.5, -0.5, 0]),
            np.array([1.0, -1.8, 0]),
            np.array([3.0, -0.6, 0]),
            np.array([5.0, -1.0, 0]),
        ]
        dog_path = VMobject()
        dog_path.set_points_smoothly(dog_anchors)
        dog_path.set_color(COLOR_MEASUREMENT)
        dog_path.set_stroke(width=4)

        # Labels for paths
        lbl_person = Text(
            "Person's path", color=COLOR_PREDICTION, font_size=SMALL_FONT_SIZE,
        )
        lbl_person.next_to(person_path.point_from_proportion(0.05), UP, buff=0.15)
        lbl_dog = Text(
            "Dog's path", color=COLOR_MEASUREMENT, font_size=SMALL_FONT_SIZE,
        )
        lbl_dog.next_to(dog_path.point_from_proportion(0.05), DOWN, buff=0.15)

        with self.voiceover(
            text="Imagine two curved roads. A person walks along one, "
                 "and their dog walks along the other. "
                 "They are connected by a leash."
        ) as tracker:
            self.play(
                Create(person_path), Create(dog_path),
                run_time=SLOW_ANIM,
            )
            self.play(
                FadeIn(lbl_person, shift=DOWN * 0.1),
                FadeIn(lbl_dog, shift=UP * 0.1),
                run_time=FAST_ANIM,
            )
            self.wait(PAUSE_SHORT)

        # ── Dog-walking animation ─────────────────────────────────────
        person_dot = Dot(radius=0.12, color=CREAM).set_z_index(5)
        dog_dot = Dot(radius=0.10, color=COLOR_HIGHLIGHT).set_z_index(5)
        person_dot.move_to(person_path.point_from_proportion(0))
        dog_dot.move_to(dog_path.point_from_proportion(0))

        person_lbl = Text("Person", color=CREAM, font_size=CHART_LABEL_FONT_SIZE)
        dog_lbl = Text("Dog", color=COLOR_HIGHLIGHT, font_size=CHART_LABEL_FONT_SIZE)
        person_lbl.next_to(person_dot, UP, buff=0.12)
        dog_lbl.next_to(dog_dot, DOWN, buff=0.12)

        leash = always_redraw(
            lambda: DashedLine(
                person_dot.get_center(), dog_dot.get_center(),
                color=CREAM, stroke_width=2, stroke_opacity=0.7,
                dash_length=0.1,
            )
        )

        # Leash length display
        leash_label = always_redraw(
            lambda: Text(
                f"Leash: {np.linalg.norm(person_dot.get_center() - dog_dot.get_center()):.2f}",
                color=CREAM, font_size=CHART_LABEL_FONT_SIZE,
            ).move_to(
                (person_dot.get_center() + dog_dot.get_center()) / 2 + RIGHT * 1.2
            )
        )

        self.play(
            FadeIn(person_dot, scale=1.5), FadeIn(dog_dot, scale=1.5),
            FadeIn(person_lbl), FadeIn(dog_lbl),
            run_time=FAST_ANIM,
        )
        self.add(leash, leash_label)

        # Track maximum leash during walk
        max_leash = ValueTracker(0)

        max_leash_label = always_redraw(
            lambda: Text(
                f"Max leash: {max_leash.get_value():.2f}",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            ).to_edge(DOWN, buff=0.5)
        )
        self.add(max_leash_label)

        # Animate the walk using ValueTracker for synchronised motion
        progress = ValueTracker(0)

        # Dog walks slightly non-uniformly to show independence
        def dog_alpha(t):
            """Dog advances a bit faster in the middle, slower at ends."""
            return t  # keep synchronised for clarity

        person_dot.add_updater(
            lambda m: m.move_to(person_path.point_from_proportion(
                np.clip(progress.get_value(), 0, 1)
            ))
        )
        dog_dot.add_updater(
            lambda m: m.move_to(dog_path.point_from_proportion(
                np.clip(dog_alpha(progress.get_value()), 0, 1)
            ))
        )
        person_lbl.add_updater(lambda m: m.next_to(person_dot, UP, buff=0.12))
        dog_lbl.add_updater(lambda m: m.next_to(dog_dot, DOWN, buff=0.12))

        # Custom updater to track max leash
        def update_max(m):
            dist = np.linalg.norm(
                person_dot.get_center() - dog_dot.get_center()
            )
            if dist > max_leash.get_value():
                max_leash.set_value(dist)
        person_dot.add_updater(update_max)

        with self.voiceover(
            text="The key rule: neither person nor dog can go backwards. "
                 "They can speed up, slow down, even stop and wait, "
                 "but they must always move forward along their path. "
                 "The Frechet distance is the shortest possible longest leash "
                 "needed across all possible walks."
        ) as tracker:
            self.play(
                progress.animate.set_value(1),
                run_time=6.0,
                rate_func=smooth,
            )
            self.wait(PAUSE_MEDIUM)

        # Clean up updaters
        person_dot.clear_updaters()
        dog_dot.clear_updaters()
        person_lbl.clear_updaters()
        dog_lbl.clear_updaters()

        # Flash the max leash value
        with self.voiceover(
            text="That maximum leash length, minimized over all valid walks, "
                 "is the Frechet distance between these two curves."
        ) as tracker:
            self.play(
                max_leash_label.animate.set_color(COLOR_HIGHLIGHT).scale(1.15),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_MEDIUM)

        # ── Transition: clear walking animation, show paper figure ────
        walk_group = VGroup(
            person_dot, dog_dot, person_lbl, dog_lbl,
            lbl_person, lbl_dog,
        )
        self.remove(leash, leash_label, max_leash_label)
        self.play(
            FadeOut(walk_group),
            FadeOut(max_leash_label),
            person_path.animate.set_stroke(opacity=0.3),
            dog_path.animate.set_stroke(opacity=0.3),
            run_time=NORMAL_ANIM,
        )

        # ── Figure 4: Frechet vs Hausdorff from paper ─────────────────
        fig = ImageMobject(fig_path("fig4_frechet_hausdorff.png"))
        fig.scale_to_fit_width(8).move_to(ORIGIN + DOWN * 0.3)

        with self.voiceover(
            text="Here is how the paper visualizes this. "
                 "Hausdorff distance allows backtracking. It simply asks: "
                 "what is the farthest any point on one curve is from "
                 "the closest point on the other? "
                 "But Frechet is stricter. You must walk forward."
        ) as tracker:
            self.play(
                FadeOut(person_path), FadeOut(dog_path),
                FadeIn(fig),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_LONG)

        # ── Comparison cards: Frechet vs Hausdorff ────────────────────
        self.play(FadeOut(fig), run_time=FAST_ANIM)

        dc = DISTANCE_COMPARISON

        def make_card(key, card_color, pos):
            info = dc[key]
            bg = RoundedRectangle(
                width=4.5, height=3.2, corner_radius=0.2,
                fill_color=DARK_SLATE, fill_opacity=0.85,
                stroke_color=card_color, stroke_width=2,
            ).move_to(pos)
            name = Text(
                info["name"], color=card_color, font_size=HEADING_FONT_SIZE,
            ).move_to(bg.get_top() + DOWN * 0.45)
            prop = Text(
                info["property"], color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
            ).next_to(name, DOWN, buff=0.3)
            strength = Text(
                info["strength"], color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
            ).next_to(prop, DOWN, buff=0.25)
            weight_val = 3 if key == "frechet" else 2
            weight = Text(
                f"Weight = {weight_val}",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            ).next_to(strength, DOWN, buff=0.25)
            return VGroup(bg, name, prop, strength, weight)

        frechet_card = make_card("frechet", TEAL, LEFT * 2.8)
        hausdorff_card = make_card("hausdorff", SLATE, RIGHT * 2.8)

        with self.voiceover(
            text="Frechet respects the direction of travel, "
                 "which is critical for roads that are one-way or have a clear flow. "
                 "That is why the authors gave Frechet a weight of 3, "
                 "versus only 2 for Hausdorff."
        ) as tracker:
            self.play(
                FadeIn(frechet_card, shift=UP * 0.3),
                FadeIn(hausdorff_card, shift=UP * 0.3),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_LONG)

        # Highlight Frechet card
        with self.voiceover(
            text="But wait, why not just use Frechet alone? "
                 "Because Hausdorff catches cases where two curves are close "
                 "everywhere but walk in opposite directions. "
                 "Together, they are complementary."
        ) as tracker:
            self.play(
                frechet_card[0].animate.set_stroke(color=COLOR_HIGHLIGHT, width=3),
                run_time=FAST_ANIM,
            )
            self.wait(PAUSE_LONG)
            self.play(
                frechet_card[0].animate.set_stroke(color=TEAL, width=2),
                hausdorff_card[0].animate.set_stroke(color=COLOR_HIGHLIGHT, width=3),
                run_time=FAST_ANIM,
            )
            self.wait(PAUSE_MEDIUM)

        # ── Fade out ──────────────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=NORMAL_ANIM,
        )
