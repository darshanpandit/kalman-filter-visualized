"""Scene 03: Walking the Dog — Frechet distance via the dog-walking metaphor.

Two-voice format (Jenny = narrator, Tony = Darshan) using Azure TTS.
The mathematical heart of the conflation chapter. Explains Frechet distance
using the classic dog-walking analogy, then contrasts with Hausdorff distance.

Voices: narrator (chat), narrator rate=-10%, narrator_whisper, darshan (friendly)

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
from pandit2019_conflation.data import DISTANCE_COMPARISON, fig_path


class SceneWalkingTheDog(VoiceoverScene, MovingCameraScene):
    """Beat 3 — Walking the Dog (Frechet distance)."""

    def construct(self):
        # ── Voice setup ─────────────────────────────────────────────
        narrator = AzureService(voice="en-US-JennyNeural", style="chat")
        narrator_whisper = AzureService(
            voice="en-US-JennyNeural", style="whispering",
        )
        darshan = AzureService(voice="en-US-TonyNeural", style="friendly")
        self.set_speech_service(narrator)
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────
        title = Text(
            "Walking the Dog", color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text=(
                "To match road segments, we need to measure how similar "
                "two curves are. And there's a beautiful idea from "
                "computational geometry that does exactly this."
            ),
        ) as tracker:
            self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Build two curvy paths ───────────────────────────────────
        # Person path (red) — upper curve
        person_anchors = [
            np.array([-5.0, 0.8, 0]),
            np.array([-3.0, 1.6, 0]),
            np.array([-0.5, 0.2, 0]),
            np.array([1.5, 1.4, 0]),
            np.array([3.5, 0.5, 0]),
            np.array([5.0, 1.1, 0]),
        ]
        person_path = VMobject()
        person_path.set_points_smoothly(person_anchors)
        person_path.set_color(COLOR_PREDICTION)
        person_path.set_stroke(width=4)

        # Dog path (blue) — lower curve, offset and differently shaped
        dog_anchors = [
            np.array([-5.0, -0.9, 0]),
            np.array([-2.5, -1.7, 0]),
            np.array([-0.5, -0.6, 0]),
            np.array([1.0, -1.9, 0]),
            np.array([3.0, -0.7, 0]),
            np.array([5.0, -1.1, 0]),
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
            text=(
                "Imagine a person walking along one road, and their dog "
                "walking along another. They're connected by a leash."
            ),
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

        # ── Person and dog dots + leash ─────────────────────────────
        person_dot = Dot(radius=0.12, color=CREAM).set_z_index(5)
        dog_dot = Dot(radius=0.10, color=COLOR_HIGHLIGHT).set_z_index(5)
        person_dot.move_to(person_path.point_from_proportion(0))
        dog_dot.move_to(dog_path.point_from_proportion(0))

        person_lbl = Text(
            "Person", color=CREAM, font_size=CHART_LABEL_FONT_SIZE,
        )
        dog_lbl = Text(
            "Dog", color=COLOR_HIGHLIGHT, font_size=CHART_LABEL_FONT_SIZE,
        )
        person_lbl.next_to(person_dot, UP, buff=0.12)
        dog_lbl.next_to(dog_dot, DOWN, buff=0.12)

        leash = always_redraw(
            lambda: DashedLine(
                person_dot.get_center(), dog_dot.get_center(),
                color=CREAM, stroke_width=2, stroke_opacity=0.7,
                dash_length=0.1,
            )
        )

        # ── Leash length readouts ───────────────────────────────────
        leash_label = always_redraw(
            lambda: Text(
                f"Leash: {np.linalg.norm(person_dot.get_center() - dog_dot.get_center()):.2f}",
                color=CREAM, font_size=CHART_LABEL_FONT_SIZE,
            ).move_to(
                (person_dot.get_center() + dog_dot.get_center()) / 2
                + RIGHT * 1.4
            )
        )

        max_leash = ValueTracker(0)
        max_leash_label = always_redraw(
            lambda: Text(
                f"Max leash: {max_leash.get_value():.2f}",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            ).to_edge(DOWN, buff=0.5)
        )

        self.play(
            FadeIn(person_dot, scale=1.5), FadeIn(dog_dot, scale=1.5),
            FadeIn(person_lbl), FadeIn(dog_lbl),
            run_time=FAST_ANIM,
        )
        self.add(leash, leash_label, max_leash_label)

        # ── The key rule (narrator, slowed prosody) ─────────────────
        self.set_speech_service(narrator)
        with self.voiceover(
            text=(
                "Here is the rule. Neither the person nor the dog can go "
                "backwards. They can speed up, slow down, even stop and "
                "wait. But they must always move forward."
            ),
            prosody={"rate": "-10%"},
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Dog-walking animation ───────────────────────────────────
        # ValueTracker for synchronized forward walk
        progress = ValueTracker(0)

        # Dog walks with slightly different pacing to show independence
        def dog_alpha(t):
            """Dog advances faster in the middle, slower at ends."""
            return np.clip(t ** 0.9, 0, 1)

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

        # Track maximum leash during walk
        def update_max(m):
            dist = np.linalg.norm(
                person_dot.get_center() - dog_dot.get_center()
            )
            if dist > max_leash.get_value():
                max_leash.set_value(dist)
        person_dot.add_updater(update_max)

        with self.voiceover(
            text=(
                "The Frechet distance is the shortest possible longest "
                "leash needed to complete the walk."
            ),
            prosody={"rate": "-10%"},
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

        # ── Flash max leash ─────────────────────────────────────────
        # Remove always_redraw objects, replace with static
        self.remove(leash, leash_label, max_leash_label)
        final_max = max_leash.get_value()
        max_leash_static = Text(
            f"Max leash: {final_max:.2f}",
            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
        ).to_edge(DOWN, buff=0.5)
        self.add(max_leash_static)

        self.play(
            max_leash_static.animate.scale(1.15),
            Flash(max_leash_static, color=COLOR_HIGHLIGHT, flash_radius=0.6),
            run_time=NORMAL_ANIM,
        )

        # ── Darshan: worst-case best-case ───────────────────────────
        self.set_speech_service(darshan)
        with self.voiceover(
            text=(
                "The worst-case best-case distance. You optimize the walk "
                "to minimize the maximum leash. And that no-backtracking "
                "rule is what makes it perfect for roads. Roads have "
                "direction."
            ),
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Narrator whisper: foreshadowing ─────────────────────────
        self.set_speech_service(narrator_whisper)
        with self.voiceover(
            text=(
                "Remember this structure. Optimizing over all valid "
                "couplings to minimize cost. We'll see it again."
            ),
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Transition: clear walk, fade paths, show paper figure ───
        walk_group = VGroup(
            person_dot, dog_dot, person_lbl, dog_lbl,
            lbl_person, lbl_dog, max_leash_static,
        )
        self.play(
            FadeOut(walk_group),
            person_path.animate.set_stroke(opacity=0.2),
            dog_path.animate.set_stroke(opacity=0.2),
            run_time=NORMAL_ANIM,
        )
        self.play(
            FadeOut(person_path), FadeOut(dog_path),
            run_time=FAST_ANIM,
        )

        # ── Figure 4: Frechet vs Hausdorff from paper ───────────────
        fig = ImageMobject(fig_path("fig4_frechet_hausdorff.png"))
        fig.scale_to_fit_width(8).move_to(ORIGIN + DOWN * 0.3)

        self.set_speech_service(narrator)
        with self.voiceover(
            text=(
                "Now compare that to Hausdorff distance. Hausdorff allows "
                "backtracking. It treats curves as point sets, ignoring "
                "order. A weaker measure."
            ),
        ) as tracker:
            self.play(FadeIn(fig), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Comparison cards: Frechet vs Hausdorff ──────────────────
        self.play(FadeOut(fig), run_time=FAST_ANIM)

        # --- Frechet card (left, TEAL) ---
        frechet_bg = RoundedRectangle(
            width=4.8, height=3.4, corner_radius=0.2,
            fill_color=DARK_SLATE, fill_opacity=0.85,
            stroke_color=TEAL, stroke_width=2,
        ).shift(LEFT * 2.9 + DOWN * 0.3)

        frechet_items = VGroup(
            Text("Frechet Distance", color=TEAL, font_size=HEADING_FONT_SIZE),
            Text("No backtracking", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
            Text("Order-aware", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
            Text("Weight = 3", color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE),
        )
        frechet_items.arrange(DOWN, buff=0.25)
        frechet_items.move_to(frechet_bg)
        frechet_card = VGroup(frechet_bg, frechet_items)

        # --- Hausdorff card (right, SLATE) ---
        hausdorff_bg = RoundedRectangle(
            width=4.8, height=3.4, corner_radius=0.2,
            fill_color=DARK_SLATE, fill_opacity=0.85,
            stroke_color=SLATE, stroke_width=2,
        ).shift(RIGHT * 2.9 + DOWN * 0.3)

        hausdorff_items = VGroup(
            Text("Hausdorff Distance", color=SLATE, font_size=HEADING_FONT_SIZE),
            Text("Backtracking OK", color=COLOR_TEXT, font_size=BODY_FONT_SIZE),
            Text("Set-based", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
            Text("Weight = 2", color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE),
        )
        hausdorff_items.arrange(DOWN, buff=0.25)
        hausdorff_items.move_to(hausdorff_bg)
        hausdorff_card = VGroup(hausdorff_bg, hausdorff_items)

        self.set_speech_service(darshan)
        with self.voiceover(
            text=(
                "I use both in my algorithm. Frechet is stronger, so it "
                "gets weight three. Hausdorff gets two."
            ),
        ) as tracker:
            self.play(
                FadeIn(frechet_card, shift=UP * 0.3),
                FadeIn(hausdorff_card, shift=UP * 0.3),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_MEDIUM)

        # ── Narrator: why not Frechet alone? ────────────────────────
        self.set_speech_service(narrator)
        with self.voiceover(
            text="Why not Frechet alone?",
        ) as tracker:
            self.play(
                frechet_bg.animate.set_stroke(color=COLOR_HIGHLIGHT, width=3),
                run_time=FAST_ANIM,
            )
            self.wait(PAUSE_SHORT)

        # ── Darshan: complementary measures ─────────────────────────
        self.set_speech_service(darshan)
        with self.voiceover(
            text=(
                "Because Hausdorff catches cases where two curves are close "
                "everywhere but the endpoints are swapped. HPMS segments "
                "don't encode direction consistently — one line for both "
                "carriageways. So I compute the Frechet distance twice, "
                "forwards and reversed, and take the minimum. Hausdorff "
                "handles that naturally."
            ),
        ) as tracker:
            self.play(
                frechet_bg.animate.set_stroke(color=TEAL, width=2),
                hausdorff_bg.animate.set_stroke(color=COLOR_HIGHLIGHT, width=3),
                run_time=FAST_ANIM,
            )
            self.wait(PAUSE_MEDIUM)
            self.play(
                hausdorff_bg.animate.set_stroke(color=SLATE, width=2),
                run_time=FAST_ANIM,
            )

        # ── Fade out ────────────────────────────────────────────────
        self.wait(PAUSE_MEDIUM)
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob is not title],
            run_time=NORMAL_ANIM,
        )
        self.wait(PAUSE_MEDIUM)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=NORMAL_ANIM,
        )
