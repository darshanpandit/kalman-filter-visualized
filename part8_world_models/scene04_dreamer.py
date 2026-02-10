"""Part 8, Scene 4: Dreamer — Learning by Imagining.

Data: DMControl benchmark scores (published results)

Dreamer uses the RSSM world model to imagine trajectories in latent
space, then trains a policy on those imagined trajectories. Three
generations: DreamerV1, V2, V3, each improving sample efficiency.

Papers:
- Hafner et al. (2020, ICLR) — DreamerV1: Dream to Control
- Hafner et al. (2021, NeurIPS) — DreamerV2: Mastering Atari
- Hafner et al. (2023, JMLR) — DreamerV3: Mastering Diverse Domains
- Barth-Maron et al. (2018) — D4PG baseline
- Haarnoja et al. (2018) — SAC baseline
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.comparison_table import ComparisonTable
from kalman_manim.mobjects.observation_note import make_observation_note
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneDreamer(VoiceoverScene, MovingCameraScene):
    """Dreamer: learning to act by imagining in a learned world model.

    Visual: Imagination pipeline + DMControl benchmark scores table.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        with self.voiceover(
            text="Dreamer takes the RSSM world model and asks a simple "
                 "question: if we can predict the future in latent space, "
                 "why not train a policy entirely from imagined "
                 "trajectories?"
        ) as tracker:
            title = Text(
                "Dreamer: Learning by Imagining",
                color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
            )
            title.to_edge(UP, buff=0.3).set_z_index(10)
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── The imagination pipeline ──────────────────────────────────
        with self.voiceover(
            text="The pipeline has three phases. First, experience: the "
                 "agent interacts with the real environment and collects "
                 "data. Second, world model: the RSSM learns to predict "
                 "what happens next. Third, imagination: the agent rolls "
                 "out thousands of trajectories in latent space and learns "
                 "a policy from these imagined experiences."
        ) as tracker:
            # Phase 1: Experience
            exp_box = RoundedRectangle(
                corner_radius=0.1, width=2.5, height=1.2,
                color=COLOR_MEASUREMENT, fill_opacity=0.2, stroke_width=2,
            )
            exp_title = Text("1. Experience", color=COLOR_MEASUREMENT,
                             font_size=18)
            exp_detail = Text("Interact with\nreal environment",
                              color=SLATE, font_size=14)
            exp_group = VGroup(exp_title, exp_detail).arrange(DOWN, buff=0.1)
            exp_group.move_to(exp_box)
            exp_block = VGroup(exp_box, exp_group)

            # Phase 2: World Model
            wm_box = RoundedRectangle(
                corner_radius=0.1, width=2.5, height=1.2,
                color=COLOR_FILTER_TF, fill_opacity=0.2, stroke_width=2,
            )
            wm_title = Text("2. World Model", color=COLOR_FILTER_TF,
                            font_size=18)
            wm_detail = Text("RSSM learns\ndynamics", color=SLATE,
                             font_size=14)
            wm_group = VGroup(wm_title, wm_detail).arrange(DOWN, buff=0.1)
            wm_group.move_to(wm_box)
            wm_block = VGroup(wm_box, wm_group)

            # Phase 3: Imagination
            img_box = RoundedRectangle(
                corner_radius=0.1, width=2.5, height=1.2,
                color=COLOR_HIGHLIGHT, fill_opacity=0.2, stroke_width=2,
            )
            img_title = Text("3. Imagination", color=COLOR_HIGHLIGHT,
                             font_size=18)
            img_detail = Text("Roll out in\nlatent space", color=SLATE,
                              font_size=14)
            img_group = VGroup(img_title, img_detail).arrange(DOWN, buff=0.1)
            img_group.move_to(img_box)
            img_block = VGroup(img_box, img_group)

            pipeline = VGroup(exp_block, wm_block, img_block)
            pipeline.arrange(RIGHT, buff=0.6)
            pipeline.next_to(title, DOWN, buff=0.7)

            arrows = VGroup(
                Arrow(exp_block.get_right(), wm_block.get_left(),
                      color=SLATE, stroke_width=2, buff=0.1),
                Arrow(wm_block.get_right(), img_block.get_left(),
                      color=SLATE, stroke_width=2, buff=0.1),
            )

            self.play(
                FadeIn(exp_block, shift=RIGHT * 0.2),
                run_time=NORMAL_ANIM,
            )
            self.play(
                FadeIn(wm_block, shift=RIGHT * 0.2),
                Create(arrows[0]),
                run_time=NORMAL_ANIM,
            )
            self.play(
                FadeIn(img_block, shift=RIGHT * 0.2),
                Create(arrows[1]),
                run_time=NORMAL_ANIM,
            )

        self.wait(PAUSE_MEDIUM)

        # ── Why imagination works ─────────────────────────────────────
        with self.voiceover(
            text="Why does imagination work? Because the RSSM's predict "
                 "step — our Kalman prediction — is cheap. You can roll "
                 "out thousands of latent trajectories in milliseconds. "
                 "No physics simulation needed. No rendering. Just matrix "
                 "multiplies in latent space."
        ) as tracker:
            benefit = VGroup(
                Text("Real environment: slow, expensive, dangerous",
                     color=SWISS_RED, font_size=SMALL_FONT_SIZE),
                Text("Imagined rollouts: fast, free, unlimited",
                     color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE),
            ).arrange(DOWN, buff=0.2)
            benefit.to_edge(DOWN, buff=0.5)

            self.play(FadeIn(benefit, shift=UP * 0.2), run_time=NORMAL_ANIM)

        self.wait(PAUSE_MEDIUM)

        # ── Evolution: V1 → V2 → V3 ──────────────────────────────────
        with self.voiceover(
            text="Dreamer evolved through three generations. Version 1 "
                 "used the basic RSSM with actor-critic in latent space. "
                 "Version 2 added discrete representations and mastered "
                 "Atari from pixels. Version 3 introduced symlog "
                 "predictions and a single architecture that works across "
                 "diverse domains — from continuous control to Minecraft."
        ) as tracker:
            self.play(
                FadeOut(pipeline), FadeOut(arrows), FadeOut(benefit),
                run_time=FAST_ANIM,
            )

            versions = VGroup(
                VGroup(
                    Text("DreamerV1 (2020)", color=COLOR_SSM,
                         font_size=BODY_FONT_SIZE),
                    Text("RSSM + actor-critic in latent space",
                         color=SLATE, font_size=SMALL_FONT_SIZE),
                ).arrange(DOWN, buff=0.05),
                VGroup(
                    Text("DreamerV2 (2021)", color=COLOR_FILTER_TF,
                         font_size=BODY_FONT_SIZE),
                    Text("Discrete latents, Atari mastery",
                         color=SLATE, font_size=SMALL_FONT_SIZE),
                ).arrange(DOWN, buff=0.05),
                VGroup(
                    Text("DreamerV3 (2023)", color=COLOR_HIGHLIGHT,
                         font_size=BODY_FONT_SIZE),
                    Text("Symlog predictions, one architecture for all",
                         color=SLATE, font_size=SMALL_FONT_SIZE),
                ).arrange(DOWN, buff=0.05),
            ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
            versions.next_to(title, DOWN, buff=0.7)

            # Arrows between versions
            v_arrows = VGroup(
                Arrow(
                    versions[0].get_bottom() + DOWN * 0.05,
                    versions[1].get_top() + UP * 0.05,
                    color=SLATE, stroke_width=1.5, buff=0.05,
                ),
                Arrow(
                    versions[1].get_bottom() + DOWN * 0.05,
                    versions[2].get_top() + UP * 0.05,
                    color=SLATE, stroke_width=1.5, buff=0.05,
                ),
            )

            for i, v in enumerate(versions):
                self.play(FadeIn(v, shift=RIGHT * 0.2), run_time=0.5)
                if i < len(v_arrows):
                    self.play(Create(v_arrows[i]), run_time=0.3)

        self.wait(PAUSE_MEDIUM)

        # ── DMControl benchmark scores ────────────────────────────────
        with self.voiceover(
            text="Here are the results on the DeepMind Control Suite, "
                 "the standard continuous control benchmark. D4PG — a "
                 "model-free method — scores 274. SAC reaches 437. PlaNet, "
                 "the first RSSM agent, jumps to 650. DreamerV1 hits 823, "
                 "and DreamerV3 reaches 853. The world model approach "
                 "consistently outperforms model-free methods."
        ) as tracker:
            self.play(
                FadeOut(versions), FadeOut(v_arrows),
                run_time=FAST_ANIM,
            )

            table = ComparisonTable(
                headers=["Method", "Type", "Mean Score"],
                rows=[
                    ["D4PG",       "Model-free",  "274"],
                    ["SAC",        "Model-free",  "437"],
                    ["PlaNet",     "World model", "650"],
                    ["DreamerV1",  "World model", "823"],
                    ["DreamerV3",  "World model", "853"],
                ],
                row_colors=[
                    SLATE, SLATE, COLOR_SSM, COLOR_FILTER_TF,
                    COLOR_HIGHLIGHT,
                ],
                title="DMControl Suite (mean normalized score)",
                highlight_best=[2],
                width=9.0,
                font_size=20,
            )
            table.next_to(title, DOWN, buff=0.6)

            self.play(FadeIn(table.bg), run_time=FAST_ANIM)
            for anim in table.animate_rows():
                self.play(anim, run_time=0.4)

        self.wait(PAUSE_MEDIUM)

        # ── Sample efficiency insight ─────────────────────────────────
        with self.voiceover(
            text="The key advantage isn't just final performance — it's "
                 "sample efficiency. Dreamer learns good policies with "
                 "far fewer environment interactions than model-free "
                 "methods, because it reuses experience through imagination. "
                 "Every real experience generates thousands of imagined ones."
        ) as tracker:
            efficiency = Text(
                "1 real step -> 1000s of imagined training steps",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            )
            efficiency.to_edge(DOWN, buff=0.4)
            self.play(FadeIn(efficiency, scale=0.9), run_time=NORMAL_ANIM)

        note = make_observation_note(
            "Hafner et al. (2020, ICLR): DreamerV1.\n"
            "Hafner et al. (2023, JMLR): DreamerV3 across diverse domains."
        )
        self.play(FadeIn(note), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
