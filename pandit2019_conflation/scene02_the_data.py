"""Scene 02: The Data — Tour of NPMRDS and HPMS datasets.

Two-voice format (Jenny = narrator, Tony = Darshan) using Azure TTS.
Shows Fig 1 (segment length histograms) and Fig 2 (AADT distributions)
to highlight how fundamentally different these two datasets are.

Voices: narrator (chat), narrator_whisper, darshan (friendly)

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
from pandit2019_conflation.data import RESULTS, STUDY_REGION, fig_path


class SceneTheData(VoiceoverScene, MovingCameraScene):
    """Beat 2 — The Data."""

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
            "The Data",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=NORMAL_ANIM)

        # ── Beat 1: What NPMRDS actually is ─────────────────────────
        self.set_speech_service(darshan)
        with self.voiceover(
            text=(
                "Let me show you what I was dealing with. The NPMRDS is "
                "built from GPS probes. Every five minutes, if a vehicle "
                "crosses a TMC segment, you get a travel time. No vehicle, "
                "no data."
            ),
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Beat 2: Fig 1 — segment length histograms ──────────────
        fig1 = ImageMobject(fig_path("fig1_segment_histograms_fullwidth.png"))
        fig1.set_width(10)
        fig1.next_to(title, DOWN, buff=0.4)

        fig1_border = SurroundingRectangle(
            fig1, color=SLATE, buff=0.05, stroke_width=1.5,
        )

        self.set_speech_service(darshan)
        with self.voiceover(
            text=(
                "Look at these segment lengths. HPMS segments peak around "
                "two hundred meters. But NPMRDS segments can be twenty "
                "kilometers long."
            ),
        ) as tracker:
            self.play(
                FadeIn(fig1), FadeIn(fig1_border),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_MEDIUM)

        # ── Narrator reacts ─────────────────────────────────────────
        self.set_speech_service(narrator)
        with self.voiceover(
            text=(
                "A hundred-to-one ratio. How do you compare a two hundred "
                "meter segment to a twenty kilometer segment?"
            ),
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Darshan: the point ──────────────────────────────────────
        self.set_speech_service(darshan)
        with self.voiceover(
            text=(
                "You don't compare them directly. That's the whole point. "
                "You need geometric similarity measures that handle "
                "different-length curves."
            ),
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Narrator: preprocessing contribution ────────────────────
        self.set_speech_service(narrator)
        with self.voiceover(
            text=(
                "So the data preprocessing alone is a research contribution. "
                "Before any matching happens, you need the right "
                "representation."
            ),
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Fade Fig 1 ──────────────────────────────────────────────
        self.play(
            FadeOut(fig1), FadeOut(fig1_border),
            run_time=FAST_ANIM,
        )

        # ── Beat 3: Fig 2 — AADT distributions ─────────────────────
        fig2 = ImageMobject(fig_path("fig2_aadt_distributions.png"))
        fig2.set_width(10)
        fig2.next_to(title, DOWN, buff=0.4)

        self.set_speech_service(darshan)
        with self.voiceover(
            text=(
                "Exactly. And here's the traffic volume. The HPMS records "
                "Average Annual Daily Traffic — AADT. But you can't use "
                "linear bins. A rural road with five hundred vehicles and "
                "a highway with two hundred thousand — ten percent means "
                "completely different things."
            ),
        ) as tracker:
            self.play(FadeIn(fig2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Darshan: exponential binning ────────────────────────────
        self.set_speech_service(darshan)
        with self.voiceover(
            text=(
                "So I designed an exponential binning scheme. Fifty bins "
                "where the width scales with magnitude. Log-transform the "
                "histogram and it becomes nearly uniform. That's how you "
                "validate the binning."
            ),
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Narrator whisper: never designed ────────────────────────
        self.set_speech_service(narrator_whisper)
        with self.voiceover(
            text="These datasets were never designed to talk to each other.",
        ) as tracker:
            self.wait(PAUSE_MEDIUM)

        # ── Fade out ────────────────────────────────────────────────
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob is not title],
            run_time=NORMAL_ANIM,
        )
        self.wait(PAUSE_MEDIUM)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=NORMAL_ANIM,
        )
