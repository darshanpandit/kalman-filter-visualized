"""SHAUM703, Scene 3: Seeing in the Dark.

Data: SENSOR_SPECS from data.py

Why thermal infrared matters for nighttime pedestrian detection.
Covers Planck's Law, Wien's Law, NETD, Beer-Lambert, and a sensor
spec comparison between visible-light (Bosch) and thermal (FLIR).

Source: Cirillo, Pandit & Momeni Rad (2025). Evaluation of Smart
Pedestrian Crosswalk Technologies. MDOT SHA Research Report.
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.observation_note import make_observation_note
from shaum703_smart_crosswalks.data import SENSOR_SPECS


class SceneSeeingInDark(VoiceoverScene, MovingCameraScene):
    """Seeing in the Dark: thermal IR physics and sensor comparison."""

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ────────────────────────────────────────────────────────
        title = Text(
            "Seeing in the Dark",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text="Half of pedestrian fatalities happen at night. Visible-"
                 "light cameras struggle in darkness, even with starlight "
                 "sensors. But there's a part of the electromagnetic "
                 "spectrum where humans literally glow."
        ) as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

        # ── Split comparison: Visible vs Thermal ─────────────────────────
        vis_panel = RoundedRectangle(
            corner_radius=0.15, width=5.0, height=2.0,
            color=COLOR_MEASUREMENT, fill_color=BG_COLOR,
            fill_opacity=0.9, stroke_width=2,
        )
        vis_title = Text("Visible + Starlight", color=COLOR_MEASUREMENT,
                         font_size=SMALL_FONT_SIZE)
        vis_desc = Text("Needs ambient light", color=SLATE,
                        font_size=CHART_LABEL_FONT_SIZE)
        vis_content = VGroup(vis_title, vis_desc).arrange(DOWN, buff=0.2)
        vis_content.move_to(vis_panel)
        vis_group = VGroup(vis_panel, vis_content)

        therm_panel = RoundedRectangle(
            corner_radius=0.15, width=5.0, height=2.0,
            color=COLOR_PREDICTION, fill_color=BG_COLOR,
            fill_opacity=0.9, stroke_width=2,
        )
        therm_title = Text("Thermal Infrared", color=COLOR_PREDICTION,
                           font_size=SMALL_FONT_SIZE)
        therm_desc = Text("Sees body heat (0 lux)", color=SLATE,
                          font_size=CHART_LABEL_FONT_SIZE)
        therm_content = VGroup(therm_title, therm_desc).arrange(DOWN, buff=0.2)
        therm_content.move_to(therm_panel)
        therm_group = VGroup(therm_panel, therm_content)

        panels = VGroup(vis_group, therm_group).arrange(RIGHT, buff=0.6)
        panels.next_to(title, DOWN, buff=0.7)

        with self.voiceover(
            text="A visible-light camera, even a high-end starlight sensor, "
                 "needs some ambient illumination. A thermal infrared camera "
                 "needs none — it detects the heat radiating from a "
                 "pedestrian's body, even in total darkness."
        ) as tracker:
            self.play(FadeIn(vis_group, shift=RIGHT * 0.2), run_time=NORMAL_ANIM)
            self.play(FadeIn(therm_group, shift=LEFT * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        self.play(FadeOut(panels), run_time=NORMAL_ANIM)

        # ── Planck's Law ─────────────────────────────────────────────────
        planck_label = Text("Planck's Law", color=COLOR_HIGHLIGHT,
                            font_size=BODY_FONT_SIZE)
        planck_label.next_to(title, DOWN, buff=0.6)

        planck_eq = MathTex(
            r"L_\lambda(T) = \frac{2hc^2}{\lambda^5}"
            r"\frac{1}{e^{hc / \lambda k_B T} - 1}",
            color=COLOR_EQUATION, font_size=EQUATION_FONT_SIZE,
        )
        planck_eq.next_to(planck_label, DOWN, buff=0.4)

        planck_note = Text(
            "Every warm object radiates light",
            color=SLATE, font_size=SMALL_FONT_SIZE,
        )
        planck_note.next_to(planck_eq, DOWN, buff=0.35)

        with self.voiceover(
            text="The physics is beautiful. Planck's Law tells us that every "
                 "object above absolute zero emits electromagnetic radiation. "
                 "The spectral radiance depends on wavelength and temperature. "
                 "Warmer objects radiate more, and at shorter wavelengths."
        ) as tracker:
            self.play(FadeIn(planck_label), run_time=FAST_ANIM)
            self.play(Write(planck_eq), run_time=SLOW_ANIM)
            self.play(FadeIn(planck_note), run_time=FAST_ANIM)
            self.wait(PAUSE_MEDIUM)

        self.play(
            FadeOut(planck_label), FadeOut(planck_eq), FadeOut(planck_note),
            run_time=NORMAL_ANIM,
        )

        # ── Wien's Displacement Law ──────────────────────────────────────
        wien_label = Text("Wien's Displacement Law", color=COLOR_HIGHLIGHT,
                          font_size=BODY_FONT_SIZE)
        wien_label.next_to(title, DOWN, buff=0.6)

        wien_eq = MathTex(
            r"\lambda_{\text{peak}} = \frac{2898}{T} \; \mu\text{m}",
            color=COLOR_EQUATION, font_size=EQUATION_FONT_SIZE,
        )
        wien_eq.next_to(wien_label, DOWN, buff=0.4)

        wien_result = Text(
            "Human body (310 K) -> 9.3 um",
            color=COLOR_PREDICTION, font_size=BODY_FONT_SIZE,
        )
        wien_result.next_to(wien_eq, DOWN, buff=0.4)

        with self.voiceover(
            text="Wien's Displacement Law tells us the peak wavelength. "
                 "For a human body at three hundred and ten Kelvin, the peak "
                 "emission is at nine point three micrometers — squarely in "
                 "the long-wave infrared band. This is exactly where thermal "
                 "cameras are designed to operate."
        ) as tracker:
            self.play(FadeIn(wien_label), run_time=FAST_ANIM)
            self.play(Write(wien_eq), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(wien_result, shift=UP * 0.15), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        self.play(
            FadeOut(wien_label), FadeOut(wien_eq), FadeOut(wien_result),
            run_time=NORMAL_ANIM,
        )

        # ── NETD definition ──────────────────────────────────────────────
        netd_label = Text("Noise Equivalent Temperature Difference",
                          color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE)
        netd_label.next_to(title, DOWN, buff=0.6)

        netd_eq = MathTex(
            r"\text{NETD} = \frac{V_{\text{noise}}}{\partial V / \partial T}",
            color=COLOR_EQUATION, font_size=EQUATION_FONT_SIZE,
        )
        netd_eq.next_to(netd_label, DOWN, buff=0.4)

        netd_note = Text(
            "FLIR: < 50 mK (detects 0.05 C change)",
            color=COLOR_PREDICTION, font_size=SMALL_FONT_SIZE,
        )
        netd_note.next_to(netd_eq, DOWN, buff=0.35)

        with self.voiceover(
            text="How sensitive are these cameras? The key metric is NETD — "
                 "Noise Equivalent Temperature Difference. It's the ratio of "
                 "sensor noise to thermal sensitivity. The FLIR camera in "
                 "our study has an NETD below fifty millikelvin — it can "
                 "detect a temperature change of just five hundredths of a "
                 "degree Celsius."
        ) as tracker:
            self.play(FadeIn(netd_label), run_time=FAST_ANIM)
            self.play(Write(netd_eq), run_time=NORMAL_ANIM)
            self.play(FadeIn(netd_note), run_time=FAST_ANIM)
            self.wait(PAUSE_MEDIUM)

        self.play(
            FadeOut(netd_label), FadeOut(netd_eq), FadeOut(netd_note),
            run_time=NORMAL_ANIM,
        )

        # ── Sensor spec table ────────────────────────────────────────────
        metrics = SENSOR_SPECS["metrics"]
        bosch = SENSOR_SPECS["bosch"]
        flir = SENSOR_SPECS["flir"]

        # Build a compact 3-column table (Metric, Bosch, FLIR)
        header_row = VGroup(
            Text("Metric", color=COLOR_HIGHLIGHT, font_size=CHART_LABEL_FONT_SIZE),
            Text("Bosch (Visible)", color=COLOR_MEASUREMENT,
                 font_size=CHART_LABEL_FONT_SIZE),
            Text("FLIR (Thermal)", color=COLOR_PREDICTION,
                 font_size=CHART_LABEL_FONT_SIZE),
        )
        header_row.arrange(RIGHT, buff=1.2)

        rows = VGroup()
        # Show 4 key rows to fit in frame
        key_indices = [0, 1, 2, 3]  # Resolution, Min Illum, NETD, Det Range
        for idx in key_indices:
            row = VGroup(
                Text(metrics[idx].replace("\n", " "), color=COLOR_TEXT,
                     font_size=CHART_TICK_FONT_SIZE),
                Text(bosch[idx].replace("\n", " "), color=COLOR_MEASUREMENT,
                     font_size=CHART_TICK_FONT_SIZE),
                Text(flir[idx].replace("\n", " "), color=COLOR_PREDICTION,
                     font_size=CHART_TICK_FONT_SIZE),
            )
            row.arrange(RIGHT, buff=1.2)
            rows.add(row)

        table = VGroup(header_row, *rows).arrange(DOWN, buff=0.25)
        table.next_to(title, DOWN, buff=0.6)
        table.scale_to_fit_width(min(table.width, 11.6))

        with self.voiceover(
            text="Here's how the two sensors compare. The Bosch visible "
                 "camera has higher resolution — full HD or four K. But it "
                 "needs at least some light, with a minimum illumination of "
                 "zero point zero zero four seven lux. The FLIR thermal "
                 "camera operates at zero lux — complete darkness. It has an "
                 "NETD below fifty millikelvin, and can detect pedestrians "
                 "at eighty meters based purely on body heat."
        ) as tracker:
            self.play(FadeIn(header_row), run_time=FAST_ANIM)
            for row in rows:
                self.play(FadeIn(row, shift=RIGHT * 0.1), run_time=0.6)
            self.wait(PAUSE_LONG)

        self.play(FadeOut(table), run_time=NORMAL_ANIM)

        # ── Beer-Lambert Law ─────────────────────────────────────────────
        beer_label = Text("Atmospheric Transmission",
                          color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE)
        beer_label.next_to(title, DOWN, buff=0.6)

        beer_eq = MathTex(
            r"\tau(\lambda) = e^{-\alpha(\lambda) \cdot d}",
            color=COLOR_EQUATION, font_size=EQUATION_FONT_SIZE,
        )
        beer_eq.next_to(beer_label, DOWN, buff=0.4)

        beer_note = Text(
            "8-14 um: atmospheric transmission window",
            color=TEAL, font_size=SMALL_FONT_SIZE,
        )
        beer_note.next_to(beer_eq, DOWN, buff=0.35)

        with self.voiceover(
            text="One last piece of physics. The Beer-Lambert Law governs "
                 "how much infrared radiation is absorbed by the atmosphere. "
                 "Fortunately, there's a transmission window between eight "
                 "and fourteen micrometers — right where human bodies emit "
                 "most strongly. Nature gives us a clear channel."
        ) as tracker:
            self.play(FadeIn(beer_label), run_time=FAST_ANIM)
            self.play(Write(beer_eq), run_time=NORMAL_ANIM)
            self.play(FadeIn(beer_note), run_time=FAST_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Concluding insight ───────────────────────────────────────────
        insight = Text(
            "Thermal sees what visible cannot",
            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
        )
        insight.to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text="This is why thermal infrared is a game-changer for "
                 "pedestrian safety. When visible cameras go blind, thermal "
                 "sensors see the one thing that matters most — a warm human "
                 "body, standing at the edge of a dark road."
        ) as tracker:
            self.play(FadeIn(insight, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        citation = make_observation_note(
            "Planck, Wien, Beer-Lambert: classical physics.\n"
            "Sensor data: Cirillo, Pandit & Momeni Rad (2025)"
        )
        self.play(FadeIn(citation), run_time=FAST_ANIM)
        self.wait(PAUSE_LONG)

        # ── Fade out ─────────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
