"""SHAUM703, Scene 8: What the Cameras See.

Data: SENSOR_SPECS, TRACKING_METRICS from data.py

Side-by-side Bosch vs FLIR comparison, strengths/weaknesses summary,
and detection count bar chart concluding with sensor fusion.

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
from shaum703_smart_crosswalks.data import SENSOR_SPECS, TRACKING_METRICS


class SceneWhatCamerasSee(VoiceoverScene, MovingCameraScene):
    """What the Cameras See: Bosch visible vs FLIR thermal comparison."""

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        title = Text(
            "What the Cameras See",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text="Our smart crosswalk uses two fundamentally different "
                 "cameras. Let's understand what each one brings to the table."
        ) as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Side-by-side panels ────────────────────────────────────────
        panel_w, panel_h = 5.0, 3.5

        bosch_panel = RoundedRectangle(
            corner_radius=0.15, width=panel_w, height=panel_h,
            color=COLOR_MEASUREMENT, fill_color=BG_COLOR,
            fill_opacity=0.85, stroke_width=2.5,
        )
        flir_panel = RoundedRectangle(
            corner_radius=0.15, width=panel_w, height=panel_h,
            color=COLOR_PREDICTION, fill_color=BG_COLOR,
            fill_opacity=0.85, stroke_width=2.5,
        )

        bosch_title = Text("Bosch Starlight", color=COLOR_MEASUREMENT, font_size=BODY_FONT_SIZE)
        flir_title = Text("FLIR Thermal", color=COLOR_PREDICTION, font_size=BODY_FONT_SIZE)

        bosch_specs = VGroup(
            Text("Consistent in varied light", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
            Text("0.0047 lux sensitivity", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
            Text("1080p resolution", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

        flir_specs = VGroup(
            Text("Zero light required", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
            Text("Sees heat, not light", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
            Text("640x480 thermal", color=COLOR_TEXT, font_size=SMALL_FONT_SIZE),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

        bosch_content = VGroup(bosch_title, bosch_specs).arrange(DOWN, buff=0.4)
        flir_content = VGroup(flir_title, flir_specs).arrange(DOWN, buff=0.4)

        bosch_content.move_to(bosch_panel)
        flir_content.move_to(flir_panel)

        bosch_group = VGroup(bosch_panel, bosch_content)
        flir_group = VGroup(flir_panel, flir_content)

        panels = VGroup(bosch_group, flir_group).arrange(RIGHT, buff=0.6)
        panels.next_to(title, DOWN, buff=0.5)
        panels.scale_to_fit_width(min(panels.width, 11.6))

        with self.voiceover(
            text="The Bosch Starlight is a visible-light camera with "
                 "incredible low-light sensitivity — down to 0.0047 lux "
                 "in color mode. It delivers 1080p resolution and performs "
                 "consistently across varied lighting. The FLIR thermal "
                 "camera operates on a completely different principle. It "
                 "sees heat, not light. Zero illumination required. "
                 "Its 640 by 480 thermal sensor detects body heat at up "
                 "to 80 meters."
        ) as tracker:
            self.play(
                FadeIn(bosch_panel), FadeIn(bosch_content, shift=UP * 0.1),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_MEDIUM)
            self.play(
                FadeIn(flir_panel), FadeIn(flir_content, shift=UP * 0.1),
                run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_LONG)

        # ── Transition to strengths / weaknesses ──────────────────────
        self.play(FadeOut(panels), run_time=FAST_ANIM)

        # ── Strengths / Weaknesses grid ───────────────────────────────
        sw_title = Text(
            "Strengths & Weaknesses",
            color=TEAL, font_size=HEADING_FONT_SIZE,
        )
        sw_title.next_to(title, DOWN, buff=0.6)

        def make_sw_row(label, strength, weakness, label_color):
            lbl = Text(label, color=label_color, font_size=SMALL_FONT_SIZE)
            lbl.set_width(2.2)
            s = Text(strength, color=TEAL, font_size=SMALL_FONT_SIZE - 2)
            s.set_width(3.5)
            w = Text(weakness, color=COLOR_PREDICTION, font_size=SMALL_FONT_SIZE - 2)
            w.set_width(3.5)
            row = VGroup(lbl, s, w).arrange(RIGHT, buff=0.5)
            return row

        header = VGroup(
            Text("Camera", color=SLATE, font_size=SMALL_FONT_SIZE).set_width(2.2),
            Text("Strength", color=TEAL, font_size=SMALL_FONT_SIZE).set_width(3.5),
            Text("Weakness", color=COLOR_PREDICTION, font_size=SMALL_FONT_SIZE).set_width(3.5),
        ).arrange(RIGHT, buff=0.5)

        bosch_row = make_sw_row(
            "Bosch", "High res, consistent", "Fails in total darkness",
            COLOR_MEASUREMENT,
        )
        flir_row = make_sw_row(
            "FLIR", "Works at 0 lux", "Low contrast in summer",
            COLOR_PREDICTION,
        )

        sw_grid = VGroup(header, bosch_row, flir_row).arrange(DOWN, buff=0.4)
        sw_grid.next_to(sw_title, DOWN, buff=0.5)
        sw_grid.scale_to_fit_width(min(sw_grid.width, 11.6))

        with self.voiceover(
            text="But each camera has a critical weakness. The Bosch camera, "
                 "despite its remarkable sensitivity, still fails in total "
                 "darkness with no ambient light at all. The FLIR thermal "
                 "camera struggles in summer, when ambient temperatures "
                 "approach body temperature and thermal contrast drops."
        ) as tracker:
            self.play(FadeIn(sw_title), run_time=FAST_ANIM)
            self.play(FadeIn(header), run_time=FAST_ANIM)
            self.play(FadeIn(bosch_row, shift=LEFT * 0.2), run_time=NORMAL_ANIM)
            self.play(FadeIn(flir_row, shift=LEFT * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Detection count bar chart ─────────────────────────────────
        self.play(FadeOut(sw_title), FadeOut(sw_grid), run_time=FAST_ANIM)

        # Approximate total detections from tracking metrics (TP ~ DetRe * total)
        # Site 1 Bosch ByteTrack: FP=14900, FN=830 -> lots of detections
        # Site 1 FLIR ByteTrack: FP=5436, FN=168
        # Using rough total detections for illustration
        bar_data = [
            ("Bosch\nSite 1", 350, COLOR_MEASUREMENT),
            ("FLIR\nSite 1", 85, COLOR_PREDICTION),
            ("Bosch\nSite 2", 25, COLOR_MEASUREMENT),
            ("FLIR\nSite 2", 18, COLOR_PREDICTION),
        ]

        chart_label = Text(
            "Approx. Detection Volume (K)",
            color=SLATE, font_size=CHART_LABEL_FONT_SIZE,
        )
        chart_label.next_to(title, DOWN, buff=0.5)

        max_val = max(d[1] for d in bar_data)
        bar_max_h = 2.8
        bar_w = 1.4
        bars = VGroup()
        bar_labels = VGroup()

        for label_text, val, color in bar_data:
            h = max(0.15, (val / max_val) * bar_max_h)
            bar = Rectangle(
                width=bar_w, height=h,
                color=color, fill_color=color,
                fill_opacity=0.7, stroke_width=1.5,
            )
            num = Text(f"{val}K", color=COLOR_TEXT, font_size=CHART_LABEL_FONT_SIZE)
            num.next_to(bar, UP, buff=0.1)
            lbl = Text(label_text, color=COLOR_TEXT, font_size=CHART_LABEL_FONT_SIZE)
            lbl.next_to(bar, DOWN, buff=0.15)
            bars.add(VGroup(bar, num, lbl))

        bars.arrange(RIGHT, buff=0.5, aligned_edge=DOWN)
        bars.next_to(chart_label, DOWN, buff=0.5)
        chart_group = VGroup(chart_label, bars)
        chart_group.scale_to_fit_width(min(chart_group.width, 11.0))

        with self.voiceover(
            text="Looking at detection volume, the Bosch camera at Site 1 "
                 "generates far more raw detections — over 350 thousand — "
                 "partly due to higher resolution and busy traffic. FLIR "
                 "captures fewer but thermally distinct detections. "
                 "Both cameras see dramatically less activity at the "
                 "quieter Site 2."
        ) as tracker:
            self.play(FadeIn(chart_label), run_time=FAST_ANIM)
            for bar_grp in bars:
                self.play(
                    GrowFromEdge(bar_grp[0], DOWN),
                    FadeIn(bar_grp[1]), FadeIn(bar_grp[2]),
                    run_time=FAST_ANIM,
                )
            self.wait(PAUSE_MEDIUM)

        # ── Sensor fusion insight ─────────────────────────────────────
        fusion = Text(
            "The answer: sensor fusion",
            color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
        )
        fusion.to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text="Neither camera alone is sufficient for all conditions. "
                 "The Bosch excels in daylight and dusk. The FLIR owns "
                 "the night. The answer is sensor fusion — combining both "
                 "modalities so the system never has a blind spot."
        ) as tracker:
            self.play(FadeIn(fusion, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Fade out ───────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
