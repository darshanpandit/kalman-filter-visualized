"""Scene 4: The Study — Two-site experimental setup and hardware.

Data: SITE_INFO and SENSOR_SPECS from data.py

Introduces the two data collection sites (UMD Campus and Park Road),
the sensor hardware stack (Bosch Starlight, FLIR Thermal, Jetson Orin),
and summarises 26 hours of multi-modal footage.

Reference:
- Cirillo, Pandit & Momeni Rad (2025). MDOT SHA Research Report.
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from shaum703_smart_crosswalks.data import SITE_INFO, SENSOR_SPECS


class SceneTheStudy(VoiceoverScene, MovingCameraScene):
    """Two-site experimental setup, sensor hardware, and data summary.

    Visual: Side-by-side site panels, hardware stack, total data callout.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ─────────────────────────────────────────────────────
        title = Text(
            "The Study",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text="To evaluate smart crosswalk technologies under real "
                 "conditions, we deployed sensors at two sites with very "
                 "different traffic patterns."
        ) as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Site info panels ──────────────────────────────────────────
        def make_site_panel(key: str, border_color: str) -> VGroup:
            info = SITE_INFO[key]
            name_label = Text(
                info["name"].replace("\n", " "),
                color=COLOR_TEXT, font_size=HEADING_FONT_SIZE,
            )
            duration_label = Text(
                f"{info['duration_hrs']} hours",
                color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
            )
            window_label = Text(
                info["time_window"],
                color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
            )
            cond_label = Text(
                info["conditions"],
                color=SLATE, font_size=SMALL_FONT_SIZE,
            )
            content = VGroup(
                name_label, duration_label, window_label, cond_label,
            ).arrange(DOWN, buff=0.2)

            panel = RoundedRectangle(
                corner_radius=0.15, width=5.0, height=3.0,
                stroke_color=border_color, stroke_width=2,
                fill_color=DARK_SLATE, fill_opacity=0.6,
            )
            content.move_to(panel)
            return VGroup(panel, content)

        site1_panel = make_site_panel("site1", COLOR_MEASUREMENT)
        site2_panel = make_site_panel("site2", TEAL)

        panels = VGroup(site1_panel, site2_panel).arrange(RIGHT, buff=0.6)
        panels.next_to(title, DOWN, buff=0.5)
        panels.scale_to_fit_width(min(panels.width, 11.5))

        with self.voiceover(
            text="Site one was at the University of Maryland campus, near "
                 "a dining hall. We recorded nine hours from two AM to "
                 "eleven AM, capturing the overnight lull and the morning "
                 "rush. Site two was on Park Road, where we captured "
                 "seventeen hours from one PM through five AM — a full "
                 "day-night cycle."
        ) as tracker:
            self.play(FadeIn(site1_panel, shift=RIGHT * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)
            self.play(FadeIn(site2_panel, shift=LEFT * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        self.wait(PAUSE_SHORT)

        # ── Transition: fade panels, keep title ──────────────────────
        with self.voiceover(
            text="Both sites were instrumented with the same hardware "
                 "stack, purpose-built for all-weather pedestrian detection."
        ) as tracker:
            self.play(FadeOut(panels), run_time=FAST_ANIM)
            self.wait(PAUSE_SHORT)

        # ── Hardware stack diagram ────────────────────────────────────
        hw_specs = [
            ("Bosch Starlight", COLOR_MEASUREMENT, "Visible / 4K / 0.005 lux"),
            ("FLIR Thermal", SWISS_RED, "Thermal / VGA / 0 lux"),
            ("Jetson Orin AGX", TEAL, "Edge AI / 275 TOPS"),
        ]

        hw_boxes = VGroup()
        for label_text, color, detail in hw_specs:
            box = RoundedRectangle(
                corner_radius=0.12, width=4.0, height=1.0,
                stroke_color=color, stroke_width=2.5,
                fill_color=DARK_SLATE, fill_opacity=0.6,
            )
            label = Text(label_text, color=color, font_size=BODY_FONT_SIZE)
            detail_text = Text(detail, color=SLATE, font_size=SMALL_FONT_SIZE)
            inner = VGroup(label, detail_text).arrange(DOWN, buff=0.08)
            inner.move_to(box)
            hw_boxes.add(VGroup(box, inner))

        hw_boxes.arrange(DOWN, buff=0.35)
        hw_boxes.next_to(title, DOWN, buff=0.6)

        # Connecting lines between boxes
        connectors = VGroup()
        for i in range(len(hw_boxes) - 1):
            start_pt = hw_boxes[i].get_bottom()
            end_pt = hw_boxes[i + 1].get_top()
            line = Line(
                start_pt, end_pt,
                stroke_color=SLATE, stroke_width=2,
            )
            connectors.add(line)

        with self.voiceover(
            text="The Bosch Starlight camera provides high-resolution "
                 "visible imagery down to near-total darkness. The FLIR "
                 "thermal camera works in zero lux — pure infrared. Both "
                 "feed into an NVIDIA Jetson Orin AGX, an edge compute "
                 "module with 275 tera-operations per second."
        ) as tracker:
            for i, box_group in enumerate(hw_boxes):
                self.play(FadeIn(box_group, shift=DOWN * 0.2), run_time=0.6)
                if i < len(connectors):
                    self.play(Create(connectors[i]), run_time=0.3)
            self.wait(PAUSE_MEDIUM)

        self.wait(PAUSE_SHORT)

        # ── Total data callout ────────────────────────────────────────
        with self.voiceover(
            text="In total, we collected twenty-six hours of synchronized "
                 "multi-modal data — visible and thermal video running "
                 "side by side — ready for detection and tracking."
        ) as tracker:
            total_group = VGroup(hw_boxes, connectors)
            self.play(total_group.animate.scale(0.7).shift(LEFT * 2.5),
                      run_time=NORMAL_ANIM)

            callout_text = Text(
                "26 hours\nmulti-modal data",
                color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
            )
            callout_border = RoundedRectangle(
                corner_radius=0.15, width=4.5, height=2.0,
                stroke_color=COLOR_HIGHLIGHT, stroke_width=2.5,
                fill_color=DARK_SLATE, fill_opacity=0.5,
            )
            callout_text.move_to(callout_border)
            callout = VGroup(callout_border, callout_text)
            callout.to_edge(RIGHT, buff=0.8).shift(DOWN * 0.3)

            self.play(FadeIn(callout, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Fade out ──────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
