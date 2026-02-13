"""SHAUM703, Scene 2: The Arms Race.

Data: CMF_DATA, YIELDING_DATA from data.py

Treatment hierarchy from passive markings to active signals, with bar
charts showing driver yielding rates and crash modification factors.

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
from shaum703_smart_crosswalks.data import CMF_DATA, YIELDING_DATA


def _make_bar_chart(
    labels: list[str],
    values: list[float],
    colors: list[str],
    chart_title: str,
    y_max: float,
    y_suffix: str = "",
    width: float = 8.0,
    height: float = 3.0,
) -> tuple[VGroup, VGroup, VGroup, Text]:
    """Build a manual bar chart and return (bars, bar_labels, value_labels, title)."""
    n = len(labels)
    bar_width = width / (2 * n)
    spacing = width / n

    bars = VGroup()
    bar_labels = VGroup()
    value_labels = VGroup()

    for i, (label, val, col) in enumerate(zip(labels, values, colors)):
        bar_h = (val / y_max) * height if y_max > 0 else 0
        bar_h = max(bar_h, 0.05)
        bar = Rectangle(
            width=bar_width, height=bar_h,
            color=col, fill_color=col, fill_opacity=0.8,
            stroke_width=1.5,
        )
        x_pos = -width / 2 + spacing * (i + 0.5)
        bar.move_to(np.array([x_pos, -height / 2 + bar_h / 2, 0]))
        bars.add(bar)

        bl = Text(label, color=COLOR_TEXT, font_size=CHART_LABEL_FONT_SIZE)
        bl.next_to(bar, DOWN, buff=0.15)
        bar_labels.add(bl)

        vl = Text(
            f"{val:.0f}{y_suffix}" if val == int(val) else f"{val}{y_suffix}",
            color=col, font_size=CHART_LABEL_FONT_SIZE,
        )
        vl.next_to(bar, UP, buff=0.1)
        value_labels.add(vl)

    chart_title_mob = Text(
        chart_title, color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
    )
    all_content = VGroup(bars, bar_labels, value_labels)
    chart_title_mob.next_to(all_content, UP, buff=0.25)

    return bars, bar_labels, value_labels, chart_title_mob


class SceneArmsRace(VoiceoverScene, MovingCameraScene):
    """The Arms Race: escalating treatments and their effectiveness."""

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ────────────────────────────────────────────────────────
        title = Text(
            "The Arms Race",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text="How do transportation agencies fight back? Through an "
                 "escalating arsenal of crosswalk treatments — each more "
                 "aggressive than the last."
        ) as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

        # ── Treatment hierarchy ──────────────────────────────────────────
        treatments = ["Marked\nOnly", "RRFB", "PHB", "MPS"]
        descriptions = ["Paint", "Flashing\nBeacons", "Hybrid\nBeacon", "Midblock\nSignal"]
        box_colors = [SLATE, COLOR_MEASUREMENT, COLOR_HIGHLIGHT, COLOR_PREDICTION]

        boxes = VGroup()
        for treat, desc, col in zip(treatments, descriptions, box_colors):
            bg = RoundedRectangle(
                corner_radius=0.12, width=2.3, height=1.6,
                color=col, fill_color=BG_COLOR,
                fill_opacity=0.9, stroke_width=2.5,
            )
            t_name = Text(treat, color=col, font_size=SMALL_FONT_SIZE)
            t_desc = Text(desc, color=SLATE, font_size=CHART_LABEL_FONT_SIZE)
            content = VGroup(t_name, t_desc).arrange(DOWN, buff=0.15)
            content.move_to(bg)
            boxes.add(VGroup(bg, content))

        boxes.arrange(RIGHT, buff=0.6)
        boxes.next_to(title, DOWN, buff=0.7)
        boxes.scale_to_fit_width(min(boxes.width, 11.6))

        arrows = VGroup()
        for i in range(len(boxes) - 1):
            arrow = Arrow(
                boxes[i].get_right(), boxes[i + 1].get_left(),
                buff=0.1, color=COLOR_TEXT, stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            )
            arrows.add(arrow)

        escalation_label = Text(
            "Increasing intensity",
            color=SLATE, font_size=CHART_LABEL_FONT_SIZE,
        )
        escalation_label.next_to(arrows, UP, buff=0.08)

        with self.voiceover(
            text="At the bottom, we have simple marked crosswalks — just "
                 "paint on the road. Next come Rectangular Rapid Flashing "
                 "Beacons, or RRFBs, which use bright LED flashers. Then "
                 "Pedestrian Hybrid Beacons, which add a red signal phase. "
                 "And finally, full Midblock Pedestrian Signals with "
                 "standard traffic lights."
        ) as tracker:
            for i, box in enumerate(boxes):
                self.play(FadeIn(box, shift=UP * 0.15), run_time=FAST_ANIM)
                if i < len(arrows):
                    self.play(GrowArrow(arrows[i]), run_time=0.3)
            self.play(FadeIn(escalation_label), run_time=FAST_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Transition ───────────────────────────────────────────────────
        self.play(
            FadeOut(boxes), FadeOut(arrows), FadeOut(escalation_label),
            run_time=NORMAL_ANIM,
        )

        # ── Yielding rate bar chart ──────────────────────────────────────
        y_labels = YIELDING_DATA["treatments"]
        y_vals = YIELDING_DATA["yielding_mid"]
        y_colors = [SLATE, COLOR_MEASUREMENT, COLOR_HIGHLIGHT]

        bars_y, bl_y, vl_y, title_y = _make_bar_chart(
            labels=y_labels, values=y_vals, colors=y_colors,
            chart_title="Driver Yielding Rate (%)",
            y_max=100, y_suffix="%", width=7.0, height=2.8,
        )
        yield_group = VGroup(title_y, bars_y, bl_y, vl_y)
        yield_group.next_to(title, DOWN, buff=0.6)

        # Animated bars: start at zero height
        bar_targets_y = []
        for bar in bars_y:
            target = bar.copy()
            bar.stretch(0.001, dim=1, about_edge=DOWN)
            bar_targets_y.append(target)

        with self.voiceover(
            text="How effective are these treatments? Let's look at driver "
                 "yielding rates — the percentage of drivers who actually "
                 "stop for a pedestrian. A plain crosswalk gets only about "
                 "fifteen percent compliance. An RRFB jumps to eighty-five "
                 "percent. And a PHB reaches over ninety-seven percent."
        ) as tracker:
            self.play(FadeIn(title_y), run_time=FAST_ANIM)
            self.play(
                *[FadeIn(bl) for bl in bl_y],
                run_time=FAST_ANIM,
            )
            for bar, target, vl in zip(bars_y, bar_targets_y, vl_y):
                self.play(
                    Transform(bar, target),
                    FadeIn(vl),
                    run_time=NORMAL_ANIM,
                )
            self.wait(PAUSE_MEDIUM)

        # ── Transition ───────────────────────────────────────────────────
        self.play(FadeOut(yield_group), run_time=NORMAL_ANIM)

        # ── CMF bar chart ────────────────────────────────────────────────
        cmf_labels = CMF_DATA["treatments"][1:]  # skip "Marked Only"
        cmf_vals = CMF_DATA["cmf_ped"][1:]
        # Highlight PHB as best (lowest CMF)
        cmf_colors = [COLOR_MEASUREMENT, COLOR_HIGHLIGHT, COLOR_PREDICTION]

        bars_c, bl_c, vl_c, title_c = _make_bar_chart(
            labels=cmf_labels, values=cmf_vals, colors=cmf_colors,
            chart_title="Crash Modification Factor (lower = safer)",
            y_max=1.0, width=7.0, height=2.8,
        )
        cmf_group = VGroup(title_c, bars_c, bl_c, vl_c)
        cmf_group.next_to(title, DOWN, buff=0.6)

        bar_targets_c = []
        for bar in bars_c:
            target = bar.copy()
            bar.stretch(0.001, dim=1, about_edge=DOWN)
            bar_targets_c.append(target)

        with self.voiceover(
            text="Now the crash modification factors — a CMF below one means "
                 "fewer crashes than an uncontrolled crossing. RRFBs achieve "
                 "a CMF of zero point five three — a forty-seven percent "
                 "crash reduction. PHBs do even better at zero point four "
                 "five — a fifty-five percent reduction. Midblock signals "
                 "come in at zero point five five. The PHB offers the best "
                 "safety return."
        ) as tracker:
            self.play(FadeIn(title_c), run_time=FAST_ANIM)
            self.play(
                *[FadeIn(bl) for bl in bl_c],
                run_time=FAST_ANIM,
            )
            for bar, target, vl in zip(bars_c, bar_targets_c, vl_c):
                self.play(
                    Transform(bar, target),
                    FadeIn(vl),
                    run_time=NORMAL_ANIM,
                )
            self.wait(PAUSE_MEDIUM)

        # ── Takeaway ─────────────────────────────────────────────────────
        takeaway = Text(
            "PHB: best safety per dollar",
            color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE,
        )
        takeaway.to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text="The Pedestrian Hybrid Beacon hits the sweet spot — the "
                 "highest yielding rate and the best crash reduction. But "
                 "even the best signal can't help if it can't see the "
                 "pedestrian. That's where smart sensing comes in."
        ) as tracker:
            self.play(FadeIn(takeaway, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Fade out ─────────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
