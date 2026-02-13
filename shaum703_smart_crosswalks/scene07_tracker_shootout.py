"""SHAUM703, Scene 7: Tracker Shootout.

Data: TRACKING_METRICS from data.py

Comparison table of ByteTrack, OC-SORT, and StrongSORT tracker performance
at Site 1 (Bosch Camera), then animated improvement at Site 2.

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
from shaum703_smart_crosswalks.data import TRACKING_METRICS


class SceneTrackerShootout(VoiceoverScene, MovingCameraScene):
    """Tracker Shootout: comparing ByteTrack, OC-SORT, and StrongSORT."""

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        title = Text(
            "Tracker Shootout",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text="Let's put three state-of-the-art multi-object trackers "
                 "head to head. We evaluated ByteTrack, OC-SORT, and "
                 "StrongSORT on real crosswalk footage."
        ) as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Build comparison table for Site 1 / Bosch ──────────────────
        rows = TRACKING_METRICS["rows"]
        # Site 1, Bosch rows: indices 0, 1, 2
        s1_byte = rows[0]   # ByteTrack
        s1_oc = rows[1]     # OC-SORT
        s1_strong = rows[2] # StrongSORT

        col_labels = ["Tracker", "HOTA", "FP", "FN", "IDSw"]
        col_widths = [2.4, 1.5, 1.8, 1.8, 1.5]
        col_x = []
        x_acc = -4.5
        for w in col_widths:
            col_x.append(x_acc + w / 2)
            x_acc += w

        def make_cell(text_str, x, y, color=COLOR_TEXT, font_size=SMALL_FONT_SIZE):
            t = Text(str(text_str), color=color, font_size=font_size)
            t.move_to(np.array([x, y, 0]))
            return t

        table_top = 1.8
        row_h = 0.55

        # Header row
        header_cells = VGroup()
        for i, lab in enumerate(col_labels):
            c = make_cell(lab, col_x[i], table_top, color=TEAL, font_size=BODY_FONT_SIZE - 4)
            header_cells.add(c)

        header_line = Line(
            start=np.array([-5.5, table_top - 0.3, 0]),
            end=np.array([5.0, table_top - 0.3, 0]),
            color=SLATE, stroke_width=1.5,
        )

        # Data rows
        data_rows_info = [
            ("ByteTrack", f"{s1_byte[3]:.3f}", f"{s1_byte[6]:,}", f"{s1_byte[7]:,}", f"{s1_byte[8]:,}"),
            ("OC-SORT", f"{s1_oc[3]:.3f}", f"{s1_oc[6]:,}", f"{s1_oc[7]:,}", f"{s1_oc[8]:,}"),
            ("StrongSORT", f"{s1_strong[3]:.3f}", f"{s1_strong[6]:,}", f"{s1_strong[7]:,}", f"{s1_strong[8]:,}"),
        ]

        # Highlight rules: ByteTrack HOTA in gold, OC-SORT FN in red
        highlight_map = {
            (0, 1): COLOR_HIGHLIGHT,   # ByteTrack HOTA
            (1, 3): COLOR_PREDICTION,  # OC-SORT FN
        }

        data_cells = VGroup()
        cell_refs = {}  # (row, col) -> Text mobject for later animation
        row_lines = VGroup()
        for r, row_data in enumerate(data_rows_info):
            y = table_top - (r + 1) * row_h - 0.15
            for c_idx, val in enumerate(row_data):
                color = highlight_map.get((r, c_idx), COLOR_TEXT)
                cell = make_cell(val, col_x[c_idx], y, color=color)
                data_cells.add(cell)
                cell_refs[(r, c_idx)] = cell
            sep = Line(
                start=np.array([-5.5, y - 0.25, 0]),
                end=np.array([5.0, y - 0.25, 0]),
                color=SLATE, stroke_width=0.8,
            )
            row_lines.add(sep)

        site_label = Text(
            "Site 1, Bosch Camera",
            color=SLATE, font_size=CHART_LABEL_FONT_SIZE,
        )
        site_label.next_to(row_lines, DOWN, buff=0.35)

        with self.voiceover(
            text="At Site 1, the busy UMD dining hall crossing, ByteTrack "
                 "dominates with a HOTA score of 0.953 — near-perfect recall, "
                 "though with 14,900 false positives. OC-SORT is ultra "
                 "conservative: only 1,500 false positives, but it misses "
                 "over 60,000 detections. StrongSORT sits in between, "
                 "balancing precision and recall."
        ) as tracker:
            self.play(
                FadeIn(header_cells), Create(header_line),
                run_time=NORMAL_ANIM,
            )
            for r in range(3):
                row_cells = VGroup(*[cell_refs[(r, c)] for c in range(5)])
                self.play(
                    FadeIn(row_cells, shift=UP * 0.1),
                    Create(row_lines[r]),
                    run_time=FAST_ANIM,
                )
            self.play(FadeIn(site_label), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # ── Animate Site 2 improvement ─────────────────────────────────
        s2_byte = rows[6]   # Site 2 Bosch ByteTrack
        s2_oc = rows[7]     # Site 2 Bosch OC-SORT
        s2_strong = rows[8] # Site 2 Bosch StrongSORT

        new_hota = [
            (0, f"{s2_byte[3]:.3f}"),
            (1, f"{s2_oc[3]:.3f}"),
            (2, f"{s2_strong[3]:.3f}"),
        ]
        new_fp = [
            (0, f"{s2_byte[6]:,}"),
            (1, f"{s2_oc[6]:,}"),
            (2, f"{s2_strong[6]:,}"),
        ]

        new_site_label = Text(
            "Site 2, Bosch Camera",
            color=TEAL, font_size=CHART_LABEL_FONT_SIZE,
        )
        new_site_label.move_to(site_label)

        with self.voiceover(
            text="Now watch what happens at Site 2, the simpler Park Road "
                 "crossing. Every single tracker improves dramatically. "
                 "ByteTrack reaches 0.992 HOTA. OC-SORT jumps to 0.965. "
                 "And false positives drop by orders of magnitude."
        ) as tracker:
            transforms = []
            for r, val in new_hota:
                old_cell = cell_refs[(r, 1)]
                new_cell = Text(
                    val, color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE,
                ).move_to(old_cell)
                transforms.append(Transform(old_cell, new_cell))
            for r, val in new_fp:
                old_cell = cell_refs[(r, 2)]
                new_cell = Text(
                    val, color=TEAL, font_size=SMALL_FONT_SIZE,
                ).move_to(old_cell)
                transforms.append(Transform(old_cell, new_cell))
            self.play(
                *transforms,
                Transform(site_label, new_site_label),
                run_time=SLOW_ANIM,
            )
            self.wait(PAUSE_MEDIUM)

        # ── Takeaway ──────────────────────────────────────────────────
        takeaway = Text(
            "Every tracker improves at Site 2",
            color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
        )
        takeaway.to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text="The lesson? Scene complexity matters as much as the "
                 "algorithm. A simpler crossing lets even conservative "
                 "trackers excel. ByteTrack's aggressive matching strategy "
                 "wins overall, but all trackers benefit from cleaner "
                 "geometry."
        ) as tracker:
            self.play(FadeIn(takeaway, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Fade out ───────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
