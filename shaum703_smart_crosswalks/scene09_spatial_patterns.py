"""SHAUM703, Scene 9: Spatial Patterns.

Simulated pedestrian heatmap and trajectory cluster visualization showing
how tracking data reveals desire lines and informs crosswalk placement.

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
from shaum703_smart_crosswalks.data import *


class SceneSpatialPatterns(VoiceoverScene, MovingCameraScene):
    """Spatial Patterns: heatmaps and trajectory clusters from tracking data."""

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ──────────────────────────────────────────────────────
        title = Text(
            "Spatial Patterns",
            color=COLOR_TEXT, font_size=TITLE_FONT_SIZE,
        )
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text="Tracking pedestrians is not just about counting them. "
                 "The real power is understanding where they walk — their "
                 "spatial patterns."
        ) as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Simulated heatmap ──────────────────────────────────────────
        n_cols, n_rows = 12, 8
        cell_size = 0.55
        grid_origin = np.array([-n_cols * cell_size / 2, -n_rows * cell_size / 2 - 0.2, 0])

        # Density pattern: high at center (crossing), edges (sidewalks)
        np.random.seed(42)
        density = np.zeros((n_rows, n_cols))
        for r in range(n_rows):
            for c in range(n_cols):
                # Crossing zone: center band (rows 3-4)
                cross_dist = abs(r - 3.5) / 4.0
                cross_val = max(0, 1.0 - cross_dist)
                # Sidewalk edges (cols 0-1 and cols 10-11)
                edge_val = 0.0
                if c <= 1 or c >= n_cols - 2:
                    edge_val = 0.6
                # Combine
                density[r, c] = min(1.0, cross_val * 0.8 + edge_val * 0.5 + np.random.uniform(0, 0.15))

        def density_to_color(val):
            """Interpolate from dark blue (low) to red (high)."""
            val = np.clip(val, 0, 1)
            if val < 0.5:
                t = val / 0.5
                r_c = int(30 + t * 40)
                g_c = int(30 + t * 80)
                b_c = int(120 + t * 60)
            else:
                t = (val - 0.5) / 0.5
                r_c = int(70 + t * 160)
                g_c = int(110 - t * 80)
                b_c = int(180 - t * 150)
            return f"#{r_c:02x}{g_c:02x}{b_c:02x}"

        heatmap_cells = VGroup()
        for r in range(n_rows):
            for c in range(n_cols):
                color = density_to_color(density[r, c])
                cell = Square(
                    side_length=cell_size,
                    color=color, fill_color=color,
                    fill_opacity=0.8, stroke_width=0.3,
                    stroke_color=SLATE,
                )
                cell.move_to(grid_origin + np.array([
                    (c + 0.5) * cell_size,
                    (n_rows - r - 0.5) * cell_size,
                    0,
                ]))
                heatmap_cells.add(cell)

        hm_label = Text(
            "Pedestrian Density",
            color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
        )
        hm_label.next_to(heatmap_cells, UP, buff=0.3)

        # Color bar legend
        legend_colors = [density_to_color(v) for v in np.linspace(0, 1, 6)]
        legend_cells = VGroup()
        for lc in legend_colors:
            sq = Square(
                side_length=0.3, color=lc, fill_color=lc,
                fill_opacity=0.9, stroke_width=0.5,
            )
            legend_cells.add(sq)
        legend_cells.arrange(RIGHT, buff=0.02)
        low_lbl = Text("Low", color=SLATE, font_size=CHART_LABEL_FONT_SIZE)
        high_lbl = Text("High", color=SLATE, font_size=CHART_LABEL_FONT_SIZE)
        low_lbl.next_to(legend_cells, LEFT, buff=0.15)
        high_lbl.next_to(legend_cells, RIGHT, buff=0.15)
        legend = VGroup(low_lbl, legend_cells, high_lbl)
        legend.next_to(heatmap_cells, DOWN, buff=0.35)

        with self.voiceover(
            text="When we aggregate thousands of tracked positions, a "
                 "heatmap emerges. The hottest zones cluster at the actual "
                 "crossing area and along sidewalk edges — exactly where "
                 "pedestrians accumulate before crossing."
        ) as tracker:
            self.play(
                FadeIn(hm_label),
                LaggedStart(
                    *[FadeIn(cell, scale=0.7) for cell in heatmap_cells],
                    lag_ratio=0.005,
                ),
                run_time=SLOW_ANIM,
            )
            self.play(FadeIn(legend), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # ── Trajectory clusters ────────────────────────────────────────
        self.play(
            FadeOut(hm_label), FadeOut(legend),
            heatmap_cells.animate.set_opacity(0.25),
            run_time=FAST_ANIM,
        )

        cluster_label = Text(
            "Trajectory Clusters",
            color=COLOR_TEXT, font_size=SMALL_FONT_SIZE,
        )
        cluster_label.next_to(heatmap_cells, UP, buff=0.3)

        # Cluster 1: Straight across (green)
        c1_points = [
            grid_origin + np.array([0.5 * cell_size, 3.5 * cell_size, 0]),
            grid_origin + np.array([3.0 * cell_size, 4.0 * cell_size, 0]),
            grid_origin + np.array([6.0 * cell_size, 4.2 * cell_size, 0]),
            grid_origin + np.array([9.0 * cell_size, 4.0 * cell_size, 0]),
            grid_origin + np.array([11.5 * cell_size, 3.8 * cell_size, 0]),
        ]
        curve1 = CubicBezier(
            c1_points[0], c1_points[1], c1_points[3], c1_points[4],
        ).set_color(TEAL).set_stroke(width=3.5)

        # Cluster 2: Angled toward bus stop (blue)
        c2_points = [
            grid_origin + np.array([0.5 * cell_size, 5.5 * cell_size, 0]),
            grid_origin + np.array([3.0 * cell_size, 5.0 * cell_size, 0]),
            grid_origin + np.array([8.0 * cell_size, 2.5 * cell_size, 0]),
            grid_origin + np.array([11.5 * cell_size, 1.5 * cell_size, 0]),
        ]
        curve2 = CubicBezier(
            c2_points[0], c2_points[1], c2_points[2], c2_points[3],
        ).set_color(COLOR_MEASUREMENT).set_stroke(width=3.5)

        # Cluster 3: Diagonal shortcut (orange/red)
        c3_points = [
            grid_origin + np.array([1.0 * cell_size, 7.0 * cell_size, 0]),
            grid_origin + np.array([4.0 * cell_size, 6.0 * cell_size, 0]),
            grid_origin + np.array([8.0 * cell_size, 5.0 * cell_size, 0]),
            grid_origin + np.array([11.0 * cell_size, 6.5 * cell_size, 0]),
        ]
        curve3 = CubicBezier(
            c3_points[0], c3_points[1], c3_points[2], c3_points[3],
        ).set_color(COLOR_FILTER_EKF).set_stroke(width=3.5)

        cluster_names = VGroup(
            Text("Straight crossing", color=TEAL, font_size=CHART_LABEL_FONT_SIZE),
            Text("Angled to bus stop", color=COLOR_MEASUREMENT, font_size=CHART_LABEL_FONT_SIZE),
            Text("Diagonal shortcut", color=COLOR_FILTER_EKF, font_size=CHART_LABEL_FONT_SIZE),
        ).arrange(RIGHT, buff=0.6)
        cluster_names.next_to(heatmap_cells, DOWN, buff=0.35)

        with self.voiceover(
            text="K-means clustering on the trajectories reveals three "
                 "distinct crossing behaviors. The straight path across the "
                 "road. An angled path toward the bus stop. And a diagonal "
                 "shortcut that ignores the marked crossing entirely. These "
                 "are desire lines — the paths people actually take."
        ) as tracker:
            self.play(FadeIn(cluster_label), run_time=FAST_ANIM)
            self.play(Create(curve1), run_time=NORMAL_ANIM)
            self.play(Create(curve2), run_time=NORMAL_ANIM)
            self.play(Create(curve3), run_time=NORMAL_ANIM)
            self.play(FadeIn(cluster_names, shift=UP * 0.1), run_time=FAST_ANIM)
            self.wait(PAUSE_LONG)

        # ── Key takeaway ──────────────────────────────────────────────
        takeaway = Text(
            "Put crosswalks where people cross",
            color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
        )
        takeaway.to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="The insight is simple but powerful: put crosswalks where "
                 "people actually cross, not where engineers assume they "
                 "should. Tracking data transforms guesswork into evidence."
        ) as tracker:
            self.play(FadeIn(takeaway, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Fade out ───────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
