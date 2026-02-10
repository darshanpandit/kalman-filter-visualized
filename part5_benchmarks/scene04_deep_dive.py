"""Part 5, Scene 4: Deep Dive — Best/worst case trajectory walkthroughs.

Data: synthetic (re-run at render time, fast)

Cherry-picks trajectories showing:
1. Worst case for KF (sharp turn) — KF fails, UKF tracks well
2. Case where PF is noisy on a linear path
3. Case where all filters overlap
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.data.generators import (
    generate_sharp_turn_trajectory,
    generate_linear_trajectory,
    generate_nonlinear_trajectory,
)
from benchmarks.configs import make_all_filters, FILTER_NAMES
from benchmarks.metrics import position_rmse
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

COLORS = {
    "KF": COLOR_FILTER_KF, "EKF": COLOR_FILTER_EKF,
    "UKF": COLOR_FILTER_UKF, "PF": COLOR_FILTER_PF,
}


def _draw_trajectory_comparison(scene, data, case_title, voiceover_text):
    """Helper: draw true path + all 4 filter paths side by side."""
    true_states = data["true_states"]
    meas = data["measurements"]
    dt = data["dt"]
    x0 = true_states[0, :4] if true_states.shape[1] >= 4 else true_states[0]

    filters = make_all_filters(dt=dt, x0=x0, pf_seed=42)

    # Scale
    all_pos = true_states[:, :2]
    scale = 4.0 / max(np.ptp(all_pos[:, 0]), np.ptp(all_pos[:, 1]), 1)
    ctr = all_pos.mean(axis=0)

    def to_s(xy):
        s = (xy - ctr) * scale
        return np.array([s[0], s[1], 0])

    # True path
    true_pts = [to_s(true_states[i, :2]) for i in range(len(true_states))]
    true_path = DashedVMobject(
        VMobject().set_points_smoothly(true_pts), num_dashes=40)
    true_path.set_color(COLOR_TRUE_PATH).set_stroke(width=1.5, opacity=0.6)

    header = Text(case_title, color=COLOR_TEXT, font_size=HEADING_FONT_SIZE)
    header.to_edge(UP, buff=0.3).set_z_index(10)

    with scene.voiceover(text=voiceover_text) as tracker:
        scene.play(Write(header), run_time=FAST_ANIM)
        scene.play(Create(true_path), run_time=NORMAL_ANIM)

        # Run and draw each filter
        error_texts = []
        for name in FILTER_NAMES:
            filt = filters[name]
            res = filt.run(meas)
            est = np.array([x[:2] for x in res["x_estimates"]])
            ts = true_states[:, :4] if true_states.shape[1] > 4 else true_states
            rmse = position_rmse(res["x_estimates"], ts)

            pts = [to_s(est[i]) for i in range(len(est))]
            path = VMobject().set_points_smoothly(pts)
            path.set_color(COLORS[name]).set_stroke(width=2.5)

            label = Text(f"{name}: {rmse:.3f}", color=COLORS[name],
                        font_size=SMALL_FONT_SIZE)
            error_texts.append(label)

            scene.play(Create(path), run_time=FAST_ANIM)

        # Error summary
        error_group = VGroup(*error_texts).arrange(
            DOWN, buff=0.08, aligned_edge=LEFT)
        error_group.to_corner(DL, buff=0.3).set_z_index(10)
        bg = SurroundingRectangle(error_group, color=SLATE,
                                   fill_color=BG_COLOR, fill_opacity=0.9, buff=0.1)
        scene.play(FadeIn(bg), FadeIn(error_group), run_time=FAST_ANIM)
        scene.wait(PAUSE_LONG)

    scene.play(*[FadeOut(mob) for mob in scene.mobjects], run_time=NORMAL_ANIM)


class SceneDeepDive(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Case 1: Sharp turns — KF worst case ──────────────────────────
        data1 = generate_sharp_turn_trajectory(
            n_steps=60, dt=0.5, seed=42,
        )
        _draw_trajectory_comparison(
            self, data1,
            "Case 1: Sharp Turns",
            "First, the worst case for the linear KF: sharp ninety-degree turns. "
            "The KF assumes straight-line motion and overshoots every corner. "
            "The EKF and UKF recover faster, and the Particle Filter adapts naturally.",
        )

        # ── Case 2: Linear path — PF weakness ───────────────────────────
        data2 = generate_linear_trajectory(
            n_steps=60, dt=0.5, seed=42,
        )
        _draw_trajectory_comparison(
            self, data2,
            "Case 2: Linear Path",
            "Now the opposite: a perfectly linear trajectory. "
            "Here the standard KF is optimal — it's designed exactly for this. "
            "The Particle Filter introduces slight sampling noise, "
            "paying a small accuracy penalty for its flexibility.",
        )

        # ── Case 3: Mild curvature — all overlap ────────────────────────
        data3 = generate_nonlinear_trajectory(
            n_steps=60, dt=0.5, turn_rate=0.05,
            measurement_noise_std=0.5, seed=42,
        )
        _draw_trajectory_comparison(
            self, data3,
            "Case 3: Mild Curvature",
            "With just a little curvature, all four filters produce nearly "
            "identical results. The model mismatch is too small to matter. "
            "This is the most common real-world scenario, which is why "
            "the simple KF remains so popular.",
        )

        # Closing
        with self.voiceover(text="These case studies show that no single filter dominates everywhere. The right choice depends on your problem.") as tracker:
            closing = Text(
                "No single filter dominates everywhere.",
                color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE,
            )
            closing.move_to(ORIGIN)
            self.play(FadeIn(closing, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
