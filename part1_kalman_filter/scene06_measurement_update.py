"""Scene 6: Measurement Update

Derives and visualizes the Kalman measurement update step:
  Innovation: ỹ = z - H · x̂_k⁻
  Kalman Gain: K = P_k⁻ · H^T · (H · P_k⁻ · H^T + R)^{-1}
  State update: x̂_k = x̂_k⁻ + K · ỹ
  Covariance update: P_k = (I - K·H) · P_k⁻

Shows predicted ellipse (red) and measurement (blue) merging into posterior (gold).
"""

from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from kalman_manim.mobjects.gaussian_ellipse import GaussianEllipse
from kalman_manim.mobjects.state_space import StateSpace


class SceneMeasurementUpdate(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────────
        title = Text("Measurement Update", color=COLOR_TEXT,
                      font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Equations panel (right side) ────────────────────────────────
        eq_meas_model = MathTex(
            r"\mathbf{z}_k = \mathbf{H} \, \mathbf{x}_k + \mathbf{v}_k",
            r", \quad \mathbf{v}_k \sim \mathcal{N}(0, \mathbf{R})",
            font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION,
        )

        eq_innovation = MathTex(
            r"\tilde{\mathbf{y}} = \mathbf{z} - \mathbf{H} \, \hat{\mathbf{x}}_k^{-}",
            font_size=SMALL_FONT_SIZE, color=COLOR_EQUATION,
        )

        eq_kalman_gain = MathTex(
            r"\mathbf{K} = \mathbf{P}_k^{-} \mathbf{H}^T "
            r"(\mathbf{H} \mathbf{P}_k^{-} \mathbf{H}^T + \mathbf{R})^{-1}",
            font_size=SMALL_FONT_SIZE, color=COLOR_HIGHLIGHT,
        )

        eq_state_update = MathTex(
            r"\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^{-} + \mathbf{K} \, \tilde{\mathbf{y}}",
            font_size=SMALL_FONT_SIZE, color=COLOR_POSTERIOR,
        )

        eq_cov_update = MathTex(
            r"\mathbf{P}_k = (\mathbf{I} - \mathbf{K} \mathbf{H}) \, \mathbf{P}_k^{-}",
            font_size=SMALL_FONT_SIZE, color=COLOR_POSTERIOR,
        )

        eq_stack = VGroup(
            eq_meas_model, eq_innovation, eq_kalman_gain,
            eq_state_update, eq_cov_update,
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        eq_stack.to_edge(RIGHT, buff=0.3).shift(DOWN * 0.5)

        # ── State space (left side) ─────────────────────────────────────
        ss = StateSpace(
            x_range=[-1, 5, 1], y_range=[-2, 3, 1],
            x_length=5.5, y_length=4,
            x_label=r"\text{pos}", y_label=r"\text{vel}",
        )
        ss.shift(DOWN * 1.0 + LEFT * 2.5)
        self.play(Create(ss.axes), FadeIn(ss.x_label_mob), FadeIn(ss.y_label_mob),
                  run_time=NORMAL_ANIM)

        # ── Step 1: Measurement model ───────────────────────────────────
        self.play(Write(eq_meas_model), run_time=NORMAL_ANIM)
        self.wait(PAUSE_SHORT)

        # ── Predicted state (red ellipse) ───────────────────────────────
        mean_pred = np.array([2.0, 1.0])
        P_pred = np.array([[1.2, 0.4], [0.4, 0.6]])

        pred_ellipse = GaussianEllipse(
            mean=mean_pred, cov=P_pred,
            color=COLOR_PREDICTION, axes=ss.axes,
            label=r"\hat{\mathbf{x}}^-",
        )
        self.play(FadeIn(pred_ellipse), run_time=NORMAL_ANIM)
        self.wait(PAUSE_SHORT)

        # ── Measurement arrives (blue dot + ellipse) ────────────────────
        z = np.array([2.8, 0.5])  # only observe position, but we show in 2D
        H = np.array([[1, 0], [0, 1]])  # observe full state for visualization
        R = np.array([[0.5, 0], [0, 0.5]])

        meas_dot = Dot(ss.c2p(z[0], z[1]), color=COLOR_MEASUREMENT,
                        radius=MEASUREMENT_DOT_RADIUS)
        meas_ellipse = GaussianEllipse(
            mean=z, cov=R,
            color=COLOR_MEASUREMENT, axes=ss.axes,
            label=r"\mathbf{z}",
        )

        self.play(FadeIn(meas_dot, scale=1.5), run_time=FAST_ANIM)
        self.play(FadeIn(meas_ellipse), run_time=NORMAL_ANIM)
        self.wait(PAUSE_SHORT)

        # ── Step 2: Innovation ──────────────────────────────────────────
        self.play(Write(eq_innovation), run_time=NORMAL_ANIM)

        # Show innovation arrow
        innov_arrow = Arrow(
            ss.c2p(mean_pred[0], mean_pred[1]),
            ss.c2p(z[0], z[1]),
            color=COLOR_TEXT, stroke_width=2, buff=0.1,
        )
        innov_label = MathTex(r"\tilde{\mathbf{y}}", color=COLOR_TEXT,
                               font_size=SMALL_FONT_SIZE)
        innov_label.next_to(innov_arrow, UP, buff=0.1)
        self.play(Create(innov_arrow), FadeIn(innov_label), run_time=NORMAL_ANIM)
        self.wait(PAUSE_MEDIUM)

        # ── Step 3: Kalman Gain ─────────────────────────────────────────
        self.play(Write(eq_kalman_gain), run_time=SLOW_ANIM)
        self.wait(PAUSE_MEDIUM)

        # ── Step 4: State update ────────────────────────────────────────
        self.play(Write(eq_state_update), run_time=NORMAL_ANIM)

        # Compute updated state
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        innov = z - H @ mean_pred
        mean_upd = mean_pred + K @ innov
        I_KH = np.eye(2) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

        # Morph prediction ellipse into posterior
        posterior_ellipse = GaussianEllipse(
            mean=mean_upd, cov=P_upd,
            color=COLOR_POSTERIOR, axes=ss.axes,
            label=r"\hat{\mathbf{x}}_k",
        )

        self.play(
            FadeOut(innov_arrow), FadeOut(innov_label),
            pred_ellipse.animate.set_opacity(0.15),
            meas_ellipse.animate.set_opacity(0.15),
            FadeIn(posterior_ellipse),
            run_time=SLOW_ANIM,
        )
        self.wait(PAUSE_SHORT)

        # ── Step 5: Covariance update ───────────────────────────────────
        self.play(Write(eq_cov_update), run_time=NORMAL_ANIM)

        shrink_note = Text(
            "Posterior is smaller than both prediction and measurement!",
            color=COLOR_POSTERIOR, font_size=SMALL_FONT_SIZE,
        )
        shrink_note.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(shrink_note), run_time=NORMAL_ANIM)
        self.wait(PAUSE_LONG * 2)

        # ── Fade out ───────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
