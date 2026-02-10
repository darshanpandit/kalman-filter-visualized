"""Part 2, Scene 3: EKF Equations

Side-by-side comparison of KF vs EKF equations.
Highlights: F → F(x̂), H → H(x̂), and f(x̂) replaces F·x̂.
"""

from __future__ import annotations

from manim import *
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneEKFEquations(VoiceoverScene, Scene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ───────────────────────────────────────────────────────
        title = Text("KF vs EKF Equations", color=COLOR_TEXT,
                      font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3)

        # ── Column headers ──────────────────────────────────────────────
        kf_header = Text("Standard KF", color=COLOR_MEASUREMENT,
                          font_size=HEADING_FONT_SIZE)
        ekf_header = Text("Extended KF", color=COLOR_HIGHLIGHT,
                           font_size=HEADING_FONT_SIZE)
        headers = VGroup(kf_header, ekf_header).arrange(RIGHT, buff=3.0)
        headers.next_to(title, DOWN, buff=STANDARD_BUFF)

        with self.voiceover(text="Let's compare the KF and EKF equations side by side. The structure is nearly identical.") as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)
            self.play(FadeIn(headers), run_time=FAST_ANIM)

        # ── KF equations (left) ─────────────────────────────────────────
        kf_eqs = VGroup(
            MathTex(r"\hat{\mathbf{x}}_k^- = \mathbf{F} \hat{\mathbf{x}}_{k-1}",
                    font_size=SMALL_FONT_SIZE, color=COLOR_MEASUREMENT),
            MathTex(r"\mathbf{P}_k^- = \mathbf{F} \mathbf{P}_{k-1} \mathbf{F}^T + \mathbf{Q}",
                    font_size=SMALL_FONT_SIZE, color=COLOR_MEASUREMENT),
            MathTex(r"\mathbf{K} = \mathbf{P}_k^- \mathbf{H}^T (\mathbf{H} \mathbf{P}_k^- \mathbf{H}^T + \mathbf{R})^{-1}",
                    font_size=SMALL_FONT_SIZE, color=COLOR_MEASUREMENT),
            MathTex(r"\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + \mathbf{K}(\mathbf{z} - \mathbf{H}\hat{\mathbf{x}}_k^-)",
                    font_size=SMALL_FONT_SIZE, color=COLOR_MEASUREMENT),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        kf_eqs.next_to(kf_header, DOWN, buff=STANDARD_BUFF)
        kf_eqs.align_to(kf_header, LEFT)

        # ── EKF equations (right) ───────────────────────────────────────
        ekf_eqs = VGroup(
            MathTex(r"\hat{\mathbf{x}}_k^- = f(\hat{\mathbf{x}}_{k-1})",
                    font_size=SMALL_FONT_SIZE, color=COLOR_HIGHLIGHT),
            MathTex(r"\mathbf{P}_k^- = \mathbf{F}_k \mathbf{P}_{k-1} \mathbf{F}_k^T + \mathbf{Q}",
                    font_size=SMALL_FONT_SIZE, color=COLOR_HIGHLIGHT),
            MathTex(r"\mathbf{K} = \mathbf{P}_k^- \mathbf{H}_k^T (\mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + \mathbf{R})^{-1}",
                    font_size=SMALL_FONT_SIZE, color=COLOR_HIGHLIGHT),
            MathTex(r"\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + \mathbf{K}(\mathbf{z} - h(\hat{\mathbf{x}}_k^-))",
                    font_size=SMALL_FONT_SIZE, color=COLOR_HIGHLIGHT),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        ekf_eqs.next_to(ekf_header, DOWN, buff=STANDARD_BUFF)
        ekf_eqs.align_to(ekf_header, LEFT)

        # Reveal both columns simultaneously
        with self.voiceover(text="For state prediction: the KF multiplies by matrix F, while the EKF calls the nonlinear function f. Same concept, different implementation.") as tracker:
            self.play(Write(kf_eqs[0]), Write(ekf_eqs[0]), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

        with self.voiceover(text="For covariance prediction: the EKF uses F_k, the Jacobian evaluated at the current estimate. It's the local linearization of f.") as tracker:
            self.play(Write(kf_eqs[1]), Write(ekf_eqs[1]), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

        with self.voiceover(text="The Kalman gain formula looks the same, but with Jacobians F_k and H_k replacing the constant matrices.") as tracker:
            self.play(Write(kf_eqs[2]), Write(ekf_eqs[2]), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

        with self.voiceover(text="The update replaces H times x-hat with h of x-hat — the nonlinear measurement function. Everything else stays the same.") as tracker:
            self.play(Write(kf_eqs[3]), Write(ekf_eqs[3]), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)

        self.wait(PAUSE_MEDIUM)

        # ── Highlight differences ───────────────────────────────────────
        diff_notes = VGroup(
            MathTex(r"\mathbf{F} \hat{\mathbf{x}} \rightarrow f(\hat{\mathbf{x}})",
                    font_size=BODY_FONT_SIZE, color=COLOR_POSTERIOR),
            MathTex(r"\mathbf{F} \rightarrow \mathbf{F}_k = \left.\frac{\partial f}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}}",
                    font_size=BODY_FONT_SIZE, color=COLOR_POSTERIOR),
            MathTex(r"\mathbf{H} \hat{\mathbf{x}} \rightarrow h(\hat{\mathbf{x}})",
                    font_size=BODY_FONT_SIZE, color=COLOR_POSTERIOR),
        ).arrange(DOWN, buff=0.2)
        diff_notes.to_edge(DOWN, buff=0.3)

        key_text = Text("Key changes:", color=COLOR_POSTERIOR,
                         font_size=BODY_FONT_SIZE)
        key_text.next_to(diff_notes, UP, buff=0.2)

        with self.voiceover(text="In summary: F·x becomes f(x), H·x becomes h(x), and constant matrices become step-varying Jacobians. The logic is identical — only the linearization changes.") as tracker:
            self.play(FadeIn(key_text), FadeIn(diff_notes), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG * 2)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
