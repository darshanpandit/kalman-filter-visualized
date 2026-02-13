"""Scene 6: Tracking IS Kalman Filtering — the bridge to the KF series.

Data: TRACKER_CHARS from data.py

The SORT algorithm is literally a Kalman predict/update loop with
Hungarian assignment. This scene makes that connection explicit,
shows the state vector, animates a predict-update cycle on a bounding
box, and surveys tracker innovations (ByteTrack, OC-SORT, StrongSORT).

Papers:
- Bewley et al. (2016) — SORT: Simple Online and Realtime Tracking
- Zhang et al. (2022) — ByteTrack
- Cao et al. (2023) — OC-SORT
- Du et al. (2023) — StrongSORT
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from shaum703_smart_crosswalks.data import TRACKER_CHARS


class SceneTrackingIsKalman(VoiceoverScene, MovingCameraScene):
    """SORT = Kalman predict/update + Hungarian matching.

    Visual: SORT pipeline loop, state vector, predict-update bounding box
    animation, tracker innovation cards, citations.
    """

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ─────────────────────────────────────────────────────
        title = Text("Tracking IS Kalman Filtering",
                     color=COLOR_TEXT, font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text="Here is the key insight that connects this study to "
                 "everything we've built in the Kalman filter series. "
                 "Multi-object tracking — the algorithm called SORT — "
                 "is literally a Kalman predict-update cycle."
        ) as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── SORT pipeline diagram ─────────────────────────────────────
        stage_specs = [
            ("Detection", COLOR_MEASUREMENT), ("Kalman Predict", COLOR_PREDICTION),
            ("Hungarian Match", TEAL), ("Kalman Update", COLOR_POSTERIOR),
        ]
        boxes, labels = VGroup(), VGroup()
        for text_str, color in stage_specs:
            box = RoundedRectangle(
                corner_radius=0.12, width=2.5, height=0.9,
                stroke_color=color, stroke_width=2.5,
                fill_color=DARK_SLATE, fill_opacity=0.6,
            )
            lbl = Text(text_str, color=color, font_size=SMALL_FONT_SIZE)
            lbl.move_to(box)
            boxes.add(box); labels.add(lbl)

        sort_stages = VGroup(*[VGroup(b, l) for b, l in zip(boxes, labels)])
        sort_stages.arrange(RIGHT, buff=0.5)
        sort_stages.next_to(title, DOWN, buff=0.55)
        sort_stages.scale_to_fit_width(min(sort_stages.width, 11.5))

        fwd_arrows = VGroup()
        for i in range(len(boxes) - 1):
            fwd_arrows.add(Arrow(
                boxes[i].get_right(), boxes[i + 1].get_left(),
                stroke_color=SLATE, stroke_width=2, buff=0.08,
                max_tip_length_to_length_ratio=0.2,
            ))

        loop_arrow = CurvedArrow(
            boxes[-1].get_bottom() + DOWN * 0.15,
            boxes[0].get_bottom() + DOWN * 0.15,
            angle=-TAU / 6, stroke_color=SLATE, stroke_width=2,
        )
        loop_label = Text("next frame", color=SLATE, font_size=SMALL_FONT_SIZE)
        loop_label.next_to(loop_arrow, DOWN, buff=0.08)

        with self.voiceover(
            text="SORT stands for Simple Online Realtime Tracking. Each "
                 "frame, a detector produces bounding boxes. The Kalman "
                 "filter predicts where each existing track should be. "
                 "The Hungarian algorithm matches predictions to detections. "
                 "Then the Kalman filter updates its state. And the loop "
                 "repeats — every single frame."
        ) as tracker:
            for i, stage in enumerate(sort_stages):
                self.play(FadeIn(stage, shift=RIGHT * 0.2), run_time=0.4)
                if i < len(fwd_arrows):
                    self.play(GrowArrow(fwd_arrows[i]), run_time=0.25)
            self.play(Create(loop_arrow), FadeIn(loop_label), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Shrink pipeline, show state vector ────────────────────────
        sort_group = VGroup(sort_stages, fwd_arrows, loop_arrow, loop_label)

        with self.voiceover(
            text="What exactly does the Kalman filter track? The state "
                 "vector contains the bounding box center, its width and "
                 "height, and all four velocities — eight dimensions."
        ) as tracker:
            self.play(
                sort_group.animate.scale(0.55).to_edge(UP, buff=0.35),
                title.animate.scale(0.65).to_corner(UL, buff=0.15),
                run_time=NORMAL_ANIM,
            )
            state_vec = MathTex(
                r"\mathbf{x} = "
                r"\begin{bmatrix} x \\ y \\ w \\ h \\"
                r" \dot{x} \\ \dot{y} \\ \dot{w} \\ \dot{h} \end{bmatrix}",
                color=COLOR_TEXT, font_size=BODY_FONT_SIZE,
            )
            state_label = Text("Bounding box + velocities",
                               color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE)
            state_group = VGroup(state_vec, state_label).arrange(DOWN, buff=0.25)
            state_group.move_to(LEFT * 3.2 + DOWN * 0.6)
            self.play(FadeIn(state_group, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Predict-update bounding box animation ─────────────────────
        bbox_w, bbox_h = 0.9, 1.5
        predicted_box = RoundedRectangle(
            corner_radius=0.06, width=bbox_w, height=bbox_h,
            stroke_color=COLOR_PREDICTION, stroke_width=2.5, fill_opacity=0.0,
        )
        predicted_box.move_to(RIGHT * 1.5 + DOWN * 0.6)
        pred_tag = Text("Predict", color=COLOR_PREDICTION, font_size=SMALL_FONT_SIZE)
        pred_tag.next_to(predicted_box, UP, buff=0.1)

        det_offset = RIGHT * 0.6 + UP * 0.35
        detection_box = RoundedRectangle(
            corner_radius=0.06, width=bbox_w * 0.95, height=bbox_h * 1.05,
            stroke_color=COLOR_MEASUREMENT, stroke_width=2.5,
            stroke_opacity=0.0, fill_opacity=0.0,
        )
        det_tag = Text("Detection", color=COLOR_MEASUREMENT, font_size=SMALL_FONT_SIZE)

        with self.voiceover(
            text="Watch the cycle in action. The Kalman filter predicts "
                 "where the bounding box should be — shown in red. Then "
                 "the detector produces a new measurement — in blue. The "
                 "update step fuses the two, pulling the prediction toward "
                 "the detection. This is exactly the same predict-update "
                 "cycle we derived in Part one."
        ) as tracker:
            self.play(FadeIn(predicted_box), FadeIn(pred_tag), run_time=FAST_ANIM)
            self.play(predicted_box.animate.shift(RIGHT * 1.0),
                      pred_tag.animate.shift(RIGHT * 1.0), run_time=NORMAL_ANIM)
            # Detection appears offset from predicted position
            detection_box.move_to(predicted_box.get_center() + det_offset)
            det_tag.next_to(detection_box, UP, buff=0.1)
            self.play(detection_box.animate.set_stroke(opacity=1.0),
                      FadeIn(det_tag), run_time=FAST_ANIM)
            self.wait(PAUSE_SHORT)
            # Update: predicted box morphs toward detection
            updated_center = predicted_box.get_center() + det_offset * 0.6
            update_tag = Text("Update", color=COLOR_POSTERIOR, font_size=SMALL_FONT_SIZE)
            update_tag.next_to(updated_center + UP * bbox_h / 2, UP, buff=0.1)
            self.play(
                predicted_box.animate.move_to(updated_center).set_stroke(color=COLOR_POSTERIOR),
                FadeOut(pred_tag), FadeIn(update_tag), run_time=NORMAL_ANIM,
            )
            self.wait(PAUSE_MEDIUM)

        # ── Clear animation, introduce tracker cards ──────────────────
        anim_mobs = VGroup(predicted_box, detection_box, det_tag, update_tag, state_group)
        with self.voiceover(
            text="All modern trackers build on this foundation. They "
                 "differ in how they handle edge cases — low-confidence "
                 "detections, lost tracks, and appearance changes."
        ) as tracker:
            self.play(FadeOut(anim_mobs), run_time=FAST_ANIM)

        # ── Tracker innovation cards ──────────────────────────────────
        card_specs = [
            ("ByteTrack", COLOR_FILTER_TF, "Use every detection", "Low-conf second matching"),
            ("OC-SORT", TEAL, "Re-update on recovery", "Observation-centric KF"),
            ("StrongSORT", COLOR_FILTER_EKF, "Add appearance (ReID)", "Hybrid motion + looks"),
        ]
        cards = VGroup()
        for name, color, tagline, detail in card_specs:
            name_t = Text(name, color=color, font_size=HEADING_FONT_SIZE)
            tag_t = Text(tagline, color=COLOR_TEXT, font_size=BODY_FONT_SIZE)
            det_t = Text(detail, color=SLATE, font_size=SMALL_FONT_SIZE)
            content = VGroup(name_t, tag_t, det_t).arrange(DOWN, buff=0.12)
            bg = RoundedRectangle(
                corner_radius=0.12, width=content.width + 0.5,
                height=content.height + 0.4, stroke_color=color, stroke_width=2,
                fill_color=DARK_SLATE, fill_opacity=0.55,
            )
            content.move_to(bg)
            cards.add(VGroup(bg, content))
        cards.arrange(RIGHT, buff=0.4)
        cards.move_to(DOWN * 0.6)
        cards.scale_to_fit_width(min(cards.width, 11.5))

        with self.voiceover(
            text="ByteTrack uses every detection — even low-confidence "
                 "ones — through a second matching pass. This recovers "
                 "partially occluded pedestrians that other trackers drop."
        ) as tracker:
            self.play(FadeIn(cards[0], shift=UP * 0.2), run_time=NORMAL_ANIM)

        with self.voiceover(
            text="OC-SORT re-runs the Kalman update retroactively when "
                 "a lost track reappears. Instead of predicting forward "
                 "from stale data, it corrects the entire trajectory."
        ) as tracker:
            self.play(FadeIn(cards[1], shift=UP * 0.2), run_time=NORMAL_ANIM)

        with self.voiceover(
            text="StrongSORT adds a re-identification network — it learns "
                 "what each pedestrian looks like, so it can re-link "
                 "tracks even after long occlusions."
        ) as tracker:
            self.play(FadeIn(cards[2], shift=UP * 0.2), run_time=NORMAL_ANIM)

        self.wait(PAUSE_MEDIUM)

        # ── Common thread callout ─────────────────────────────────────
        with self.voiceover(
            text="But notice — every single one of these trackers uses "
                 "a Kalman filter at its core. The motion model is always "
                 "a linear constant-velocity Kalman filter, the same one "
                 "we built from scratch in Part one."
        ) as tracker:
            common = Text("All use Kalman at core",
                          color=COLOR_HIGHLIGHT, font_size=HEADING_FONT_SIZE)
            common.to_edge(DOWN, buff=0.4)
            self.play(FadeIn(common, scale=0.9), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Citations ─────────────────────────────────────────────────
        with self.voiceover(
            text="Bewley and colleagues introduced SORT in twenty sixteen. "
                 "Zhang and colleagues extended it with ByteTrack in "
                 "twenty twenty-two. Every innovation since has kept "
                 "the Kalman predict-update loop intact."
        ) as tracker:
            self.play(FadeOut(cards), FadeOut(common), run_time=FAST_ANIM)
            cite1 = Text("Bewley et al. (2016) — SORT",
                         color=COLOR_TEXT, font_size=BODY_FONT_SIZE)
            cite2 = Text("Zhang et al. (2022) — ByteTrack",
                         color=COLOR_TEXT, font_size=BODY_FONT_SIZE)
            cite_group = VGroup(cite1, cite2).arrange(DOWN, buff=0.25)
            cite_group.move_to(DOWN * 0.5)
            self.play(FadeIn(cite_group, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Fade out ──────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
