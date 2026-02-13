"""Scene 5: Detection Pipeline — from pixels to bounding boxes.

Data: None (conceptual diagram scene)

Introduces the object detection pipeline (backbone, feature maps, detections),
IoU metric, two-stage vs one-stage detectors, and YOLO evolution.

Papers:
- Ren et al. (2015) — Faster R-CNN
- Redmon et al. (2016) — YOLO v1
- Jocher et al. (2023) — YOLOv8
"""

from __future__ import annotations

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *


class SceneDetectionPipeline(VoiceoverScene, MovingCameraScene):
    """Object detection fundamentals for pedestrian crosswalk monitoring."""

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR

        # ── Title ─────────────────────────────────────────────────────
        title = Text("Detection Pipeline", color=COLOR_TEXT, font_size=TITLE_FONT_SIZE)
        title.to_edge(UP, buff=0.3).set_z_index(10)

        with self.voiceover(
            text="Before we can track pedestrians, we need to detect them "
                 "in every frame. Let's walk through the modern object "
                 "detection pipeline."
        ) as tracker:
            self.play(Write(title), run_time=NORMAL_ANIM)

        # ── Pipeline diagram ──────────────────────────────────────────
        stage_specs = [
            ("Image", COLOR_MEASUREMENT), ("CNN Backbone", TEAL),
            ("Feature Maps", COLOR_FILTER_TF), ("Detections", COLOR_HIGHLIGHT),
        ]
        boxes, labels = VGroup(), VGroup()
        for text_str, color in stage_specs:
            box = RoundedRectangle(
                corner_radius=0.1, width=2.2, height=0.8,
                stroke_color=color, stroke_width=2.5,
                fill_color=DARK_SLATE, fill_opacity=0.6,
            )
            lbl = Text(text_str, color=color, font_size=SMALL_FONT_SIZE)
            lbl.move_to(box)
            boxes.add(box); labels.add(lbl)

        pipeline_boxes = VGroup(*[VGroup(b, l) for b, l in zip(boxes, labels)])
        pipeline_boxes.arrange(RIGHT, buff=0.6)
        pipeline_boxes.next_to(title, DOWN, buff=0.6)
        pipeline_boxes.scale_to_fit_width(min(pipeline_boxes.width, 11.5))

        arrows = VGroup()
        for i in range(len(boxes) - 1):
            arrows.add(Arrow(
                boxes[i].get_right(), boxes[i + 1].get_left(),
                stroke_color=SLATE, stroke_width=2, buff=0.1,
                max_tip_length_to_length_ratio=0.2,
            ))

        with self.voiceover(
            text="An image enters a convolutional neural network backbone, "
                 "which extracts feature maps at multiple scales. These "
                 "features are decoded into bounding box detections — "
                 "rectangles around each pedestrian."
        ) as tracker:
            for i, stage in enumerate(pipeline_boxes):
                self.play(FadeIn(stage, shift=RIGHT * 0.2), run_time=0.4)
                if i < len(arrows):
                    self.play(GrowArrow(arrows[i]), run_time=0.3)
            self.wait(PAUSE_MEDIUM)

        # ── Transition to IoU ─────────────────────────────────────────
        pipeline_group = VGroup(pipeline_boxes, arrows)
        with self.voiceover(
            text="But how do we know if a detection is correct? We "
                 "compare it to the ground truth using Intersection "
                 "over Union."
        ) as tracker:
            self.play(
                pipeline_group.animate.scale(0.6).to_edge(UP, buff=0.4).shift(DOWN * 0.1),
                title.animate.scale(0.7).to_corner(UL, buff=0.2),
                run_time=NORMAL_ANIM,
            )

        # ── IoU diagram ──────────────────────────────────────────────
        pred_sq = Square(side_length=1.6, stroke_color=COLOR_MEASUREMENT,
                         stroke_width=2.5, fill_color=COLOR_MEASUREMENT, fill_opacity=0.15)
        gt_sq = Square(side_length=1.6, stroke_color=TEAL, stroke_width=2.5,
                       fill_color=TEAL, fill_opacity=0.15)
        pred_sq.shift(LEFT * 0.35 + DOWN * 0.25)
        gt_sq.shift(RIGHT * 0.35 + UP * 0.25)
        overlap = Intersection(pred_sq, gt_sq, stroke_width=0,
                               fill_color=COLOR_HIGHLIGHT, fill_opacity=0.45)

        pred_label = Text("Predicted", color=COLOR_MEASUREMENT, font_size=SMALL_FONT_SIZE)
        pred_label.next_to(pred_sq, DOWN, buff=0.15)
        gt_label = Text("Ground Truth", color=TEAL, font_size=SMALL_FONT_SIZE)
        gt_label.next_to(gt_sq, UP, buff=0.15)

        iou_formula = MathTex(r"\text{IoU} = \frac{\text{Intersection}}{\text{Union}}",
                              color=COLOR_TEXT, font_size=BODY_FONT_SIZE)
        iou_threshold = Text("IoU > 0.5 = True Positive",
                             color=COLOR_HIGHLIGHT, font_size=BODY_FONT_SIZE)
        iou_text = VGroup(iou_formula, iou_threshold).arrange(DOWN, buff=0.2)

        iou_group = VGroup(pred_sq, gt_sq, overlap, pred_label, gt_label)
        iou_group.move_to(LEFT * 2.5 + DOWN * 0.8)
        iou_text.next_to(iou_group, RIGHT, buff=0.8)

        with self.voiceover(
            text="The predicted box overlaps the ground truth box. The "
                 "ratio of their intersection area to their union area "
                 "is the IoU score. If it exceeds zero point five, we "
                 "call it a true positive."
        ) as tracker:
            self.play(FadeIn(pred_sq), FadeIn(gt_sq),
                      FadeIn(pred_label), FadeIn(gt_label), run_time=NORMAL_ANIM)
            self.play(FadeIn(overlap), run_time=FAST_ANIM)
            self.play(FadeIn(iou_text, shift=LEFT * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── Two-stage vs one-stage ────────────────────────────────────
        iou_all = VGroup(iou_group, iou_text)

        def make_panel(heading, hcolor, line1, line2):
            h = Text(heading, color=hcolor, font_size=HEADING_FONT_SIZE)
            t1 = Text(line1, color=COLOR_TEXT, font_size=SMALL_FONT_SIZE)
            t2 = Text(line2, color=SLATE, font_size=SMALL_FONT_SIZE)
            content = VGroup(h, t1, t2).arrange(DOWN, buff=0.15)
            bg = RoundedRectangle(
                corner_radius=0.12, width=content.width + 0.6,
                height=content.height + 0.5, stroke_color=hcolor,
                stroke_width=2, fill_color=DARK_SLATE, fill_opacity=0.5)
            content.move_to(bg)
            return VGroup(bg, content)

        with self.voiceover(
            text="Now, there are two families of detectors. Two-stage "
                 "detectors like Faster R-CNN first propose regions, then "
                 "classify each one — accurate but slow. One-stage detectors "
                 "like YOLO predict boxes directly in a single pass — "
                 "fast enough for real-time edge deployment."
        ) as tracker:
            self.play(FadeOut(iou_all), run_time=FAST_ANIM)
            two_stage = make_panel("Two-Stage", COLOR_MEASUREMENT,
                                   "Faster R-CNN", "Accurate, but slow")
            one_stage = make_panel("One-Stage", COLOR_HIGHLIGHT,
                                   "YOLO", "Fast, real-time ready")
            comparison = VGroup(two_stage, one_stage).arrange(RIGHT, buff=0.8)
            comparison.move_to(DOWN * 0.6)
            comparison.scale_to_fit_width(min(comparison.width, 11.0))
            self.play(FadeIn(two_stage, shift=RIGHT * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_SHORT)
            self.play(FadeIn(one_stage, shift=LEFT * 0.3), run_time=NORMAL_ANIM)
            self.wait(PAUSE_MEDIUM)

        # ── YOLO evolution timeline ───────────────────────────────────
        with self.voiceover(
            text="YOLO has evolved rapidly. Version one in twenty sixteen "
                 "introduced the single-pass philosophy. By YOLOv8 in "
                 "twenty twenty-three, the architecture is anchor-free "
                 "with direct bounding box regression — dramatically "
                 "simpler and faster. This is what runs on our Jetson."
        ) as tracker:
            self.play(FadeOut(comparison), run_time=FAST_ANIM)
            yolo_title = Text("YOLO Evolution", color=COLOR_TEXT,
                              font_size=HEADING_FONT_SIZE)
            arrow_line = Arrow(LEFT * 4, RIGHT * 4, stroke_color=SLATE,
                               stroke_width=2, max_tip_length_to_length_ratio=0.05)
            yolo_title.next_to(arrow_line, UP, buff=0.5)

            milestones = [("YOLOv1", "2016", LEFT * 3.5), ("YOLOv3", "2018", LEFT * 1.0),
                          ("YOLOv5", "2020", RIGHT * 1.5), ("YOLOv8", "2023", RIGHT * 3.5)]
            dots_and_labels = VGroup()
            for name, year, pos in milestones:
                dot = Dot(arrow_line.get_center() + pos, radius=0.06, color=COLOR_HIGHLIGHT)
                nt = Text(name, color=COLOR_HIGHLIGHT, font_size=SMALL_FONT_SIZE)
                yt = Text(year, color=SLATE, font_size=SMALL_FONT_SIZE)
                nt.next_to(dot, UP, buff=0.15); yt.next_to(dot, DOWN, buff=0.15)
                dots_and_labels.add(VGroup(dot, nt, yt))

            timeline = VGroup(yolo_title, arrow_line, dots_and_labels)
            timeline.move_to(DOWN * 0.5)
            subtitle = Text("Anchor-free, direct regression",
                            color=TEAL, font_size=BODY_FONT_SIZE)
            subtitle.next_to(timeline, DOWN, buff=0.5)

            self.play(FadeIn(yolo_title), GrowArrow(arrow_line), run_time=NORMAL_ANIM)
            for ms in dots_and_labels:
                self.play(FadeIn(ms, scale=1.3), run_time=0.4)
            self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=NORMAL_ANIM)
            self.wait(PAUSE_LONG)

        # ── Fade out ──────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
