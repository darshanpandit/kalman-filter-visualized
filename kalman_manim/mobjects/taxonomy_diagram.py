"""Grand taxonomy diagram for the series finale.

Shows the classification: KF -> EKF -> UKF -> PF -> TF -> World Models.
"""

from __future__ import annotations

from manim import *

from kalman_manim.style import *


class GrandTaxonomyDiagram(VGroup):
    """Series-spanning taxonomy of filtering and dynamics models.

    Tree structure from classical to modern.
    """

    def __init__(self, width: float = 12.0, **kwargs):
        super().__init__(**kwargs)

        entries = [
            ("KF", COLOR_FILTER_KF, "Linear, known"),
            ("EKF", COLOR_FILTER_EKF, "Nonlinear, known"),
            ("UKF", COLOR_FILTER_UKF, "Nonlinear, known"),
            ("PF", COLOR_FILTER_PF, "Arbitrary, known"),
            ("Transformer", COLOR_FILTER_TF, "Learned"),
            ("KalmanNet", COLOR_FILTER_KALMANNET, "Hybrid"),
            ("IMM", COLOR_FILTER_IMM, "Multi-model"),
            ("PHD", COLOR_FILTER_PHD, "Multi-target"),
            ("RSSM", "#3498db", "Learned latent"),
            ("Neural ODE", TEAL, "Continuous learned"),
            ("HNN", "#e67e22", "Energy-conserving"),
        ]

        self.nodes = VGroup()
        for i, (name, color, desc) in enumerate(entries):
            col = i % 4
            row = i // 4

            box = RoundedRectangle(
                corner_radius=0.08, width=2.2, height=0.6,
                color=color, fill_color=color, fill_opacity=0.15,
                stroke_width=1.5,
            )
            label = Text(name, color=color, font_size=16)
            label.move_to(box.get_center() + UP * 0.08)
            sublabel = Text(desc, color=SLATE, font_size=10)
            sublabel.move_to(box.get_center() + DOWN * 0.12)

            node = VGroup(box, label, sublabel)
            node.move_to(RIGHT * (col * 2.8 - 4.2) + DOWN * (row * 1.0))
            self.nodes.add(node)

        # Scale to fit
        if self.nodes.width > width:
            self.nodes.scale(width / self.nodes.width)

        self.add(self.nodes)

        # Column headers
        headers = VGroup(
            Text("Classical", color=SLATE, font_size=14),
            Text("Extended", color=SLATE, font_size=14),
            Text("Multi-Agent", color=SLATE, font_size=14),
            Text("Learned", color=SLATE, font_size=14),
        )
        for i, h in enumerate(headers):
            h.move_to(self.nodes[i].get_top() + UP * 0.4)
        self.add(headers)
