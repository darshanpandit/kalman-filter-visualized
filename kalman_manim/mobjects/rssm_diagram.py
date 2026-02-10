"""RSSM architecture diagram for Part 8 world model scenes.

Shows the GRU (deterministic) + stochastic nodes with arrows,
annotated with KF equivalences.
"""

from __future__ import annotations

from manim import *

from kalman_manim.style import (
    COLOR_TEXT, COLOR_PREDICTION, COLOR_MEASUREMENT, COLOR_POSTERIOR,
    SLATE, DARK_SLATE, BODY_FONT_SIZE, SMALL_FONT_SIZE,
)


def _block(label: str, color: str, w: float = 2.0, h: float = 0.7,
           font_size: int = 18) -> VGroup:
    rect = RoundedRectangle(
        corner_radius=0.1, width=w, height=h,
        color=color, fill_color=color, fill_opacity=0.2, stroke_width=2,
    )
    text = Text(label, color=color, font_size=font_size)
    text.move_to(rect)
    return VGroup(rect, text)


class RSSMDiagram(VGroup):
    """RSSM architecture: deterministic h_t + stochastic s_t.

    Shows predict (prior) and update (posterior) paths.
    """

    def __init__(self, width: float = 10.0, **kwargs):
        super().__init__(**kwargs)

        det = _block("GRU (h_t)", "#3498db", w=2.2)
        prior = _block("Prior p(s_t|h_t)", COLOR_PREDICTION, w=2.8)
        post = _block("Posterior q(s_t|h_t,o_t)", COLOR_POSTERIOR, w=3.2)
        obs = _block("Observation o_t", COLOR_MEASUREMENT, w=2.5)

        # Layout: det on left, prior/post stacked on right, obs below post
        det.move_to(LEFT * 3)
        prior.move_to(RIGHT * 1.5 + UP * 1.0)
        post.move_to(RIGHT * 1.5 + DOWN * 0.5)
        obs.move_to(RIGHT * 1.5 + DOWN * 2.0)

        blocks = VGroup(det, prior, post, obs)
        if blocks.width > width:
            blocks.scale(width / blocks.width)

        arrows = VGroup(
            Arrow(det.get_right(), prior.get_left(), color=SLATE,
                  stroke_width=2, buff=0.1),
            Arrow(det.get_right(), post.get_left(), color=SLATE,
                  stroke_width=2, buff=0.1),
            Arrow(obs.get_top(), post.get_bottom(), color=SLATE,
                  stroke_width=2, buff=0.1),
        )

        # KL annotation
        kl_label = Text("KL(posterior || prior)", color=SLATE, font_size=14)
        kl_label.move_to((prior.get_center() + post.get_center()) / 2 + RIGHT * 2.5)
        kl_arrow = Arrow(
            kl_label.get_left(),
            (prior.get_right() + post.get_right()) / 2,
            color=SLATE, stroke_width=1, buff=0.1,
        )

        self.blocks = blocks
        self.arrows = arrows
        self.add(blocks, arrows, kl_label, kl_arrow)


class GraphicalModel(VGroup):
    """Simple plate notation graphical model for DKF / DVBF."""

    def __init__(self, n_steps: int = 4, **kwargs):
        super().__init__(**kwargs)

        z_nodes = VGroup()
        x_nodes = VGroup()
        arrows = VGroup()

        for t in range(n_steps):
            x_pos = t * 1.5
            # Latent z_t
            z = Circle(radius=0.3, color=COLOR_PREDICTION,
                       fill_opacity=0.2, stroke_width=2)
            z.move_to(UP * 1.0 + RIGHT * x_pos)
            z_label = Text(f"z_{t}", color=COLOR_PREDICTION, font_size=14)
            z_label.move_to(z)
            z_nodes.add(VGroup(z, z_label))

            # Observed x_t
            x = Circle(radius=0.3, color=COLOR_MEASUREMENT,
                       fill_opacity=0.3, stroke_width=2)
            x.move_to(DOWN * 0.5 + RIGHT * x_pos)
            x_label = Text(f"x_{t}", color=COLOR_MEASUREMENT, font_size=14)
            x_label.move_to(x)
            x_nodes.add(VGroup(x, x_label))

            # z_t -> x_t
            arrows.add(Arrow(z.get_bottom(), x.get_top(), color=SLATE,
                              stroke_width=1.5, buff=0.1))

            # z_{t-1} -> z_t
            if t > 0:
                arrows.add(Arrow(
                    z_nodes[t - 1][0].get_right(),
                    z.get_left(),
                    color=SLATE, stroke_width=1.5, buff=0.1,
                ))

        all_nodes = VGroup(z_nodes, x_nodes)
        all_nodes.center()
        arrows.center()
        self.add(all_nodes, arrows)

        # Plate notation
        plate = RoundedRectangle(
            corner_radius=0.15,
            width=all_nodes.width + 0.6,
            height=all_nodes.height + 0.6,
            color=SLATE, stroke_width=1,
        )
        plate.move_to(all_nodes)
        plate_label = Text("T steps", color=SLATE, font_size=12)
        plate_label.next_to(plate, DR, buff=-0.3)
        self.add(plate, plate_label)
