"""Stylized architecture diagrams for transformer and KalmanNet.

Block diagrams showing the flow of information, used in Part 6 scenes.
"""

from __future__ import annotations

from manim import *

from kalman_manim.style import (
    COLOR_TEXT, COLOR_PREDICTION, COLOR_MEASUREMENT, COLOR_POSTERIOR,
    COLOR_FILTER_TF, COLOR_FILTER_KALMANNET, SLATE, DARK_SLATE,
    BODY_FONT_SIZE, SMALL_FONT_SIZE,
)


def _make_block(label: str, color: str, width: float = 1.8,
                height: float = 0.7, font_size: int = 18) -> VGroup:
    """Create a labeled rounded rectangle block."""
    rect = RoundedRectangle(
        corner_radius=0.1, width=width, height=height,
        color=color, fill_color=color, fill_opacity=0.2,
        stroke_width=2,
    )
    text = Text(label, color=color, font_size=font_size)
    text.move_to(rect)
    return VGroup(rect, text)


def _make_arrow(start: VGroup, end: VGroup, color: str = SLATE) -> Arrow:
    """Create an arrow from one block to another."""
    return Arrow(
        start.get_right(), end.get_left(),
        color=color, stroke_width=2, buff=0.1,
        max_tip_length_to_length_ratio=0.15,
    )


class TransformerDiagram(VGroup):
    """Stylized transformer encoder diagram for filtering.

    Shows: Input Embeddings → Multi-Head Attention → FFN → Output
    with causal masking annotation.
    """

    def __init__(self, width: float = 10.0, **kwargs):
        super().__init__(**kwargs)

        # Blocks
        embed = _make_block("Embed z_t", COLOR_MEASUREMENT)
        pos_enc = _make_block("+ Pos Enc", SLATE)
        attention = _make_block("Causal\nAttention", COLOR_FILTER_TF, width=2.0)
        ffn = _make_block("FFN", COLOR_FILTER_TF)
        output = _make_block("x\u0302_t", COLOR_POSTERIOR)

        blocks = VGroup(embed, pos_enc, attention, ffn, output)
        blocks.arrange(RIGHT, buff=0.5)

        # Scale to fit width
        if blocks.width > width:
            blocks.scale(width / blocks.width)

        # Arrows
        arrows = VGroup(
            _make_arrow(embed, pos_enc),
            _make_arrow(pos_enc, attention),
            _make_arrow(attention, ffn),
            _make_arrow(ffn, output),
        )

        # Causal mask annotation
        mask_note = Text(
            "causal mask: attend only to past",
            color=SLATE, font_size=14,
        )
        mask_note.next_to(attention, DOWN, buff=0.3)

        self.blocks = blocks
        self.arrows = arrows
        self.mask_note = mask_note
        self.add(blocks, arrows, mask_note)


class KalmanNetDiagram(VGroup):
    """KalmanNet architecture: GRU replaces Kalman gain computation.

    Shows: Predict → Innovation → GRU → Kalman Gain → Update
    """

    def __init__(self, width: float = 10.0, **kwargs):
        super().__init__(**kwargs)

        predict = _make_block("Predict\nF\u00b7x", COLOR_PREDICTION)
        innov = _make_block("Innovation\nz - H\u00b7x\u0302", COLOR_MEASUREMENT)
        gru = _make_block("GRU\n(learned)", COLOR_FILTER_KALMANNET, width=2.0)
        gain = _make_block("K_t", COLOR_FILTER_KALMANNET)
        update = _make_block("Update\nx + K\u00b7y", COLOR_POSTERIOR)

        blocks = VGroup(predict, innov, gru, gain, update)
        blocks.arrange(RIGHT, buff=0.5)

        if blocks.width > width:
            blocks.scale(width / blocks.width)

        arrows = VGroup(
            _make_arrow(predict, innov),
            _make_arrow(innov, gru),
            _make_arrow(gru, gain),
            _make_arrow(gain, update),
        )

        note = Text(
            "GRU replaces analytic Kalman gain",
            color=SLATE, font_size=14,
        )
        note.next_to(gru, DOWN, buff=0.3)

        self.blocks = blocks
        self.arrows = arrows
        self.note = note
        self.add(blocks, arrows, note)


class SSMDiagram(VGroup):
    """State-space model (S4/Mamba) diagram showing KF correspondence.

    Shows: h_t = A h_{t-1} + B x_t ; y_t = C h_t + D x_t
    with annotations mapping to KF.
    """

    def __init__(self, width: float = 10.0, **kwargs):
        super().__init__(**kwargs)

        state_update = _make_block("h' = Ah + Bx", "#1abc9c", width=2.5)
        output = _make_block("y = Ch + Dx", "#1abc9c", width=2.5)
        selective = _make_block("Selective\nGating", "#e67e22", width=2.0)

        blocks = VGroup(state_update, output, selective)
        blocks.arrange(RIGHT, buff=0.8)

        if blocks.width > width:
            blocks.scale(width / blocks.width)

        arrows = VGroup(
            _make_arrow(state_update, output),
            _make_arrow(selective, state_update),
        )

        # Recurrence arrow (feedback loop)
        feedback = CurvedArrow(
            state_update.get_top() + RIGHT * 0.3,
            state_update.get_top() + LEFT * 0.3,
            color=SLATE, stroke_width=1.5, angle=-PI,
        )

        # KF correspondence labels
        kf_label = Text(
            "A,B,C,D = F,B,H,0 in KF",
            color=SLATE, font_size=14,
        )
        kf_label.next_to(blocks, DOWN, buff=0.3)

        mamba_label = Text(
            "Mamba: A,B,C depend on input (selective)",
            color="#e67e22", font_size=14,
        )
        mamba_label.next_to(kf_label, DOWN, buff=0.15)

        self.blocks = blocks
        self.arrows = arrows
        self.add(blocks, arrows, feedback, kf_label, mamba_label)
