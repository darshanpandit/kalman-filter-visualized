"""Swiss / Axonvibe-inspired color palette and style constants for the KF video series."""

from manim import ManimColor

# ── Background ──────────────────────────────────────────────────────────────
BG_COLOR = "#1a1a2e"  # Deep navy-charcoal (set in manim.cfg)

# ── Swiss Palette ───────────────────────────────────────────────────────────
SWISS_RED = "#e63946"       # Prediction / prior Gaussians
ROYAL_BLUE = "#457b9d"      # Measurement Gaussians
LIGHT_BLUE = "#a8dadc"      # Measurement uncertainty fill (lighter)
GOLD = "#f4a261"            # Posterior / estimate
TEAL = "#2a9d8f"            # Process noise, auxiliary highlights
CREAM = "#f1faee"           # Text, labels, white-ish elements
SLATE = "#4a4e69"           # Grid lines, axes, subtle structure
DARK_SLATE = "#2b2d42"      # Secondary background panels

# ── Semantic Aliases (use these in scenes) ──────────────────────────────────
COLOR_PREDICTION = SWISS_RED
COLOR_MEASUREMENT = ROYAL_BLUE
COLOR_MEASUREMENT_LIGHT = LIGHT_BLUE
COLOR_POSTERIOR = GOLD
COLOR_PROCESS_NOISE = TEAL
COLOR_TRUE_PATH = CREAM
COLOR_GRID = SLATE
COLOR_TEXT = CREAM
COLOR_EQUATION = CREAM
COLOR_HIGHLIGHT = GOLD

# ── Comparison Colors (multi-filter scenes) ──────────────────────────
COLOR_FILTER_KF  = SWISS_RED       # red — consistent with prediction color
COLOR_FILTER_EKF = "#e07c42"       # orange — between red and gold
COLOR_FILTER_UKF = TEAL            # teal
COLOR_FILTER_PF  = GOLD            # gold

# ── Part 6+: ML / Transformer Colors ────────────────────────────
COLOR_FILTER_TF        = "#9b59b6"  # violet — transformers
COLOR_FILTER_KALMANNET = "#e74c3c"  # bright red — KalmanNet
COLOR_SSM              = "#1abc9c"  # emerald — state-space models (S4/Mamba)
COLOR_FILTER_IMM       = "#8e44ad"  # purple — IMM (Part 7)
COLOR_FILTER_PHD       = "#27ae60"  # green — PHD filter (Part 7)
COLOR_SOCIAL           = "#f39c12"  # amber — social prediction (Part 7)

# ── Ellipse Defaults ───────────────────────────────────────────────────────
ELLIPSE_FILL_OPACITY = 0.25
ELLIPSE_STROKE_WIDTH = 2.5
ELLIPSE_N_SIGMA = 2  # Number of std deviations for ellipse boundary

# ── Typography ──────────────────────────────────────────────────────────────
TITLE_FONT_SIZE = 48
HEADING_FONT_SIZE = 36
BODY_FONT_SIZE = 28
EQUATION_FONT_SIZE = 36
SMALL_FONT_SIZE = 22

# ── Spacing ─────────────────────────────────────────────────────────────────
STANDARD_BUFF = 0.5
SMALL_BUFF = 0.25
LARGE_BUFF = 1.0

# ── Animation Timing ───────────────────────────────────────────────────────
FAST_ANIM = 0.5
NORMAL_ANIM = 1.0
SLOW_ANIM = 2.0
PAUSE_SHORT = 0.5
PAUSE_MEDIUM = 1.0
PAUSE_LONG = 2.0

# ── Dot / Point Sizes ──────────────────────────────────────────────────────
DOT_RADIUS_SMALL = 0.04
DOT_RADIUS_MEDIUM = 0.06
DOT_RADIUS_LARGE = 0.08
MEASUREMENT_DOT_RADIUS = 0.05

# ── Chart Constants ───────────────────────────────────────────────────────
COLOR_CHART_GRID = SLATE
COLOR_CHART_AXIS = CREAM
CHART_LABEL_FONT_SIZE = 20
CHART_TICK_FONT_SIZE = 16
