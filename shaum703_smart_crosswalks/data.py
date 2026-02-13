"""Precomputed data from the SHAUM703 report for scene consumption.

All numbers are digitized from published tables in:
  Cirillo, Pandit & Momeni Rad (2025). Evaluation of Smart Pedestrian
  Crosswalk Technologies. MDOT SHA Research Report.
"""

from __future__ import annotations

import numpy as np


# ── Crash Modification Factors (Table 1.2) ────────────────────────────
# CMF < 1 means crash reduction; lower = better
CMF_DATA = {
    "treatments": ["Marked Only", "RRFB", "PHB", "MPS"],
    "cmf_ped": [1.0, 0.53, 0.45, 0.55],
    "reduction_pct": [0, 47, 55, 45],
}

# ── Driver Yielding Rates ─────────────────────────────────────────────
YIELDING_DATA = {
    "treatments": ["Baseline\n(No Control)", "RRFB", "PHB"],
    "yielding_low": [10, 80, 95],  # lower bound %
    "yielding_high": [20, 90, 100],  # upper bound %
    "yielding_mid": [15, 85, 97.5],  # midpoint for bar charts
}

# ── Sensor Specifications (Table 2.1) ─────────────────────────────────
SENSOR_SPECS = {
    "metrics": [
        "Resolution",
        "Min. Illumination",
        "NETD",
        "Detection Range\n(Pedestrian)",
        "Frame Rate",
        "Temp. Range",
    ],
    "bosch": [
        "1080p / 4K UHD",
        "0.0047 lux (color)",
        "N/A (visible)",
        "120 m (headlights)\n60 m (no light)",
        "Up to 30 fps",
        "-40 to +65 C",
    ],
    "flir": [
        "640x480 (VGA)",
        "0 lux (thermal)",
        "< 50 mK",
        "80 m (pedestrian)",
        "Up to 30 fps",
        "-40 to +74 C",
    ],
}

# ── Tracking Metrics (Table 5.1) ──────────────────────────────────────
# Rows: [Site, Camera, Tracker, HOTA, DetRe, DetPr, FP, FN, IDSw]
TRACKING_METRICS = {
    "columns": ["Site", "Camera", "Tracker", "HOTA", "DetRe", "DetPr", "FP", "FN", "IDSw"],
    "rows": [
        # Site 1
        ["1", "Bosch", "ByteTrack",   0.953, 0.998, 0.959, 14900,    830,  647],
        ["1", "Bosch", "OC-SORT",     0.808, 0.819, 0.994,  1509,  60079, 1978],
        ["1", "Bosch", "StrongSORT",  0.881, 0.902, 0.986,  4294,  33511, 3078],
        ["1", "FLIR",  "ByteTrack",   0.932, 0.998, 0.939,  5436,    168,  162],
        ["1", "FLIR",  "OC-SORT",     0.823, 0.837, 0.991,   443,  10076,  382],
        ["1", "FLIR",  "StrongSORT",  0.873, 0.901, 0.981,  1289,   7188,  753],
        # Site 2
        ["2", "Bosch", "ByteTrack",   0.992, 0.999, 0.994,   159,     27,   22],
        ["2", "Bosch", "OC-SORT",     0.965, 0.968, 0.999,    14,    736,   54],
        ["2", "Bosch", "StrongSORT",  0.977, 0.982, 0.998,    53,    447,   67],
        ["2", "FLIR",  "ByteTrack",   0.992, 1.000, 0.994,   114,      9,   14],
        ["2", "FLIR",  "OC-SORT",     0.954, 0.959, 0.999,    20,    604,   40],
        ["2", "FLIR",  "StrongSORT",  0.970, 0.975, 0.997,    42,    400,   42],
    ],
}

# ── Tracker Characteristics (Table 4.1) ───────────────────────────────
TRACKER_CHARS = {
    "trackers": ["ByteTrack", "OC-SORT", "StrongSORT", "BoT-SORT"],
    "fps": [100, 50, 25, 30],  # approximate
    "motion_model": ["Kalman", "Kalman+OCA", "Kalman+EMA", "Kalman+CMC"],
    "appearance": ["None", "None", "ReID (BoT)", "ReID (BoT)"],
    "key_innovation": [
        "Low-conf 2nd match",
        "Obs-centric re-update",
        "Hybrid motion+appear",
        "Camera motion comp.",
    ],
}

# ── Fatality Statistics ───────────────────────────────────────────────
FATALITY_STATS = {
    "non_intersection_pct": 74,
    "unsignalized_midblock_pct": 93,
    "nighttime_pct": 50,  # "nearly half"
}

# ── Treatment Cost Ranges ─────────────────────────────────────────────
TREATMENT_COSTS = {
    "treatments": ["Marked\nCrosswalk", "RRFB", "PHB", "MPS"],
    "cost_low_k": [2, 15, 80, 80],   # in thousands USD
    "cost_high_k": [5, 30, 150, 150],  # in thousands USD
}

# ── Site Summary ──────────────────────────────────────────────────────
SITE_INFO = {
    "site1": {
        "name": "UMD Campus\n(Dining Hall)",
        "duration_hrs": 9,
        "time_window": "2:00 AM - 11:00 AM",
        "conditions": "Overnight + morning surge",
    },
    "site2": {
        "name": "Park Road",
        "duration_hrs": 17,
        "time_window": "1:00 PM - 5:00 AM",
        "conditions": "Full day-night cycle",
    },
}
