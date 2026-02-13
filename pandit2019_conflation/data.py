"""Data from Pandit, Kaushik & Cirillo (2019) TRR paper for scene consumption.

All numbers are from:
  Pandit, Kaushik & Cirillo (2019). Coupling NPMRDS and HPMS Datasets
  on a Geospatial Level. Transportation Research Record, 2673(4), 583-592.
"""

from __future__ import annotations

import os

# ── Figure paths ──────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "pandit2019_figures"
)


def fig_path(name: str) -> str:
    """Return absolute path to a figure PNG."""
    return os.path.join(FIGURES_DIR, name)


# ── Paper metadata ────────────────────────────────────────────────────────
PAPER_INFO = {
    "authors": "Pandit, Kaushik & Cirillo",
    "year": 2019,
    "title": "Coupling NPMRDS and HPMS Datasets on a Geospatial Level",
    "journal": "Transportation Research Record",
    "volume": "2673(4)",
    "pages": "583-592",
    "doi": "10.1177/0361198119838983",
}

# ── Study region ──────────────────────────────────────────────────────────
STUDY_REGION = {
    "states": ["Delaware", "Maryland", "Washington DC"],
    "npmrds_year": 2017,
    "hpms_year": 2016,
}

# ── Five scoring measures and weights ─────────────────────────────────────
SCORING_MEASURES = {
    "names": [
        "Angular\nParallelism",
        "Frechet\nDistance",
        "Hausdorff\nDistance",
        "Road Number\n(Levenshtein)",
        "Road Name\n(Levenshtein)",
    ],
    "short_names": ["Angle", "Frechet", "Hausdorff", "Rd Number", "Rd Name"],
    "symbols": ["A", "F", "H", "U", "N"],
    "weights": [3, 3, 2, 4, 1],
    "weight_rationale": [
        "As discriminating as Frechet",
        "Stronger curve similarity",
        "Weaker curve similarity",
        "Most consistent semantic data",
        "Inconsistent in HPMS",
    ],
    "types": ["geometric", "geometric", "geometric", "semantic", "semantic"],
}

# ── Key results ───────────────────────────────────────────────────────────
RESULTS = {
    "excess_tmc_pct": 5.11,  # TMC segments without HPMS match
    "missing_npmrds_pct": 3.10,  # HPMS segments without NPMRDS match
    "buffer_size_m": 150,  # search buffer in meters
    "score_percentile_cutoff": 0.5,  # top 0.5% discarded
    "exponential_bins": 50,
    "smallest_bin": (112, 125),
    "largest_bin": (246_364, 265_272),
}

# ── Frechet vs Hausdorff comparison ───────────────────────────────────────
DISTANCE_COMPARISON = {
    "frechet": {
        "name": "Frechet Distance",
        "property": "No backtracking",
        "metaphor": "Dog walking on leash",
        "strength": "Stronger (order-aware)",
        "complexity": "O(mn)",
    },
    "hausdorff": {
        "name": "Hausdorff Distance",
        "property": "Backtracking allowed",
        "metaphor": "Closest-point sweep",
        "strength": "Weaker (set-based)",
        "complexity": "O(mn)",
    },
}

# ── Mathematical hierarchy (for scenes 6-8) ──────────────────────────────
MATH_HIERARCHY = {
    "local": {
        "label": "Local Features",
        "methods": ["Frechet", "Hausdorff", "Angular", "Levenshtein"],
        "paper": "Pandit et al. (2019)",
        "what": "Compare individual segment pairs",
    },
    "topology": {
        "label": "Graph Topology",
        "methods": ["APSG", "Node-Arc", "Network structure"],
        "paper": "Kim et al. (2022)",
        "what": "Exploit network connectivity",
    },
    "optimal_transport": {
        "label": "Optimal Transport",
        "methods": ["Wasserstein", "Hungarian", "Network flow"],
        "paper": "Future direction",
        "what": "Global mass-preserving assignment",
    },
}

# ── Connection to tracking (bridge to SHAUM703/KF series) ────────────────
TRACKING_CONNECTION = {
    "conflation_matching": "Min-cost segment assignment",
    "sort_matching": "Hungarian algorithm (bounding boxes)",
    "shared_math": "Both are optimal assignment problems",
    "frechet_wasserstein": "2-Wasserstein = Levy-Frechet metric on distributions",
}
