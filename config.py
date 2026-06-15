"""
config.py
---------
Central configuration for all pipeline hyper-parameters.
Keeping every magic number here makes tuning easy without touching any module.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR          = Path("data")          # default input bundle location
OUTPUT_DIR        = Path("output")        # predictions written here

INPUT_GEOJSON     = DATA_DIR / "input.geojson"
IMAGERY_TIF       = DATA_DIR / "imagery.tif"
BOUNDARIES_TIF    = DATA_DIR / "boundaries.tif"
EXAMPLE_TRUTHS    = DATA_DIR / "example_truths.geojson"   # optional
PREDICTIONS_OUT   = OUTPUT_DIR / "predictions.geojson"

# ── Phase 1 – Global drift estimation ─────────────────────────────────────
# Maximum plausible drift (metres).  Plots shifted further are treated as
# outliers and excluded from the global median calculation.
DRIFT_OUTLIER_THRESHOLD_M = 50.0

# ── Phase 2 – Per-plot alignment search ───────────────────────────────────
# Patch padding around each plot bounding box (pixels).
PATCH_PADDING_PX   = 40

# Canny edge detection thresholds (applied to grayscale imagery).
CANNY_LOW          = 30
CANNY_HIGH         = 90

# Grid of candidate translations to test (metres).
# We search ±SEARCH_RADIUS_M in steps of SEARCH_STEP_M around the global drift.
SEARCH_RADIUS_M    = 12.0
SEARCH_STEP_M      = 2.0

# Weight of boundary-hint channel relative to imagery edges.
BOUNDARY_WEIGHT    = 0.4      # 0 = only imagery edges, 1 = only boundaries

# Morphological dilation radius applied to edge maps (pixels) before scoring.
EDGE_DILATE_PX     = 2

# ── Phase 3 – Confidence calibration ──────────────────────────────────────
# Weights for the three confidence sub-scores.
W_OVERLAP          = 0.50     # raw overlap fraction
W_MARGIN           = 0.30     # margin between best and second-best score
W_AREA             = 0.20     # area consistency (shifted vs original)

# Margin is clipped to [0, MARGIN_CLIP] then normalised.
MARGIN_CLIP        = 0.20

# Area ratio outside this band gets a penalty.
AREA_RATIO_BAND    = (0.85, 1.15)   # ±15 %

# ── Phase 4 – Decision thresholds ─────────────────────────────────────────
CONFIDENCE_CORRECT = 0.55     # ≥ this → "corrected"
CONFIDENCE_FLAG    = 0.30     # ≥ this → "flagged"; below → omit
