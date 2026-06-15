"""
src/confidence_estimator.py
----------------------------
Phase 3 – Derive a calibrated confidence score in [0, 1] for each alignment.

Three independent sub-scores are blended:

1. overlap_score  (weight W_OVERLAP)
   The raw fraction of polygon-boundary pixels that land on an edge.
   This is the primary signal – high overlap means the shifted boundary
   closely follows a real field edge.

2. margin_score   (weight W_MARGIN)
   Normalised gap between the best and second-best alignment scores.
   A large margin means the best translation stands out clearly; a small
   margin suggests the scoring landscape is flat / ambiguous.
   Formula:  margin_score = clip(best - second, 0, MARGIN_CLIP) / MARGIN_CLIP

3. area_score     (weight W_AREA)
   Penalises cases where the shifted polygon's area differs significantly
   from the original (this can happen if the polygon is partially clipped
   by the raster boundary during scoring).
   Formula:  1.0 if ratio in AREA_RATIO_BAND, else smooth falloff.

Final confidence = W_OVERLAP * overlap + W_MARGIN * margin + W_AREA * area

Calibration notes
-----------------
* The weights are empirically set in config.py; they should be tuned on a
  labelled validation set when ground truth is available.
* Because we use a simple linear blend, the final score is interpretable and
  each component can be logged separately for debugging.
* If a threshold-free calibration (Platt scaling, isotonic regression) were
  available, we would fit it on the validation set and save the calibrator.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from shapely.geometry.base import BaseGeometry

import config
from src.alignment_scorer import AlignmentResult


@dataclass
class ConfidenceResult:
    confidence: float        # final blended score [0, 1]
    overlap_sub: float       # sub-score 1
    margin_sub: float        # sub-score 2
    area_sub: float          # sub-score 3


def compute_confidence(
    alignment: AlignmentResult,
    original_geom: BaseGeometry,
) -> ConfidenceResult:
    """
    Compute a calibrated confidence for one plot's alignment.

    Parameters
    ----------
    alignment     : output of find_best_alignment()
    original_geom : the original (un-shifted) polygon
    """
    # ── Sub-score 1: overlap ───────────────────────────────────────────────
    overlap_sub = float(np.clip(alignment.score, 0.0, 1.0))

    # ── Sub-score 2: margin ────────────────────────────────────────────────
    margin = alignment.score - alignment.second_score
    margin_sub = float(np.clip(margin, 0.0, config.MARGIN_CLIP) / config.MARGIN_CLIP)

    # ── Sub-score 3: area consistency ─────────────────────────────────────
    orig_area    = original_geom.area
    shifted_area = alignment.shifted_geom.area
    if orig_area > 0:
        ratio = shifted_area / orig_area
    else:
        ratio = 1.0

    lo, hi = config.AREA_RATIO_BAND
    if lo <= ratio <= hi:
        area_sub = 1.0
    else:
        # Smooth falloff: 1 - (deviation beyond band) / band_width
        deviation = max(lo - ratio, ratio - hi)
        band_half = (hi - lo) / 2
        area_sub = float(max(0.0, 1.0 - deviation / band_half))

    # ── Weighted blend ────────────────────────────────────────────────────
    confidence = (
        config.W_OVERLAP * overlap_sub
        + config.W_MARGIN  * margin_sub
        + config.W_AREA    * area_sub
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return ConfidenceResult(
        confidence   = confidence,
        overlap_sub  = overlap_sub,
        margin_sub   = margin_sub,
        area_sub     = area_sub,
    )
