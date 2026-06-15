"""
src/decision_maker.py
----------------------
Phase 4 – Translate a confidence score into one of three outcomes:

  "corrected"  – confidence ≥ CONFIDENCE_CORRECT
                 Emit the shifted geometry.

  "flagged"    – CONFIDENCE_FLAG ≤ confidence < CONFIDENCE_CORRECT
                 Emit the *original* geometry with a warning that manual
                 review is needed.  We do NOT emit a shifted geometry here
                 because a low-confidence shift could make things worse.

  (omitted)    – confidence < CONFIDENCE_FLAG
                 The plot is entirely excluded from the output.
                 This keeps the predictions file clean: only plots where we
                 have at least a minimal signal are included.

Design rationale
----------------
Using two thresholds rather than one gives the consumer a meaningful tristate:
- Corrected plots can be used directly.
- Flagged plots are prioritised for manual review.
- Omitted plots (if any) were too ambiguous to act on.

The thresholds live in config.py and should be calibrated against a labelled
validation set using precision/recall curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from shapely.geometry.base import BaseGeometry

import config
from src.alignment_scorer import AlignmentResult
from src.confidence_estimator import ConfidenceResult

Status = Literal["corrected", "flagged"]


@dataclass
class PlotDecision:
    plot_number: str
    status: Status
    confidence: Optional[float]     # None for flagged plots
    method_note: str
    geometry: BaseGeometry          # corrected or original


def make_decision(
    plot_number: str,
    original_geom: BaseGeometry,
    alignment: AlignmentResult,
    conf_result: ConfidenceResult,
    global_dx: float,
    global_dy: float,
) -> Optional[PlotDecision]:
    """
    Apply threshold rules and return a PlotDecision (or None to omit).

    Parameters
    ----------
    plot_number   : identifier string
    original_geom : the original un-shifted polygon
    alignment     : best alignment found in Phase 2
    conf_result   : confidence breakdown from Phase 3
    global_dx/dy  : global drift used (for method note)

    Returns
    -------
    PlotDecision or None (omit the plot)
    """
    conf = conf_result.confidence

    if conf >= config.CONFIDENCE_CORRECT:
        note = (
            f"Aligned via edge-overlap search. "
            f"Global drift ({global_dx:.1f}, {global_dy:.1f}) m; "
            f"local shift ({alignment.dx:.1f}, {alignment.dy:.1f}) m. "
            f"Overlap={conf_result.overlap_sub:.2f} "
            f"Margin={conf_result.margin_sub:.2f} "
            f"Area={conf_result.area_sub:.2f}."
        )
        return PlotDecision(
            plot_number = plot_number,
            status      = "corrected",
            confidence  = round(conf, 4),
            method_note = note,
            geometry    = alignment.shifted_geom,
        )

    elif conf >= config.CONFIDENCE_FLAG:
        note = (
            f"Low confidence ({conf:.2f}) – manual review recommended. "
            f"Best overlap={conf_result.overlap_sub:.2f} "
            f"Margin={conf_result.margin_sub:.2f}."
        )
        return PlotDecision(
            plot_number = plot_number,
            status      = "flagged",
            confidence  = None,
            method_note = note,
            geometry    = original_geom,   # return original geometry
        )

    else:
        # confidence too low – omit entirely
        return None
