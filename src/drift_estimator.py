"""
src/drift_estimator.py
-----------------------
Phase 1 – Estimate a single village-wide (dx, dy) translation in metres.

Why a global drift?
-------------------
Cadastral datasets are often digitised from scanned paper maps that were
subsequently geo-referenced as a single rigid block.  The whole village
therefore shifts by roughly the same vector.  Knowing this baseline narrows
the per-plot search window dramatically (Phase 2) and improves robustness.

Algorithm
---------
1. For each (input_plot, truth_plot) pair sharing the same plot_number,
   compute the centroid-to-centroid displacement.
2. Collect all (dx, dy) values.
3. Return the *median* displacement – robust to the handful of outlier plots
   that may have been individually re-surveyed.
4. If no truths are available (file missing) return (0, 0) – the search grid
   in Phase 2 is then centred on zero.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import geopandas as gpd

import config

log = logging.getLogger(__name__)

DriftVector = Tuple[float, float]   # (dx_m, dy_m) in the imagery CRS units


def estimate_global_drift(
    plots: gpd.GeoDataFrame,
    truths: gpd.GeoDataFrame | None,
) -> DriftVector:
    """
    Compute the median centroid displacement from input plots to truth plots.

    Parameters
    ----------
    plots  : reprojected input plots (same CRS as imagery)
    truths : example truths in the same CRS, or None

    Returns
    -------
    (dx, dy) in CRS units (usually metres for a projected CRS).
    """
    if truths is None or truths.empty:
        log.info("No example truths – global drift set to (0, 0).")
        return (0.0, 0.0)

    # Merge on plot_number (inner join keeps only matched pairs).
    merged = plots[["plot_number", "geometry"]].merge(
        truths[["plot_number", "geometry"]],
        on="plot_number",
        suffixes=("_input", "_truth"),
    )

    if merged.empty:
        log.warning(
            "No matching plot_numbers between input and truths – "
            "global drift set to (0, 0)."
        )
        return (0.0, 0.0)

    displacements = []
    for _, row in merged.iterrows():
        c_in  = row["geometry_input"].centroid
        c_tr  = row["geometry_truth"].centroid

        # Skip if either centroid is empty / NaN (failed reprojection).
        import math
        try:
            cx_in, cy_in = c_in.x, c_in.y
            cx_tr, cy_tr = c_tr.x, c_tr.y
        except Exception:
            log.debug("Skipping pair plot_number=%s – invalid centroid.", row["plot_number"])
            continue

        if any(math.isnan(v) for v in (cx_in, cy_in, cx_tr, cy_tr)):
            log.debug("Skipping pair plot_number=%s – NaN centroid.", row["plot_number"])
            continue

        dx = cx_tr - cx_in
        dy = cy_tr - cy_in

        dist = np.hypot(dx, dy)
        if dist > config.DRIFT_OUTLIER_THRESHOLD_M:
            log.debug(
                "Skipping outlier pair plot_number=%s  dist=%.1f m",
                row["plot_number"], dist,
            )
            continue
        displacements.append((dx, dy))

    if not displacements:
        log.warning("All pairs were outliers – global drift set to (0, 0).")
        return (0.0, 0.0)

    arr = np.array(displacements)
    dx_med = float(np.median(arr[:, 0]))
    dy_med = float(np.median(arr[:, 1]))

    log.info(
        "Global drift estimated from %d pairs:  dx=%.2f m  dy=%.2f m",
        len(displacements), dx_med, dy_med,
    )
    return (dx_med, dy_med)
