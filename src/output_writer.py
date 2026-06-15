"""
src/output_writer.py
--------------------
Serialise PlotDecision objects to a GeoJSON FeatureCollection
(predictions.geojson) in the output directory.

Output schema per feature
--------------------------
  plot_number  : str
  status       : "corrected" | "flagged"
  confidence   : float | null
  method_note  : str
  geometry     : GeoJSON geometry (same CRS as input imagery)

Design notes
------------
* We write via geopandas.to_file() with driver="GeoJSON" so that the CRS
  metadata is embedded correctly.
* Before writing we reproject back to EPSG:4326 (WGS-84 geographic) unless
  the input was already in a geographic CRS – most consumers expect lat/lon.
* The output directory is created if it does not exist.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import geopandas as gpd
from shapely.geometry import mapping

from src.decision_maker import PlotDecision

log = logging.getLogger(__name__)


def write_predictions(
    decisions: List[PlotDecision],
    output_path: Path,
    imagery_crs,
    reproject_to_wgs84: bool = True,
) -> None:
    """
    Write a list of PlotDecision objects to *output_path* as GeoJSON.

    Parameters
    ----------
    decisions          : list of PlotDecision (corrected or flagged)
    output_path        : destination file path (created if absent)
    imagery_crs        : CRS of the geometries (rasterio CRS object or string)
    reproject_to_wgs84 : if True, reproject to EPSG:4326 before writing
    """
    if not decisions:
        log.warning("No decisions to write – output file will be empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in decisions:
        rows.append(
            {
                "plot_number": d.plot_number,
                "status":      d.status,
                "confidence":  d.confidence,
                "method_note": d.method_note,
                "geometry":    d.geometry,
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=imagery_crs)

    if reproject_to_wgs84 and not _is_geographic(gdf.crs):
        log.info("Reprojecting output to EPSG:4326 (WGS-84).")
        gdf = gdf.to_crs("EPSG:4326")

    gdf.to_file(output_path, driver="GeoJSON")
    log.info("Wrote %d predictions → %s", len(gdf), output_path)


# ── helpers ────────────────────────────────────────────────────────────────

def _is_geographic(crs) -> bool:
    """Return True if *crs* is a geographic (lat/lon) CRS."""
    try:
        return crs.is_geographic
    except Exception:
        return False
