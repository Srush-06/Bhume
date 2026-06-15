"""
src/data_loader.py
------------------
Responsible for loading all three input artefacts:
  • input.geojson   – official (possibly drifted) cadastral plot polygons
  • imagery.tif     – multi-band satellite imagery
  • boundaries.tif  – single-band raster of rough field-edge hints

Design decisions
----------------
* We reproject every vector layer to the CRS of the imagery raster so that
  pixel arithmetic and metre-based offset searches share one coordinate frame.
* We expose the raw rasterio DatasetReader objects rather than loading all
  pixel data into RAM – patches are read on demand in patch_extractor.py.
* example_truths.geojson is loaded only when the file exists (optional).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import rasterio

log = logging.getLogger(__name__)


# ── public API ─────────────────────────────────────────────────────────────

def load_plots(geojson_path: Path) -> gpd.GeoDataFrame:
    """
    Load the input plot GeoJSON and ensure it has a 'plot_number' column.

    If the source file uses a different identifier column (e.g. 'id', 'fid',
    'parcel_id') we detect it heuristically and rename it.
    """
    gdf = gpd.read_file(geojson_path)
    log.info("Loaded %d plots from %s", len(gdf), geojson_path)

    # Normalise the plot identifier column.
    gdf = _ensure_plot_number(gdf)

    # Drop any rows with null or empty geometries.
    before = len(gdf)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    if len(gdf) < before:
        log.warning("Dropped %d rows with null/empty geometry.", before - len(gdf))

    return gdf


def load_imagery(tif_path: Path) -> rasterio.DatasetReader:
    """Open the satellite imagery TIF for lazy patch reading."""
    ds = rasterio.open(tif_path)
    log.info(
        "Opened imagery %s  CRS=%s  size=%dx%d  bands=%d",
        tif_path, ds.crs, ds.width, ds.height, ds.count,
    )
    return ds


def load_boundaries(tif_path: Path) -> rasterio.DatasetReader:
    """Open the field-boundary hint raster for lazy patch reading."""
    ds = rasterio.open(tif_path)
    log.info(
        "Opened boundaries %s  CRS=%s  size=%dx%d",
        tif_path, ds.crs, ds.width, ds.height,
    )
    return ds


def load_example_truths(geojson_path: Path) -> Optional[gpd.GeoDataFrame]:
    """
    Load ground-truth geometries used to estimate the global village drift.
    Returns None if the file does not exist (drift estimation is then skipped).
    """
    if not geojson_path.exists():
        log.info("example_truths.geojson not found – skipping global drift estimation.")
        return None

    gdf = gpd.read_file(geojson_path)
    gdf = _ensure_plot_number(gdf)
    log.info("Loaded %d example truths from %s", len(gdf), geojson_path)
    return gdf


def reproject_to_imagery_crs(
    gdf: gpd.GeoDataFrame,
    imagery_ds: rasterio.DatasetReader,
) -> gpd.GeoDataFrame:
    """
    Reproject a GeoDataFrame to match the CRS of the imagery raster.

    We use the imagery CRS as the single source of truth because:
    1. Pixel-space arithmetic (patch extraction, edge scoring) lives there.
    2. Metre-based offset searches are meaningful in a projected CRS.
    """
    target_crs = imagery_ds.crs
    if gdf.crs is None:
        log.warning("GeoDataFrame has no CRS – assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    if gdf.crs != target_crs:
        log.info("Reprojecting plots from %s → %s", gdf.crs, target_crs)
        gdf = gdf.to_crs(target_crs)

    return gdf


# ── helpers ────────────────────────────────────────────────────────────────

_CANDIDATE_ID_COLS = [
    "plot_number", "plot_no", "parcel_id", "id", "fid",
    "objectid", "gid", "plot_id",
]


def _ensure_plot_number(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Guarantee the frame has a column named 'plot_number'.
    Searches known alternatives; falls back to the row index.
    """
    cols_lower = {c.lower(): c for c in gdf.columns}

    for candidate in _CANDIDATE_ID_COLS:
        if candidate in cols_lower:
            original = cols_lower[candidate]
            if original != "plot_number":
                gdf = gdf.rename(columns={original: "plot_number"})
                log.info("Renamed column '%s' → 'plot_number'", original)
            return gdf

    # Last resort: use the integer index.
    log.warning(
        "No recognised plot-ID column found; using row index as 'plot_number'."
    )
    gdf = gdf.copy()
    gdf["plot_number"] = gdf.index.astype(str)
    return gdf
