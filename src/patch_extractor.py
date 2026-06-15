"""
src/patch_extractor.py
-----------------------
Read pixel patches from the raster datasets for a given plot polygon.

Design decisions
----------------
* We use the plot's bounding box plus a configurable padding (PATCH_PADDING_PX)
  so that edge detectors have context outside the polygon boundary.
* Both the imagery and boundary rasters are clipped to the *same* Window so
  that all subsequent pixel operations share one coordinate frame.
* We return raw numpy arrays; no colour conversion happens here – that is the
  responsibility of the edge_detector.
* The function also returns the rasterio `Transform` of the extracted window
  so callers can convert between pixel offsets and CRS coordinates.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import rasterio
import rasterio.windows
from rasterio.transform import from_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

import config

log = logging.getLogger(__name__)

PatchResult = Tuple[
    np.ndarray,        # imagery_patch  (H, W) or (H, W, C)
    np.ndarray,        # boundary_patch (H, W)
    rasterio.Affine,   # window transform (pixel → CRS)
]


def patch_for_plot(
    geom: BaseGeometry,
    imagery_ds: rasterio.DatasetReader,
    boundaries_ds: rasterio.DatasetReader,
    padding_px: int = config.PATCH_PADDING_PX,
) -> PatchResult | None:
    """
    Extract imagery and boundary raster patches for *geom*.

    Parameters
    ----------
    geom         : plot polygon in the imagery CRS
    imagery_ds   : open rasterio dataset for the satellite imagery
    boundaries_ds: open rasterio dataset for the boundary hints
    padding_px   : extra pixel border added around the bounding box

    Returns
    -------
    (imagery_patch, boundary_patch, window_transform)
    or None if the geometry falls entirely outside the raster.
    """
    # ── 1. Convert polygon bbox to a rasterio Window ───────────────────────
    minx, miny, maxx, maxy = geom.bounds
    try:
        row_off, col_off = imagery_ds.index(minx, maxy)   # top-left corner
        row_end, col_end = imagery_ds.index(maxx, miny)   # bottom-right corner
    except Exception:
        log.debug("Geometry outside imagery extent – skipping.")
        return None

    # Add padding and clamp to raster size.
    row_off = max(0, row_off - padding_px)
    col_off = max(0, col_off - padding_px)
    row_end = min(imagery_ds.height, row_end + padding_px)
    col_end = min(imagery_ds.width,  col_end + padding_px)

    if row_end <= row_off or col_end <= col_off:
        log.debug("Degenerate window after clamping – skipping.")
        return None

    window = rasterio.windows.Window(
        col_off=col_off,
        row_off=row_off,
        width=col_end  - col_off,
        height=row_end - row_off,
    )

    # ── 2. Read imagery patch ──────────────────────────────────────────────
    img_data = imagery_ds.read(window=window)   # (bands, H, W)

    # Convert to (H, W, C) uint8 for OpenCV; handle both 1-band and RGB.
    img_data = np.clip(img_data, 0, 255).astype(np.uint8)
    if img_data.shape[0] == 1:
        img_patch = img_data[0]                 # (H, W) grayscale
    else:
        # Take first 3 bands; rasterio is (C,H,W) → numpy needs (H,W,C)
        img_patch = np.transpose(img_data[:3], (1, 2, 0))

    # ── 3. Read boundary patch (reproject window to boundary CRS if needed) ─
    bnd_patch = _read_resampled(boundaries_ds, imagery_ds, window)

    # ── 4. Compute the affine transform for this window ────────────────────
    win_transform = imagery_ds.window_transform(window)

    return img_patch, bnd_patch, win_transform


# ── helpers ────────────────────────────────────────────────────────────────

def _read_resampled(
    src_ds: rasterio.DatasetReader,
    ref_ds: rasterio.DatasetReader,
    ref_window: rasterio.windows.Window,
) -> np.ndarray:
    """
    Read a patch from *src_ds* that corresponds spatially to *ref_window* in
    *ref_ds*, resampling to the same pixel grid if the two rasters differ in
    resolution or CRS.

    Returns a 2-D uint8 array of shape (H, W).
    """
    h = int(ref_window.height)
    w = int(ref_window.width)

    # Compute the geographic bounds of the reference window.
    bounds = rasterio.windows.bounds(ref_window, ref_ds.transform)

    try:
        from rasterio.warp import reproject, Resampling
        import rasterio.transform as rtrans

        src_data = src_ds.read(1)   # full raster, single band

        # Build a destination array matching the reference window size.
        dst = np.zeros((h, w), dtype=np.float32)
        dst_transform = from_bounds(*bounds, width=w, height=h)

        reproject(
            source=src_data,
            destination=dst,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=dst_transform,
            dst_crs=ref_ds.crs,
            resampling=Resampling.bilinear,
        )
    except Exception as exc:
        log.debug("Boundary reproject failed (%s) – returning zeros.", exc)
        dst = np.zeros((h, w), dtype=np.float32)

    # Normalise to uint8.
    mn, mx = dst.min(), dst.max()
    if mx > mn:
        dst = (dst - mn) / (mx - mn) * 255.0
    return dst.astype(np.uint8)
