"""
src/alignment_scorer.py
------------------------
Phase 2 core: search for the best (dx, dy) translation that aligns a plot
polygon to field edges visible in the imagery.

Algorithm
---------
For each candidate translation (dx, dy) in a grid centred on the global drift:

1.  Shift the polygon by (dx, dy) in CRS units.
2.  Rasterise the *boundary* of the shifted polygon into the patch pixel grid.
3.  Compute the fraction of boundary pixels that overlap with the edge map
    (i.e. mask[boundary_pixels > 0] > 0).
4.  Record the score.

The best translation is the one with the highest overlap score.

Why boundary rasterisation (not filled polygon)?
------------------------------------------------
Field edges appear as thin lines in the imagery.  Scoring the overlap of the
polygon *outline* with the detected edge map is much more discriminative than
scoring a filled polygon interior, which would always score high simply because
it covers a large area.

Rasterisation uses OpenCV's polylines / fillPoly on a uint8 canvas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.transform import rowcol
from shapely.geometry.base import BaseGeometry
from shapely.affinity import translate

import config

log = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    dx: float               # best translation in CRS units (metres)
    dy: float
    score: float            # overlap fraction of the best candidate [0, 1]
    second_score: float     # overlap fraction of the runner-up [0, 1]
    all_scores: np.ndarray  # full score grid (for debugging)
    shifted_geom: BaseGeometry


def find_best_alignment(
    geom: BaseGeometry,
    edge_map: np.ndarray,
    win_transform: rasterio.Affine,
    global_dx: float,
    global_dy: float,
    search_radius: float = config.SEARCH_RADIUS_M,
    search_step: float   = config.SEARCH_STEP_M,
) -> AlignmentResult:
    """
    Exhaustive grid search over candidate (dx, dy) translations.

    Parameters
    ----------
    geom          : original plot polygon in the imagery CRS
    edge_map      : fused edge image (H, W) uint8
    win_transform : affine transform  pixel ↔ CRS  for this patch
    global_dx/dy  : centre of the search grid (global village drift)
    search_radius : half-width of the search grid in CRS units
    search_step   : step size in CRS units

    Returns
    -------
    AlignmentResult with the best translation and diagnostic scores
    """
    offsets = _build_search_grid(global_dx, global_dy, search_radius, search_step)
    scores  = []

    for (dx, dy) in offsets:
        shifted = translate(geom, xoff=dx, yoff=dy)
        bnd_mask = _rasterise_boundary(shifted, edge_map.shape, win_transform)
        score = _overlap_score(edge_map, bnd_mask)
        scores.append(score)

    scores_arr = np.array(scores)

    best_idx    = int(np.argmax(scores_arr))
    best_dx, best_dy = offsets[best_idx]
    best_score  = float(scores_arr[best_idx])

    # Runner-up: best score among all *other* candidates.
    if len(scores_arr) > 1:
        masked = scores_arr.copy()
        masked[best_idx] = -1
        second_score = float(masked.max())
    else:
        second_score = 0.0

    shifted_geom = translate(geom, xoff=best_dx, yoff=best_dy)

    return AlignmentResult(
        dx           = best_dx,
        dy           = best_dy,
        score        = best_score,
        second_score = second_score,
        all_scores   = scores_arr,
        shifted_geom = shifted_geom,
    )


# ── helpers ────────────────────────────────────────────────────────────────

def _build_search_grid(
    cx: float, cy: float, radius: float, step: float,
) -> List[Tuple[float, float]]:
    """Return a flat list of (dx, dy) candidate offsets."""
    vals = np.arange(-radius, radius + step / 2, step)
    grid = [(cx + dx, cy + dy) for dx in vals for dy in vals]
    return grid


def _rasterise_boundary(
    geom: BaseGeometry,
    shape: Tuple[int, int],
    transform: rasterio.Affine,
) -> np.ndarray:
    """
    Rasterise the *outline* of *geom* into a binary mask of size *shape*.

    Uses OpenCV polylines for speed; handles MultiPolygon transparently.
    """
    h, w = shape
    canvas = np.zeros((h, w), dtype=np.uint8)

    polys = _extract_rings(geom)
    for ring in polys:
        if len(ring) < 2:
            continue
        # Convert CRS coordinates → pixel indices.
        pts = _coords_to_pixels(ring, transform, h, w)
        if pts is None or len(pts) < 2:
            continue
        cv2.polylines(canvas, [pts], isClosed=True, color=255, thickness=1)

    return canvas


def _overlap_score(edge_map: np.ndarray, boundary_mask: np.ndarray) -> float:
    """
    Fraction of boundary pixels that coincide with the edge map.

    Score = |{p : boundary[p]>0 AND edge[p]>0}| / |{p : boundary[p]>0}|

    Returns 0 if the boundary mask is empty (off-screen translation).
    """
    bnd_pixels = boundary_mask > 0
    n_bnd = int(bnd_pixels.sum())
    if n_bnd == 0:
        return 0.0
    n_hit = int((edge_map[bnd_pixels] > 0).sum())
    return n_hit / n_bnd


def _extract_rings(geom: BaseGeometry) -> List[List[Tuple[float, float]]]:
    """Extract exterior rings from Polygon or MultiPolygon."""
    from shapely.geometry import Polygon, MultiPolygon
    rings: List[List[Tuple[float, float]]] = []

    if isinstance(geom, Polygon):
        rings.append([(x, y) for x, y, *_ in geom.exterior.coords])
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            rings.append([(x, y) for x, y, *_ in part.exterior.coords])
    return rings


def _coords_to_pixels(
    coords: List[Tuple[float, float]],
    transform: rasterio.Affine,
    h: int, w: int,
) -> np.ndarray | None:
    """
    Convert a sequence of CRS (x, y) coordinates to pixel (col, row) pairs,
    clipped to the patch bounds.
    """
    inv = ~transform
    pts = []
    for (x, y) in coords:
        col, row = inv * (x, y)
        # Keep even slightly out-of-bounds points; clamp to edge.
        col = int(np.clip(round(col), 0, w - 1))
        row = int(np.clip(round(row), 0, h - 1))
        pts.append([col, row])

    if len(pts) < 2:
        return None
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
