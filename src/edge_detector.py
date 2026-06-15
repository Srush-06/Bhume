"""
src/edge_detector.py
---------------------
Convert raw raster patches into a single normalised edge-probability map.

Design decisions
----------------
1.  **Imagery edges** – We use OpenCV Canny on a grayscale luminance image.
    Canny is preferred over simple Sobel because:
    • It suppresses noise through Gaussian blur.
    • Non-maximum suppression produces thin, crisp edges (1 pixel wide).
    • Hysteresis linking reduces fragmented edges caused by texture.

2.  **Boundary hints** – The boundaries.tif channel already encodes field-edge
    likelihood.  We threshold it lightly and apply a small dilation so it
    behaves like a soft mask rather than a sharp binary edge.

3.  **Fusion** – We compute a weighted sum of the two edge sources.
    BOUNDARY_WEIGHT controls how much the hint raster contributes.
    Having a tunable knob lets us handle cases where the hint raster is
    absent, noisy, or perfectly clean.

4.  **Morphological dilation** – Before scoring we dilate the fused edge map
    slightly (EDGE_DILATE_PX) to tolerate sub-pixel alignment errors between
    the polygon boundary raster and the detected edges.

All outputs are uint8 in [0, 255].  The scoring function in
alignment_scorer.py treats pixels > 0 as "edge present".
"""

from __future__ import annotations

import cv2
import numpy as np

import config


def build_edge_map(
    img_patch: np.ndarray,
    bnd_patch: np.ndarray,
    canny_low: int  = config.CANNY_LOW,
    canny_high: int = config.CANNY_HIGH,
    boundary_weight: float = config.BOUNDARY_WEIGHT,
    dilate_px: int  = config.EDGE_DILATE_PX,
) -> np.ndarray:
    """
    Produce a fused edge-probability map from an imagery patch and a boundary
    hint patch.

    Parameters
    ----------
    img_patch        : (H, W) grayscale  OR  (H, W, 3) RGB uint8
    bnd_patch        : (H, W) uint8 boundary hint
    canny_low/high   : Canny hysteresis thresholds
    boundary_weight  : blend weight for the boundary channel [0, 1]
    dilate_px        : dilation kernel radius in pixels

    Returns
    -------
    edge_map : (H, W) uint8, values in [0, 255]
    """
    # ── 1. Grayscale luminance ─────────────────────────────────────────────
    if img_patch.ndim == 3:
        gray = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_patch.copy()

    # ── 2. Imagery edges via Canny ─────────────────────────────────────────
    # A mild Gaussian blur (3×3) reduces high-frequency texture noise before
    # Canny runs its own internal Gaussian (5×5 by default).
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny   = cv2.Canny(blurred, canny_low, canny_high)  # uint8, 0 or 255

    # ── 3. Boundary hint edges ─────────────────────────────────────────────
    # Treat the boundary channel as an additional edge-strength signal.
    # Normalise to [0,255] float for blending.
    bnd_f   = bnd_patch.astype(np.float32)

    # ── 4. Weighted fusion ────────────────────────────────────────────────
    canny_f = canny.astype(np.float32)
    fused   = (1.0 - boundary_weight) * canny_f + boundary_weight * bnd_f
    fused   = np.clip(fused, 0, 255).astype(np.uint8)

    # ── 5. Morphological dilation ─────────────────────────────────────────
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        fused = cv2.dilate(fused, kernel, iterations=1)

    return fused
