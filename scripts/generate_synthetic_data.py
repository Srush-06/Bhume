"""
scripts/generate_synthetic_data.py
------------------------------------
Creates a minimal but realistic synthetic village bundle so the pipeline
can be run immediately without real data.

What it generates
-----------------
data/
  input.geojson       – 20 rectangular plots in WGS-84, each shifted
                        by a known (dx, dy) from the "true" UTM position
  imagery.tif         – synthetic single-band raster (UTM, field edges)
  boundaries.tif      – single-band field-edge hint raster (UTM)
  example_truths.geojson – 5 plots at their correct (true) positions

Notes
-----
* GeoJSON files are written in EPSG:4326 (WGS-84) as the GeoJSON spec
  (RFC 7946) requires.  Geometries are built in UTM and reprojected before
  writing so coordinates are valid lat/lon values.
* Raster files stay in UTM (EPSG:32644 – UTM zone 44N, central India).

Usage
-----
    python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.affinity import translate
from shapely.geometry import box

# ── Settings ────────────────────────────────────────────────────────────────
SEED          = 42
N_PLOTS       = 20
N_TRUTHS      = 5            # plots that also appear in example_truths
TRUE_DX       = 8.0          # village-wide drift (metres) – X
TRUE_DY       = -5.0         # village-wide drift (metres) – Y
NOISE_SIGMA   = 1.5          # per-plot Gaussian noise around the true drift
PIXEL_SIZE    = 1.0          # metres per pixel
IMG_SIZE      = 600          # pixels (square raster)

# UTM zone 44N origin — approx. 20.0°N, 77.0°E (Nagpur, India).
# These are valid easting/northing values; PROJ is happy with them.
ORIGIN_X      = 530_000.0    # metres Easting
ORIGIN_Y      = 2_211_000.0  # metres Northing (positive, valid UTM)
UTM_CRS       = "EPSG:32644"
DATA_DIR      = Path("data")

random.seed(SEED)
np.random.seed(SEED)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. True plot geometries on a UTM grid ─────────────────────────────
    plot_w, plot_h = 40.0, 30.0   # metres
    margin         = 10.0
    cols           = 5

    true_plots = []
    for i in range(N_PLOTS):
        col = i % cols
        row = i // cols
        x0  = ORIGIN_X + col * (plot_w + margin)
        y0  = ORIGIN_Y - row * (plot_h + margin)
        true_plots.append({
            "plot_number": str(i + 1),
            "geometry":    box(x0, y0 - plot_h, x0 + plot_w, y0),
        })

    # ── 2. Drifted "official" input plots ─────────────────────────────────
    input_plots = []
    for p in true_plots:
        dx = TRUE_DX + random.gauss(0, NOISE_SIGMA)
        dy = TRUE_DY + random.gauss(0, NOISE_SIGMA)
        input_plots.append({
            "plot_number": p["plot_number"],
            "geometry":    translate(p["geometry"], xoff=dx, yoff=dy),
        })

    # ── 3. Write GeoJSON files in EPSG:4326 ───────────────────────────────
    _write_geojson_wgs84(input_plots, DATA_DIR / "input.geojson")
    print(f"✓ Wrote {DATA_DIR / 'input.geojson'}  ({N_PLOTS} plots)")

    truths = true_plots[:N_TRUTHS]
    _write_geojson_wgs84(truths, DATA_DIR / "example_truths.geojson")
    print(f"✓ Wrote {DATA_DIR / 'example_truths.geojson'}  ({N_TRUTHS} plots)")

    # ── 4. Build rasters (stay in UTM) ────────────────────────────────────
    transform = from_origin(ORIGIN_X, ORIGIN_Y, PIXEL_SIZE, PIXEL_SIZE)
    crs       = CRS.from_epsg(32644)

    img = _generate_imagery(true_plots, IMG_SIZE, transform)
    bnd = _generate_boundaries(true_plots, IMG_SIZE, transform)

    _write_tif(img, DATA_DIR / "imagery.tif",    transform, crs)
    _write_tif(bnd, DATA_DIR / "boundaries.tif", transform, crs)
    print(f"✓ Wrote {DATA_DIR / 'imagery.tif'} and {DATA_DIR / 'boundaries.tif'}")
    print("\nAll synthetic data written to ./data/  — run:  python main.py")


# ── helpers ─────────────────────────────────────────────────────────────────

def _write_geojson_wgs84(plots: list, path: Path) -> None:
    """
    Build a GeoDataFrame in UTM, reproject to WGS-84, save as GeoJSON.

    The GeoJSON spec (RFC 7946) mandates EPSG:4326 coordinates.
    Our synthetic plots are built in UTM space; we reproject here so
    geopandas reads them back as valid lat/lon when loading the pipeline.
    """
    gdf = gpd.GeoDataFrame(
        [{"plot_number": p["plot_number"]} for p in plots],
        geometry=[p["geometry"] for p in plots],
        crs=UTM_CRS,
    )
    gdf.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")


def _generate_imagery(true_plots: list, size: int, transform) -> np.ndarray:
    """Synthetic NDVI-like image with dark field-boundary lines."""
    img = np.random.randint(80, 160, (size, size), dtype=np.uint8)
    inv = ~transform

    for p in true_plots:
        minx, miny, maxx, maxy = p["geometry"].bounds
        c0, r0 = inv * (minx, maxy)
        c1, r1 = inv * (maxx, miny)
        r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)
        r0, r1 = max(0, r0), min(size - 1, r1)
        c0, c1 = max(0, c0), min(size - 1, c1)
        if r1 <= r0 or c1 <= c0:
            continue
        img[r0:r1, c0:c1] = np.random.randint(120, 200, (r1 - r0, c1 - c0), dtype=np.uint8)
        img[r0, c0:c1] = 30
        img[r1, c0:c1] = 30
        img[r0:r1, c0] = 30
        img[r0:r1, c1] = 30

    return img


def _generate_boundaries(true_plots: list, size: int, transform) -> np.ndarray:
    """Boundary hint raster: bright where plot edges are, plus noise."""
    bnd = np.zeros((size, size), dtype=np.uint8)
    inv = ~transform

    for p in true_plots:
        minx, miny, maxx, maxy = p["geometry"].bounds
        c0, r0 = inv * (minx, maxy)
        c1, r1 = inv * (maxx, miny)
        r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)
        r0, r1 = max(0, r0), min(size - 1, r1)
        c0, c1 = max(0, c0), min(size - 1, c1)
        if r1 <= r0 or c1 <= c0:
            continue
        bnd[r0, c0:c1] = 255
        bnd[r1, c0:c1] = 255
        bnd[r0:r1, c0] = 255
        bnd[r0:r1, c1] = 255

    noise = np.random.randint(0, 40, (size, size), dtype=np.uint8)
    bnd = np.clip(bnd.astype(np.int16) + noise - 20, 0, 255).astype(np.uint8)
    return bnd


def _write_tif(arr: np.ndarray, path: Path, transform, crs) -> None:
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr, 1)


if __name__ == "__main__":
    main()
