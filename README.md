# Cadastral Boundary Correction Pipeline

> **Take-Home Assignment Solution — Geospatial Computer Vision**

Correct historical cadastral plot boundaries using satellite imagery and classical computer vision — no deep learning required.

---

## Problem Statement

Cadastral maps digitised from scanned paper surveys are often geo-referenced as a single rigid block, introducing a systematic translation offset (drift) between the official plot polygons and the true field boundaries visible in satellite imagery.

This pipeline:
1. Estimates the village-wide drift from a small set of known-correct example plots.
2. Searches for the best per-plot alignment by comparing a rasterised polygon boundary against image edges.
3. Scores each alignment with a calibrated confidence metric.
4. Outputs corrected geometries (high confidence) or flags plots for manual review (medium confidence).

---

## Project Structure

```
bhume/
├── main.py                          # Pipeline entry point
├── config.py                        # All hyper-parameters in one place
├── requirements.txt
│
├── src/
│   ├── data_loader.py               # Load GeoJSON + rasters; normalise CRS
│   ├── drift_estimator.py           # Phase 1 – global village drift
│   ├── patch_extractor.py           # Phase 2 – per-plot raster patch crop
│   ├── edge_detector.py             # Phase 2 – Canny + boundary fusion
│   ├── alignment_scorer.py          # Phase 2 – grid search + overlap score
│   ├── confidence_estimator.py      # Phase 3 – calibrated confidence
│   └── decision_maker.py            # Phase 4 – corrected / flagged / omit
│   └── output_writer.py             # Write predictions.geojson
│
├── scripts/
│   └── generate_synthetic_data.py   # Create a test bundle with known drift
│
├── tests/
│   └── test_pipeline.py             # Unit + integration tests (pytest)
│
└── data/                            # Place your village bundle here
    ├── input.geojson
    ├── imagery.tif
    ├── boundaries.tif
    └── example_truths.geojson       # optional but recommended
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your data

Place the village bundle in `data/`:

```
data/
  input.geojson
  imagery.tif
  boundaries.tif
  example_truths.geojson   ← optional; enables global drift estimation
```

Or generate a synthetic bundle for immediate testing:

```bash
python scripts/generate_synthetic_data.py
```

### 3. Run the pipeline

```bash
python main.py
```

Output is written to `output/predictions.geojson`.

#### Optional flags

```bash
python main.py --data-dir /path/to/bundle --output-dir /path/to/out --debug
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Output Format

Each feature in `predictions.geojson` contains:

| Field | Type | Description |
|-------|------|-------------|
| `plot_number` | string | Matches the input plot identifier |
| `status` | `"corrected"` \| `"flagged"` | Decision outcome |
| `confidence` | float (0–1) or null | Calibrated confidence; null for flagged |
| `method_note` | string | Human-readable explanation |
| `geometry` | GeoJSON geometry | Corrected or original polygon |

---

## Algorithm — Four Phases

### Phase 1 — Global Drift Estimation

**Input:** `example_truths.geojson` (a handful of plots with known-correct positions)

**Method:** For each matched pair *(input plot, truth plot)*, compute the centroid-to-centroid displacement. Take the **median** of all displacements (robust to individual outliers).

**Output:** A single `(dx, dy)` translation in metres that approximates the village-wide registration error.

> **Design rationale:** Cadastral datasets are often geo-referenced as a single rigid body. A global median translation captures this systematic error without fitting a deformation model, which would require far more ground truth.

---

### Phase 2 — Per-Plot Alignment Search

For each plot:

1. **Extract patch** (`patch_extractor.py`)  
   Crop a padded bounding box from both `imagery.tif` and `boundaries.tif` using rasterio windowed reads (memory-efficient, lazy).

2. **Build edge map** (`edge_detector.py`)  
   - Convert imagery to grayscale.  
   - Run **OpenCV Canny** (thin, well-localised edges, noise-robust via internal Gaussian + hysteresis).  
   - Blend with the boundary hint channel using a configurable weight (`BOUNDARY_WEIGHT`).  
   - Apply morphological dilation (`EDGE_DILATE_PX`) to tolerate sub-pixel misregistration.

3. **Grid search** (`alignment_scorer.py`)  
   Test every `(dx, dy)` offset in a grid centred on the global drift, spanning `±SEARCH_RADIUS_M` in steps of `SEARCH_STEP_M`.  
   For each candidate:  
   - Shift the polygon by `(dx, dy)`.  
   - Rasterise the polygon **boundary** (not interior) into the patch coordinate frame.  
   - Score = fraction of boundary pixels that coincide with edge-map pixels.

> **Why boundary rasterisation?** Scoring the polygon outline against edge pixels is far more discriminative than scoring the filled interior. A filled polygon always covers a large area and scores high regardless of alignment.

---

### Phase 3 — Confidence Estimation

Three independent sub-scores (weights configurable in `config.py`):

| Sub-score | Weight | Formula |
|-----------|--------|---------|
| **Overlap** | 0.50 | Raw fraction of boundary pixels on an edge |
| **Margin** | 0.30 | `clip(best − runner_up, 0, 0.20) / 0.20` |
| **Area** | 0.20 | 1.0 if area ratio in [0.85, 1.15], else smooth falloff |

```
confidence = 0.50 × overlap + 0.30 × margin + 0.20 × area
```

- **Overlap** is the primary signal.
- **Margin** rewards unambiguous alignments (the score landscape has a clear peak).
- **Area** penalises off-screen translations where the polygon is partially clipped.

---

### Phase 4 — Decision

| Confidence | Outcome | Geometry emitted |
|------------|---------|-----------------|
| ≥ 0.55 | `corrected` | Shifted polygon |
| 0.30 – 0.55 | `flagged` | Original polygon |
| < 0.30 | *(omitted)* | — |

Thresholds are set in `config.py` (`CONFIDENCE_CORRECT`, `CONFIDENCE_FLAG`).

---

## Confidence Calibration — Design Decisions & Improvement Suggestions

### Current approach

The confidence is a manually weighted linear blend of three interpretable sub-scores. Weights were set by reasoning about their relative information value, not by fitting to data.

### Suggestions for improvement

| Method | When to use |
|--------|-------------|
| **Platt scaling** | If you have ≥50 labelled plots, fit a logistic regression on the raw scores to calibrate probabilities. |
| **Isotonic regression** | Non-parametric, better than Platt when the score distribution is non-monotonic. |
| **Reliability diagrams** | Plot `mean confidence` vs `fraction correct` in bins to diagnose over/under-confidence. |
| **Cross-validation** | Use leave-one-village-out CV to avoid overfitting the threshold to a single village's drift pattern. |
| **Richer features** | Add gradient magnitude variance inside the polygon, SSIM between original and shifted patch, or a histogram of edge orientations as additional confidence signals. |
| **Search refinement** | Replace the coarse grid with a gradient-free optimiser (Nelder-Mead or Powell) initialised at the grid best, for sub-pixel accuracy without a large search budget. |

---

## Hyper-Parameter Reference

All parameters are in [`config.py`](config.py).

| Parameter | Default | Effect |
|-----------|---------|--------|
| `PATCH_PADDING_PX` | 40 | Context pixels around plot bbox |
| `CANNY_LOW / HIGH` | 30 / 90 | Canny thresholds — lower = more edges |
| `SEARCH_RADIUS_M` | 12 | Half-width of offset search grid |
| `SEARCH_STEP_M` | 2 | Grid spacing (smaller = slower but more precise) |
| `BOUNDARY_WEIGHT` | 0.4 | Blend weight for boundary hint raster |
| `EDGE_DILATE_PX` | 2 | Dilation tolerance for edge alignment |
| `CONFIDENCE_CORRECT` | 0.55 | Minimum confidence to mark as corrected |
| `CONFIDENCE_FLAG` | 0.30 | Minimum confidence to include as flagged |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `geopandas` | Vector I/O, CRS management, GeoJSON output |
| `shapely` | Geometry operations (translate, bounds, rings) |
| `rasterio` | Raster I/O, windowed reads, affine transforms |
| `numpy` | Array arithmetic throughout |
| `opencv-python` | Canny, polylines rasterisation, morphology |
| `scipy` | (available for future use, e.g. optimisation) |
| `tqdm` | Progress bar |

---

## License

MIT
