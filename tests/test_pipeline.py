"""
tests/test_pipeline.py
-----------------------
Unit and integration tests for every module.

Run with:
    pytest tests/ -v

The tests use the synthetic data bundle created by
    python scripts/generate_synthetic_data.py

If the synthetic bundle is absent, integration tests are skipped.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure the project root is on the path.
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── fixtures ────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")
DATA_READY = (DATA_DIR / "input.geojson").exists()


@pytest.fixture(scope="session")
def synthetic_data():
    """Generate synthetic data once per session if not already present."""
    if not DATA_READY:
        import subprocess
        subprocess.run(
            [sys.executable, "scripts/generate_synthetic_data.py"],
            check=True,
        )
    import rasterio
    import geopandas as gpd
    return {
        "plots":       gpd.read_file(DATA_DIR / "input.geojson"),
        "truths":      gpd.read_file(DATA_DIR / "example_truths.geojson"),
        "imagery_ds":  rasterio.open(DATA_DIR / "imagery.tif"),
        "boundary_ds": rasterio.open(DATA_DIR / "boundaries.tif"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1.  data_loader
# ═══════════════════════════════════════════════════════════════════════════

class TestDataLoader:
    def test_load_plots_returns_gdf(self, synthetic_data):
        gdf = synthetic_data["plots"]
        assert len(gdf) > 0
        assert "plot_number" in gdf.columns
        assert gdf.geometry.notnull().all()

    def test_load_example_truths(self, synthetic_data):
        gdf = synthetic_data["truths"]
        assert len(gdf) > 0
        assert "plot_number" in gdf.columns

    def test_reproject_to_imagery_crs(self, synthetic_data):
        from src.data_loader import reproject_to_imagery_crs
        gdf = synthetic_data["plots"]
        ds  = synthetic_data["imagery_ds"]
        reprojected = reproject_to_imagery_crs(gdf.copy(), ds)
        assert reprojected.crs == ds.crs

    def test_ensure_plot_number_fallback(self):
        """If no recognised ID column exists, fall back to integer index."""
        import geopandas as gpd
        from shapely.geometry import box
        from src.data_loader import _ensure_plot_number

        gdf = gpd.GeoDataFrame({"value": [1, 2]}, geometry=[box(0,0,1,1), box(1,0,2,1)])
        result = _ensure_plot_number(gdf)
        assert "plot_number" in result.columns


# ═══════════════════════════════════════════════════════════════════════════
# 2.  drift_estimator
# ═══════════════════════════════════════════════════════════════════════════

class TestDriftEstimator:
    def test_estimate_global_drift_known_offset(self):
        """Median drift should recover the known TRUE_DX / TRUE_DY."""
        import geopandas as gpd
        from shapely.geometry import box
        from shapely.affinity import translate
        from src.drift_estimator import estimate_global_drift

        TRUE_DX, TRUE_DY = 8.0, -5.0
        polys = [box(i * 50, 0, i * 50 + 40, 30) for i in range(10)]
        truths = gpd.GeoDataFrame(
            {"plot_number": [str(i) for i in range(10)]},
            geometry=polys,
            crs="EPSG:32644",
        )
        inputs = gpd.GeoDataFrame(
            {"plot_number": [str(i) for i in range(10)]},
            geometry=[translate(p, TRUE_DX, TRUE_DY) for p in polys],
            crs="EPSG:32644",
        )
        dx, dy = estimate_global_drift(inputs, truths)
        assert abs(dx - (-TRUE_DX)) < 1.0, f"dx={dx} expected ≈{-TRUE_DX}"
        assert abs(dy - (-TRUE_DY)) < 1.0, f"dy={dy} expected ≈{-TRUE_DY}"

    def test_estimate_global_drift_no_truths(self):
        import geopandas as gpd
        from shapely.geometry import box
        from src.drift_estimator import estimate_global_drift

        plots = gpd.GeoDataFrame(
            {"plot_number": ["1"]},
            geometry=[box(0, 0, 10, 10)],
            crs="EPSG:32644",
        )
        dx, dy = estimate_global_drift(plots, None)
        assert dx == 0.0 and dy == 0.0

    def test_estimate_global_drift_no_matching_ids(self):
        import geopandas as gpd
        from shapely.geometry import box
        from src.drift_estimator import estimate_global_drift

        plots  = gpd.GeoDataFrame({"plot_number": ["A"]}, geometry=[box(0,0,1,1)])
        truths = gpd.GeoDataFrame({"plot_number": ["B"]}, geometry=[box(1,0,2,1)])
        dx, dy = estimate_global_drift(plots, truths)
        assert dx == 0.0 and dy == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 3.  patch_extractor
# ═══════════════════════════════════════════════════════════════════════════

class TestPatchExtractor:
    def test_patch_shape_and_types(self, synthetic_data):
        from src.patch_extractor import patch_for_plot
        from src.data_loader import reproject_to_imagery_crs

        plots = reproject_to_imagery_crs(
            synthetic_data["plots"], synthetic_data["imagery_ds"]
        )
        geom = plots.iloc[0].geometry
        result = patch_for_plot(
            geom,
            synthetic_data["imagery_ds"],
            synthetic_data["boundary_ds"],
        )
        assert result is not None, "Expected a valid patch"
        img, bnd, transform = result
        assert img.ndim in (2, 3)
        assert bnd.ndim == 2
        assert img.dtype == np.uint8
        assert bnd.dtype == np.uint8

    def test_patch_outside_raster_returns_none(self, synthetic_data):
        from src.patch_extractor import patch_for_plot
        from shapely.geometry import box

        # Geometry far outside the raster extent.
        far_geom = box(0, 0, 1, 1)   # near lat/lon origin
        result = patch_for_plot(
            far_geom,
            synthetic_data["imagery_ds"],
            synthetic_data["boundary_ds"],
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 4.  edge_detector
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeDetector:
    def test_edge_map_shape_dtype(self):
        from src.edge_detector import build_edge_map

        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        bnd = np.zeros((100, 100), dtype=np.uint8)
        edge = build_edge_map(img, bnd)
        assert edge.shape == (100, 100)
        assert edge.dtype == np.uint8

    def test_edge_map_detects_hard_edge(self):
        """A sharp vertical edge in the image should produce nonzero pixels."""
        from src.edge_detector import build_edge_map

        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 200       # hard left-right edge at column 50
        bnd = np.zeros((100, 100), dtype=np.uint8)
        edge = build_edge_map(img, bnd, boundary_weight=0.0, dilate_px=0)
        assert edge[:, 48:52].max() > 0, "Canny should detect the sharp edge"

    def test_rgb_input(self):
        from src.edge_detector import build_edge_map

        img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        bnd = np.zeros((80, 80), dtype=np.uint8)
        edge = build_edge_map(img, bnd)
        assert edge.shape == (80, 80)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  alignment_scorer
# ═══════════════════════════════════════════════════════════════════════════

class TestAlignmentScorer:
    def _make_fixtures(self):
        """Create a simple scenario: box polygon + matching edge image."""
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import box

        size      = 100
        pixel_m   = 1.0
        ox, oy    = 0.0, 100.0
        transform = from_origin(ox, oy, pixel_m, pixel_m)

        # True geometry at (20,60)–(60,80) in CRS space.
        geom = box(20, 60, 60, 80)

        # Edge map with bright pixels exactly on the box boundary.
        edge = np.zeros((size, size), dtype=np.uint8)
        # top row of box: CRS y=80 → row=20;  CRS x 20..60 → col 20..60
        edge[20, 20:60] = 255
        edge[40, 20:60] = 255
        edge[20:40, 20] = 255
        edge[20:40, 60] = 255

        return geom, edge, transform

    def test_best_alignment_zero_drift(self):
        from src.alignment_scorer import find_best_alignment

        geom, edge, transform = self._make_fixtures()
        result = find_best_alignment(
            geom, edge, transform,
            global_dx=0.0, global_dy=0.0,
            search_radius=5.0, search_step=1.0,
        )
        # At zero offset the boundary exactly matches the edge.
        assert result.score > 0.5, f"Expected high score, got {result.score:.3f}"

    def test_result_fields(self):
        from src.alignment_scorer import find_best_alignment

        geom, edge, transform = self._make_fixtures()
        result = find_best_alignment(
            geom, edge, transform,
            global_dx=0.0, global_dy=0.0,
            search_radius=3.0, search_step=1.0,
        )
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.second_score <= 1.0
        assert result.shifted_geom is not None


# ═══════════════════════════════════════════════════════════════════════════
# 6.  confidence_estimator
# ═══════════════════════════════════════════════════════════════════════════

class TestConfidenceEstimator:
    def _make_alignment(self, score, second_score):
        from src.alignment_scorer import AlignmentResult
        from shapely.geometry import box

        geom = box(0, 0, 10, 10)
        return AlignmentResult(
            dx=0, dy=0,
            score=score,
            second_score=second_score,
            all_scores=np.array([score, second_score]),
            shifted_geom=geom,
        )

    def test_high_score_high_confidence(self):
        from src.confidence_estimator import compute_confidence
        from shapely.geometry import box

        alignment = self._make_alignment(score=0.9, second_score=0.5)
        result = compute_confidence(alignment, box(0, 0, 10, 10))
        assert result.confidence > 0.5

    def test_low_score_low_confidence(self):
        from src.confidence_estimator import compute_confidence
        from shapely.geometry import box

        alignment = self._make_alignment(score=0.1, second_score=0.09)
        result = compute_confidence(alignment, box(0, 0, 10, 10))
        assert result.confidence < 0.5

    def test_confidence_in_range(self):
        from src.confidence_estimator import compute_confidence
        from shapely.geometry import box

        for s in [0.0, 0.3, 0.6, 0.9]:
            alignment = self._make_alignment(score=s, second_score=max(0, s - 0.1))
            result = compute_confidence(alignment, box(0, 0, 10, 10))
            assert 0.0 <= result.confidence <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 7.  decision_maker
# ═══════════════════════════════════════════════════════════════════════════

class TestDecisionMaker:
    def _make_inputs(self, conf_value):
        from src.alignment_scorer import AlignmentResult
        from src.confidence_estimator import ConfidenceResult
        from shapely.geometry import box

        geom = box(0, 0, 10, 10)
        alignment = AlignmentResult(
            dx=1.0, dy=1.0,
            score=conf_value,
            second_score=max(0, conf_value - 0.1),
            all_scores=np.array([conf_value]),
            shifted_geom=geom,
        )
        conf_result = ConfidenceResult(
            confidence=conf_value,
            overlap_sub=conf_value,
            margin_sub=0.5,
            area_sub=1.0,
        )
        return geom, alignment, conf_result

    def test_high_confidence_corrected(self):
        from src.decision_maker import make_decision
        import config

        geom, alignment, conf = self._make_inputs(config.CONFIDENCE_CORRECT + 0.05)
        decision = make_decision("1", geom, alignment, conf, 0.0, 0.0)
        assert decision is not None
        assert decision.status == "corrected"
        assert decision.confidence is not None

    def test_medium_confidence_flagged(self):
        from src.decision_maker import make_decision
        import config

        mid = (config.CONFIDENCE_FLAG + config.CONFIDENCE_CORRECT) / 2
        geom, alignment, conf = self._make_inputs(mid)
        decision = make_decision("2", geom, alignment, conf, 0.0, 0.0)
        assert decision is not None
        assert decision.status == "flagged"
        assert decision.confidence is None

    def test_low_confidence_omitted(self):
        from src.decision_maker import make_decision
        import config

        geom, alignment, conf = self._make_inputs(config.CONFIDENCE_FLAG - 0.05)
        decision = make_decision("3", geom, alignment, conf, 0.0, 0.0)
        assert decision is None


# ═══════════════════════════════════════════════════════════════════════════
# 8.  End-to-end integration test
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_full_pipeline_runs(self, tmp_path):
        """Run main.run() on synthetic data and verify output structure."""
        import subprocess, sys

        # Generate data if needed.
        if not DATA_READY:
            subprocess.run(
                [sys.executable, "scripts/generate_synthetic_data.py"],
                check=True,
            )

        from main import run
        out_dir = tmp_path / "output"
        run(DATA_DIR, out_dir)

        pred_path = out_dir / "predictions.geojson"
        assert pred_path.exists(), "predictions.geojson should be created"

        import geopandas as gpd
        gdf = gpd.read_file(pred_path)
        assert len(gdf) > 0, "Predictions GeoDataFrame should not be empty"

        # Check schema.
        for col in ["plot_number", "status", "confidence", "method_note"]:
            assert col in gdf.columns, f"Missing column: {col}"

        assert set(gdf["status"].unique()).issubset({"corrected", "flagged"})
