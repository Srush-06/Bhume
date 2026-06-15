"""
main.py
-------
Entry point for the cadastral boundary correction pipeline.

Usage
-----
    python main.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR] [--debug]

The script orchestrates the four phases defined in the assignment:

    Phase 1 – Estimate village-wide drift from example_truths.geojson
    Phase 2 – Per-plot edge-based alignment search
    Phase 3 – Confidence estimation
    Phase 4 – Decision (corrected / flagged / omit)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

import config
from src.data_loader import (
    load_plots, load_imagery, load_boundaries,
    load_example_truths, reproject_to_imagery_crs,
)
from src.drift_estimator   import estimate_global_drift
from src.patch_extractor   import patch_for_plot
from src.edge_detector     import build_edge_map
from src.alignment_scorer  import find_best_alignment
from src.confidence_estimator import compute_confidence
from src.decision_maker    import make_decision
from src.output_writer     import write_predictions


# ── logging setup ──────────────────────────────────────────────────────────

def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Correct cadastral plot boundaries using satellite imagery."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help="Directory containing input.geojson, imagery.tif, boundaries.tif",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR,
        help="Directory where predictions.geojson will be written",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return p.parse_args()


# ── pipeline ───────────────────────────────────────────────────────────────

def run(data_dir: Path, output_dir: Path) -> None:
    log = logging.getLogger(__name__)

    # Resolve paths.
    input_geojson  = data_dir / "input.geojson"
    imagery_tif    = data_dir / "imagery.tif"
    boundaries_tif = data_dir / "boundaries.tif"
    example_truths = data_dir / "example_truths.geojson"
    output_path    = output_dir / "predictions.geojson"

    for p in [input_geojson, imagery_tif, boundaries_tif]:
        if not p.exists():
            log.error("Required input not found: %s", p)
            sys.exit(1)

    # ── Load inputs ────────────────────────────────────────────────────────
    log.info("═══ Loading inputs ═══")
    imagery_ds    = load_imagery(imagery_tif)
    boundaries_ds = load_boundaries(boundaries_tif)
    plots         = load_plots(input_geojson)
    truths        = load_example_truths(example_truths)

    # Reproject everything to imagery CRS (likely UTM).
    plots = reproject_to_imagery_crs(plots, imagery_ds)
    if truths is not None:
        truths = reproject_to_imagery_crs(truths, imagery_ds)

    # ── Phase 1: Global drift ──────────────────────────────────────────────
    log.info("═══ Phase 1: Global drift estimation ═══")
    global_dx, global_dy = estimate_global_drift(plots, truths)

    # ── Phases 2–4: Per-plot processing ───────────────────────────────────
    log.info("═══ Phases 2–4: Per-plot alignment → confidence → decision ═══")
    decisions = []
    skipped   = 0
    omitted   = 0

    for _, row in tqdm(plots.iterrows(), total=len(plots), desc="plots"):
        plot_number = str(row["plot_number"])
        geom        = row["geometry"]

        # ── Phase 2a: Extract patches ──────────────────────────────────────
        patch_result = patch_for_plot(geom, imagery_ds, boundaries_ds)
        if patch_result is None:
            log.debug("Plot %s: outside raster extent – skipped.", plot_number)
            skipped += 1
            continue

        img_patch, bnd_patch, win_transform = patch_result

        # ── Phase 2b: Build edge map ───────────────────────────────────────
        edge_map = build_edge_map(img_patch, bnd_patch)

        # ── Phase 2c: Alignment search ─────────────────────────────────────
        alignment = find_best_alignment(
            geom, edge_map, win_transform, global_dx, global_dy
        )

        # ── Phase 3: Confidence ────────────────────────────────────────────
        conf_result = compute_confidence(alignment, geom)

        log.debug(
            "Plot %s  score=%.3f  conf=%.3f  (ov=%.2f mg=%.2f ar=%.2f)",
            plot_number,
            alignment.score,
            conf_result.confidence,
            conf_result.overlap_sub,
            conf_result.margin_sub,
            conf_result.area_sub,
        )

        # ── Phase 4: Decision ──────────────────────────────────────────────
        decision = make_decision(
            plot_number   = plot_number,
            original_geom = geom,
            alignment     = alignment,
            conf_result   = conf_result,
            global_dx     = global_dx,
            global_dy     = global_dy,
        )

        if decision is None:
            omitted += 1
        else:
            decisions.append(decision)

    # ── Write output ───────────────────────────────────────────────────────
    log.info("═══ Writing output ═══")
    write_predictions(decisions, output_path, imagery_ds.crs)

    corrected = sum(1 for d in decisions if d.status == "corrected")
    flagged   = sum(1 for d in decisions if d.status == "flagged")
    log.info(
        "Done.  corrected=%d  flagged=%d  omitted=%d  skipped=%d",
        corrected, flagged, omitted, skipped,
    )

    imagery_ds.close()
    boundaries_ds.close()


# ── entry ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    _setup_logging(args.debug)
    run(args.data_dir, args.output_dir)
