# Antigravity Transcript

## Initial Prompt

Build a Python solution that:
- Reads input.geojson
- Reads imagery.tif
- Reads boundaries.tif
- Produces predictions.geojson

Requirements:
- Classical computer vision
- Confidence calibration
- Corrected / Flagged decisions

## Generated Architecture

Files:
- data_loader.py
- drift_estimator.py
- patch_extractor.py
- edge_detector.py
- alignment_scorer.py
- confidence_estimator.py
- decision_maker.py

## Bug Fixes

Issue:
- GeoJSON written with UTM coordinates

Fix:
- Reproject output to EPSG:4326 before saving
