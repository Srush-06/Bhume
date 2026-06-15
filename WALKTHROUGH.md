# 5-Minute Interview / Video Walkthrough Script

**Cadastral Boundary Correction Pipeline**

---

> *This script is written as spoken text.  
> Suggested timings are in square brackets [ ].  
> Terminal commands to run live are prefixed with `$`.*

---

## [0:00 – 0:30]  Introduction

> "Hi — I'm going to walk you through my solution to the cadastral boundary correction problem.
> The goal is simple: official plot boundaries recorded in old cadastral maps are often misregistered
> with what you can actually see in satellite imagery. My pipeline detects that mis-registration
> and corrects it using purely classical computer vision — no deep learning, no magic."

---

## [0:30 – 1:00]  Project Layout

> "Let me start with the structure."

```
$ ls -1
config.py
main.py
README.md
requirements.txt
src/
scripts/
tests/
data/
output/
```

> "Everything lives in flat, purpose-named modules. `config.py` holds every single magic number
> in one place — thresholds, search radii, blend weights — so tuning never touches the logic.
> `main.py` is a thin orchestrator. The real work is in `src/`."

---

## [1:00 – 2:00]  Phase 1 — Global Drift

> "The first insight is that most cadastral datasets drift as a *rigid body* —
> every plot is off by roughly the same vector because the whole sheet was geo-referenced together.
> So I use a handful of verified example plots to measure that global offset."

```
$ cat src/drift_estimator.py
```

> "I match input plots to truth plots by `plot_number`, compute the centroid-to-centroid
> displacement for each pair, and take the **median**. Median, not mean — because a single
> re-surveyed outlier plot would corrupt a mean. With five example truths I can already
> get a decent global shift estimate."

> "If no `example_truths.geojson` is present, the drift defaults to zero and the search
> grid is just centred on the original position. The pipeline degrades gracefully."

---

## [2:00 – 3:15]  Phase 2 — Per-Plot Alignment

> "With the global drift in hand, I run a tight grid search for each plot."

```
$ cat src/alignment_scorer.py
```

> "Step one: extract a padded pixel patch from both the satellite image and the boundary hint
> raster using rasterio windowed reads — efficient, no need to load the full image into RAM."

> "Step two: build an edge map.  I use **OpenCV Canny** on the grayscale image.
> Canny is ideal here: it produces thin, well-localised one-pixel-wide edges thanks to
> non-maximum suppression and hysteresis linking. I then blend in the boundary hint channel
> with a configurable weight — useful when the hint raster is trustworthy."

> "Step three: for every candidate `(dx, dy)` on the search grid — centred on the global drift,
> spanning ±12 metres in 2-metre steps — I shift the polygon, rasterise its *outline*
> (not its filled interior!) into the patch frame, and score the fraction of boundary pixels
> that land on an edge-map pixel."

> "Why outline rasterisation?  Because a filled polygon always covers a big area and would score
> high regardless of alignment. The outline is discriminative — it only scores high when the
> polygon edge genuinely follows a field edge in the image."

---

## [3:15 – 4:00]  Phase 3 — Confidence

> "Alignment score alone is not enough. A high score could arise just because the image
> is full of edges everywhere. So I compute three sub-scores:"

```
$ cat src/confidence_estimator.py
```

> "**Overlap** — the raw alignment fraction.  
> **Margin** — the gap between the best and second-best translation. A large margin means the
> search landscape has a clear winner; a small margin means we're guessing.  
> **Area** — penalises translations that partially clip the polygon off the raster edge."

> "These are blended 50 / 30 / 20. The weights are set by reasoning, but the right long-term
> fix is Platt scaling on a labelled validation set — just a logistic regression on top of
> these three features."

---

## [4:00 – 4:30]  Phase 4 — Decision & Output

> "Decision is a simple two-threshold rule:
> ≥ 0.55 → corrected with the shifted geometry.  
> 0.30–0.55 → flagged with the original geometry and a warning.  
> Below 0.30 → omitted entirely."

```
$ python main.py --debug
```

> "Let me run it on the synthetic test bundle. You'll see each phase logged. At the end,
> `predictions.geojson` appears in the output directory with the corrected geometries."

```
$ python -c "import geopandas as gpd; gdf=gpd.read_file('output/predictions.geojson'); print(gdf[['plot_number','status','confidence']].head(10))"
```

---

## [4:30 – 5:00]  What I'd Do Next

> "To push this from Silver to Gold:

> 1. **Sub-pixel refinement** — replace the coarse 2 m grid with a gradient-free optimiser
>    (Nelder-Mead) seeded at the grid best, for centimetre-level alignment accuracy.

> 2. **Calibrated probabilities** — fit isotonic regression on the three sub-scores
>    using a small labelled set, so that confidence 0.7 really means 70% of those plots
>    are correctly aligned.

> 3. **Richer edge signal** — compute the structure tensor or oriented gradient histograms
>    to match *direction* as well as location of field edges.

> 4. **Multi-hypothesis** — instead of a single translation, also test small rotations
>    for plots that may have been re-digitised at a slightly wrong angle.

> Thank you — happy to dive into any part of the code."

---

*End of walkthrough — total ~5 minutes*
