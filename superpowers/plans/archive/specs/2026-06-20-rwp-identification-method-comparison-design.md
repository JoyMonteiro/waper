# Design: Three-Method RWP Identification Agreement Study

**Date:** 2026-06-20
**Status:** Approved design — pending implementation plan
**Topic:** Quantify and visually diagnose where three RWP-identification methods agree and
disagree on the April 2011 forecast-bust event.

---

## 1. Motivation

WAPER's contribution (Pandey et al. 2020) was to identify RWPs from the topology of the
meridional-wind field — retaining phase, avoiding Fourier transforms, working in all
life-cycle stages — explicitly to move *beyond* envelope methods (Zimin et al. 2003/2006)
and feature trackers (Souders et al. 2014). A recent proposal to prune the WAPER graph
using the Zimin envelope (`envelope_segmentation_proposal.md`) raised a concern: does that
regress to the very approach the method was built to avoid?

Underneath that concern is a real, unanswered *identification* question: **how different are
the RWP objects produced by (a) the full topological method, (b) a simpler amplitude-only
view of the same topology, and (c) the spectral envelope?** This study answers it
empirically on the canonical April 2011 forecast-bust case, so that any future decision
about augmenting the method rests on measured agreement rather than intuition.

The three methods span a meaningful axis:

| Method | What an RWP is | Knob | Role |
|---|---|---|---|
| **Edge pruning** (full WAPER) | Ranked paths through the association graph after edge-weight pruning — connected wave trains | GT (`edge_pruning_threshold`) | swept |
| **Node scalar amplitude** | v-max / v-min *cluster* footprints, **no** edge connection — carrier-level anomaly regions | ST (`node_pruning_threshold`) | swept |
| **Zimin envelope** | Hilbert-envelope field thresholded in 2D — smooth packet envelope | 14 m/s | **fixed reference** |

Physical contrast expected: Zimin = smooth packet envelopes; node-amplitude = speckled
carrier-level blobs; edge-pruning = connected wave trains. The study measures how close the
two WAPER variants get to the envelope, and characterizes where they structurally cannot.

## 2. Goal & scope

- **Primary purpose:** agreement/disagreement analysis — quantify *where and why* the three
  methods diverge, not summarize their populations.
- **Deliverable framing:** for each of the two swept methods, **identify the threshold that
  best agrees with the fixed Zimin reference**, then characterize the *residual* (structural)
  disagreement that remains at that best-agreement point.
- **Out of scope:** tracking (this is purely about per-timestep identification); changing any
  `waper/` core code; multi-event climatology.

## 3. Dataset & domain

- **Data:** `datasets/forecast_bust_hourly.nc` — ERA5 300 hPa meridional wind `v`, NH only
  (0–90°N), 0.25°, **hourly, all of April 2011 (720 timesteps)**.
- **Preprocessing:** coarsen 0.25° → 1° (`coarsen(latitude=4, longitude=4,
  boundary="trim").mean()`); convert longitude −180…180 → 0…360
  (`assign_coords(longitude=lambda d: d.longitude % 360).sortby("longitude")`); squeeze the
  singleton `pressure_level`; rename `valid_time` → `time`. (Matches the proven preprocessing
  in `scripts/feature_tracks_gif.py`.)
- **Domain restriction:** **20–80°N** applied identically to all three masks, so the tropics
  and pole never count toward agreement or disagreement. NH only — consistent with all tuning
  to date.
- **Compute budget:** ~1° NH ≈ ~1 s/step → ~12 min per full 720-step WAPER run. A sweep of
  ~6 GT values + ~6 ST values ≈ 12 runs ≈ ~2.5 h. No timestep subsampling needed. The Zimin
  reference masks are computed once and cached.

## 4. Common representation

All three methods reduce to a **per-timestep boolean mask on the shared 512×512 north-polar
stereographic grid** produced by `waper.tracking.rwp_polygon.rasterize_all_rwps`
(`WAPER_IMAGE_SIZE`, transform from `_get_raster_transform("north")`). Using one grid for all
three makes IoU pure set algebra.

- A precomputed **band mask** (cells whose stereographic preimage lies in 20–80°N) is
  intersected with every method mask before any metric is computed.
- **Caveat (documented, not corrected):** polar-stereographic area is distorted across
  20–80°N (low-latitude cells project larger). Because IoU is computed identically for all
  methods on the same grid, the distortion cancels in *comparisons*; only absolute
  area-fraction numbers are biased toward low latitudes. Reported area fractions carry this
  caveat.

## 5. The three mask builders

A new module `scripts/method_comparison/masks.py` exposes one builder per method, each
returning a `bool` array of shape `(512, 512)` for a given timestep, already band-restricted.

### 5.1 Zimin envelope (reference, fixed)
- `compute_rwp_envelope(v_field, wavenumber_range=(3, 11))` — FFT along longitude, zero
  wavenumbers outside 3–11, inverse FFT, Hilbert transform → envelope `E` (lat–lon, ~30 lines
  numpy/scipy, matching Souders' spec; identical to Layer 4 of
  `superpowers/plans/design/validation_strategy_plan.md`).
- Threshold `E ≥ 14 m/s` → connected lat–lon regions → polygonize each region (in
  stereographic metres via `transform_to_stereographic`) → `rasterize_all_rwps` → mask.
- Computed once for all 720 steps and cached to disk (`.npy`), since it never changes across
  the sweep.

### 5.2 Edge pruning (full WAPER) — sweep GT
- Run `Waper(...).identify_rwps()` with `edge_pruning_threshold = GT`, `node_pruning_threshold`
  held at the reconciled default (≈20).
- For each ranked path, `get_polygon_for_rwp_path(...)` → list of `(polygon, id)` →
  `rasterize_all_rwps` → mask.
- **GT sweep:** `{0.0, 0.01, 0.02, 0.04, 0.06, 0.08}` (m/s)/km — Pandey's range; **do not use
  `3e-5` or the legacy `0.3`** (per the validation plan's reconciliation).

### 5.3 Node scalar amplitude — sweep ST
- The v-max / v-min *cluster* footprints **without** edge-connection: take the clustered
  extrema regions WAPER already computes (the connected regions of `|v| ≥ ST`), build a
  per-cluster footprint with `_footprint_from_region` (convex hull in stereographic metres),
  and rasterize the same way.
- **ST sweep:** `{10, 15, 20, 25, 30, 35}` m/s — spans the code default (20) and Pandey's
  25–50 lower range.
- This is the "graph-intrinsic envelope sampled at the carrier extrema" view: amplitude only,
  no wave-train connection.

## 6. Metrics

`scripts/method_comparison/metrics.py`, computed per timestep within the band, then aggregated
over the 720 steps:

- **IoU** between method mask and the Zimin mask: `|A ∩ Z| / |A ∪ Z|`. Mean over time as a
  function of the swept threshold → **argmax = best-agreement threshold** (the "identify
  thresholds" deliverable).
- **Disagreement decomposition:** mean method-only area (`|A \ Z|`) vs Zimin-only area
  (`|Z \ A|`), as fractions of the band — reveals whether a method over- or under-detects
  relative to the envelope.
- **Detection agreement:** fraction of timesteps where both flag ≥1 RWP cell.
- (Secondary) **edge-pruning vs node-amplitude** IoU at their respective best thresholds — how
  much the connection step changes the footprint relative to amplitude alone.

## 7. Outputs

### 7.1 Driver → CSV
`scripts/method_comparison/run_sweep.py`: loads + preprocesses the dataset, caches Zimin
masks, runs each swept method across its threshold grid, writes
`results/method_comparison_sweep.csv` with columns:
`method, threshold, mean_iou, detection_agreement, mean_method_only_frac, mean_zimin_only_frac, n_timesteps`.

### 7.2 Plots (the "why") — `scripts/method_comparison/method_comparison.ipynb`
1. **IoU-vs-threshold curves** for both swept methods, with the Zimin baseline marked and the
   argmax threshold annotated.
2. **Climatological disagreement map** at each best threshold: per-cell frequency over the 720
   steps of {agree, method-only, Zimin-only}, plotted on the NH map — shows *where* on the
   waveguide the methods systematically diverge.
3. **Case-study overlay frames:** a handful of timesteps (including the forecast-bust peak)
   showing the three masks together over the `v` field — the qualitative read on merge/split
   and carrier-vs-envelope behaviour.

## 8. Module structure

```
scripts/method_comparison/
├── masks.py        # zimin_mask(), edge_pruning_mask(), node_amplitude_mask() → (512,512) bool
├── metrics.py      # iou(), disagreement_decomposition(), detection_agreement()
├── run_sweep.py    # driver: preprocess → cache Zimin → sweep → CSV
└── method_comparison.ipynb   # plots + case studies
results/
└── method_comparison_sweep.csv
```

- **Implements** `compute_rwp_envelope` fresh in `masks.py` per the Layer 4 spec in
  `superpowers/plans/design/validation_strategy_plan.md` (it does not yet exist in the codebase).
- Reuses, without modification: `Waper.identify_rwps`, `get_polygon_for_rwp_path`,
  `rasterize_all_rwps`, `_footprint_from_region`, `transform_to_stereographic`,
  `_get_raster_transform`.
- No changes to `waper/` core. Pure analysis layer, mirroring the existing `scripts/`
  diagnostics conventions.

## 9. Testing

- **Unit (`tests/test_method_comparison.py`):**
  - `iou` and `disagreement_decomposition` on hand-built masks (disjoint → IoU 0; identical →
    IoU 1; one-contained-in-other → known value).
  - Each mask builder returns a `(512, 512)` bool array, all-False outside the 20–80°N band,
    on a tiny synthetic single-timestep field.
  - `compute_rwp_envelope` recovers the analytic envelope of a known modulated sinusoid
    (amplitude within tolerance).
- **Integration:** a 3-timestep smoke run of `run_sweep` over a single threshold per method,
  asserting a well-formed CSV row.
- Run with `pytest tests/test_method_comparison.py -q` (env pre-activated; do not wrap in
  `conda run`).

## 10. Open items deferred (YAGNI for this study)

- SH and multi-event climatology — only if the April 2011 result motivates generalization.
- Object-level (merge/split) matching — the user chose mask-level IoU + plots; object matching
  is a possible follow-up if the disagreement maps show systematic individuation differences.
- Any graph-pruning *change* to WAPER — this study is diagnostic input to that decision, not
  the change itself.
