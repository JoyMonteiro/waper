# Phase 0 Implementation Plan — Small-Data Sanity Check

> **Audience:** an implementer (Gemini Flash) executing Phase 0 of `superpowers/plans/design/validation_strategy_plan.md`. This document is self-contained: every API call, attribute name, data fact, and target number you need has been verified against the codebase and is written out below. Follow it literally. When in doubt, prefer the explicit recipe here over improvising.

## 0. What you are doing and why

WAPER detects Rossby Wave Packets (RWPs) by finding maxima/minima in the 300 hPa meridional wind, connecting them into packets (a graph of nodes joined by edges), drawing a polygon envelope around each packet, and tracking packets across time by polygon overlap.

Three things must be checked **before** anyone runs a multi-year analysis:
1. **Resolution sensitivity** — does it matter whether we use hourly vs 6-hourly, 0.25° vs 1.0°? This decides what (expensive) dataset the long evaluation actually needs. We expect finer space to give stronger RWPs; quantify it.
2. **Greedy over-connection** → a few hemisphere-spanning packets instead of several distinct ones. Driven mainly by `edge_pruning_threshold` being too low.
3. **Tracking graph is untested**, especially merges/splits.

Your job: run WAPER on a short window of real data at four resolutions, compute summary statistics, compare them to published climatology (numbers below) and to each other, and report whether the algorithm is sane and which resolution the long evaluation should use. This is a **GATE**: if it fails, the bigger layers do not run.

**Deliverables (all under `datasets/experiments/phase0/`):**
- `phase0_stats.py` — reusable extraction + metrics module.
- `run_phase0.py` — runner that produces the CSVs, plots, and report.
- `per_rwp.csv`, `per_track.csv` — raw extracted statistics.
- `plots/` — the diagnostic figures named in the tasks.
- `phase0_report.md` — a short written report with the pass/fail verdict on each gate criterion (also log a session entry in `datasets/experiments/`).

**Environment:** run Python directly — the conda environment is already activated. Do **not** wrap commands in `conda run`. Run scripts with `python datasets/experiments/phase0/run_phase0.py` from the repo root. Run tests (if any) with `pytest` directly.

---

## 1. Verified facts about the code and data (do not re-derive)

### 1.1 The dataset: `datasets/validation.nc`
- Variable: `v` (300 hPa meridional wind), units **m s⁻¹**, range about −85…+92, mean ≈ 0. **It is real m/s** — the published m/s targets apply directly.
- Dims of `v`: `(valid_time, pressure_level, latitude, longitude)`. **You must `.squeeze()` to drop `pressure_level`** (size 1).
- Time coordinate is **`valid_time`** (NOT `time`), **hourly**, 2024-01-01T00 through ~2024-04-01 (≈2182 steps), plus one stray step at 2024-12-31 you should drop.
- Grid: **global**, 0.25° (latitude 90→−90, 721 pts; longitude 0→359.75, 1440 pts). This is large and high-cadence — you must subset (see §2.1).
- Coordinate labels to pass to WAPER: `latitude_label="latitude"`, `longitude_label="longitude"`, `time_label="valid_time"`, `scalar_name="v"`.

### 1.2 The public API (verified in `waper/interface/api.py`)
```python
from waper import Waper
from waper.tracking import tracking_graph as tg

waper = Waper(
    data_array=ds,                  # an xarray.Dataset containing variable "v"
    scalar_name="v",
    latitude_label="latitude",
    longitude_label="longitude",
    time_label="valid_time",
    clip_value=2,                   # keep as-is (pre-filter, not the amplitude threshold)
    extrema_threshold=10,           # keep as-is
    min_latitude=20,                # NH waveguide band
    max_latitude=80,
    node_pruning_threshold=20,      # ST — amplitude threshold, m/s (see §1.4)
    edge_pruning_threshold=0.02,    # GT — (m/s)/km. USE 0.02, NOT the 3e-5 default (see §1.4)
    track_pruning_threshold=0.3,    # leave; you will bypass it (see §1.5)
    max_edge_weight=1,
    penalty_length_scale_km=4000,
    debug=False,
)
waper.identify_rwps()   # mutates internal state; returns nothing
waper.track_rwps()      # builds waper._tracking_graph
```
`Waper` does **not** return results. You read them off internal attributes (next section). The constructor does **not** expose the clustering params (`cluster_max_eps_km`, `cluster_xi`, `min_longitude_separation`) — those keep their dataclass defaults in Phase 0. Do not try to pass them.

### 1.3 Reading per-timestep RWPs (verified)
For each timestep index `t`:
```python
tsd = waper._time_step_data[t]
paths = tsd.identified_rwp_paths        # list; each item is a tuple of node-ids = one RWP
graph = tsd.pruned_graph                # networkx.Graph; node attrs include 'scalar' and 'coords'
for path in paths:
    info = tsd.rwp_info[tuple(path)]     # dict: 'polygon','rwp_id','sample_points',
                                         #       'weighted_longitude','weighted_latitude'
    node_scalars = [abs(graph.nodes[n]['scalar']) for n in path]   # |v| at each extremum, m/s
    lons = [graph.nodes[n]['coords'][0] for n in path]             # degrees east
    lats = [graph.nodes[n]['coords'][1] for n in path]
    peak_amp   = max(node_scalars)       # per-RWP peak amplitude (m/s)
    n_nodes    = len(path)               # number of highs+lows
    n_edges    = len(path) - 1
    centroid_lon = info['weighted_longitude']
    centroid_lat = info['weighted_latitude']
```
If `paths` is empty for a timestep, that timestep has zero RWPs (valid — record 0).

### 1.4 The two threshold knobs (verified in `rwp_graph.py`)
- **ST = `node_pruning_threshold`** — a node (max/min pair) survives only if its amplitude `≥ ST`. Units = field units = **m/s**. Code default 20. Literature: Souders min tracking 14 m/s; Pandey ST sweep 25–50.
- **GT = `edge_pruning_threshold`** — an edge survives only if its weight `≥ GT`, where weight `= (max_scalar − min_scalar) / distance_km × zonal_fraction`. Units = **(m/s)/km**. The `api.py` default `3e-5` is effectively **zero** (prunes almost nothing → greedy giant packets). The tuned `datasets/visualize.py` uses **`0.02`**, in the literature range **0.0–0.08**. **Use 0.02 as your baseline.**

### 1.5 The tracking graph (verified in `waper/tracking/tracking_graph.py`)
- `waper._tracking_graph` is a `networkx.DiGraph`. Nodes are `(time_index, feature_id)` tuples with attribute `coords=(lon, lat)`. Edges connect overlapping packets in consecutive timesteps with attributes `weight` (overlap fraction) and `distance` (km between centroids).
- **Merge = node with in-degree > 1. Split = node with out-degree > 1.** Compute directly:
  ```python
  raw = waper._tracking_graph
  merges = [n for n in raw.nodes if raw.in_degree(n) > 1]
  splits = [n for n in raw.nodes if raw.out_degree(n) > 1]
  ```
- To get tracks as paths:
  ```python
  pruned = tg.prune_tracking_graph(raw, threshold=8000)   # km; see warning below
  paths  = tg.get_track_paths(pruned)   # list of paths; each path is a list of (time, feature) nodes
  ```
  **WARNING — verified bug:** `waper.track_rwps()` internally prunes with `track_pruning_threshold=0.3`, but pruning keeps edges where `distance < threshold` and `distance` is in **km**. `0.3 km` deletes essentially every edge → empty graph. So **do not rely on the internally pruned graph.** Always re-prune the raw graph yourself with a km threshold (use `8000`, as `visualize.py` does). Note this bug in your report.
- `get_track_paths` collapses merges/splits into independent single strands — so for the merge/split task (§2.4) inspect the **raw** graph degrees, not the extracted paths.

### 1.6 Useful helpers (already in the repo)
```python
from waper.identification.utils import haversine_distance, _longitude_separation
# _longitude_separation(lon1, lon2) -> shortest angular gap in degrees, handles 0/360 wraparound
```

---

## 2. Tasks

Do these in order. Each task lists method, the metric extraction, and the acceptance check. Stop and report if a task's data extraction cannot be made to work — do not fabricate numbers.

### Task 2.0 — Setup: load and build the four resolution variants
The central question of Phase 0 is **how sensitive the results are to time and space resolution.** So instead of fixing one subset, build a 2×2 grid of variants from `validation.nc`, all on an **identical domain and time window** so the only thing that changes is resolution.

**Common preprocessing (apply to all variants):**
1. Open `datasets/validation.nc`, select `v`, `.squeeze()` to drop `pressure_level`.
2. Drop the stray December step: keep `valid_time < 2024-03-01`.
3. Restrict to NH: `.sel(latitude=slice(80, 20))` (latitude is descending 90→−90, so high→low).
4. **Pick a short common window for the resolution test** so the expensive hourly-0.25° variant is tractable — start with **~1 week** of winter (e.g. `valid_time` in 2024-01-01…2024-01-08, ~168 hourly steps). Widen later if cheap; keep all four variants on the *same* window.

**The four variants (identical window/domain; only resolution differs):**
| Variant | Time | Space | How |
|---|---|---|---|
| A | hourly | 0.25° | native (after common preprocessing) |
| B | hourly | 1.0° | `coarsen(latitude=4, longitude=4, boundary="trim").mean()` |
| C | 6-hourly | 0.25° | `isel(valid_time=slice(None, None, 6))` |
| D | 6-hourly | 1.0° | both of the above |

Run the **same** `Waper(...)` config (§1.2 thresholds) + `identify_rwps()` + `track_rwps()` on each. Record `dt` per variant (1 h for A/B, 6 h for C/D) — needed to convert track durations to hours.

Structure the code so one function `run_variant(ds_variant) -> (per_rwp_df, per_track_df, tracking_graph)` is reused for all four (and later for the threshold sweeps). Put it in `run_phase0.py`; put metric functions in `phase0_stats.py`.

**Acceptance:** all four variants run end-to-end; print per-variant timestep count, total RWPs, and wall-clock time. If hourly-0.25° (A) is too slow, **shrink the window further or restrict longitude** (e.g. 0–180°E) — but apply the identical cut to all four variants. Report exactly what window/domain you used.

### Task 2.1 — Resolution sensitivity comparison (the core Phase 0 question)
**Why:** this tells us what dataset the *longer* evaluation actually needs. Hourly-0.25° global multi-year is enormously expensive; if 6-hourly/1.0° (variant D) reproduces the same RWP statistics, the full evaluation can use it and run ~100× cheaper. We also expect **genuine physical differences** — finer space resolves sharper extrema, so RWP amplitudes should be higher; quantify that, don't dismiss it as noise.

**Method:** compute the same summary statistics (extraction recipes in Task 2.2) for all four variants and tabulate side by side:

| Metric | What to compare across A/B/C/D |
|---|---|
| Mean & 95th-pct peak amplitude | **expected to rise with spatial resolution** (sharper extrema). Quantify the 0.25° vs 1.0° gap. |
| RWP count per timestep | does finer space create more small-scale packets? |
| Mean nodes per packet, zonal extent | does finer space fragment or enlarge packets? |
| Implied wavenumber distribution | should be resolution-robust if the structure is real (5–8) |
| Mean track duration (hours) & propagation (deg) | **time-resolution sensitive** — report in hours/deg, never in step-counts |
| Merge/split event counts (raw graph) | hourly may over-link (near-total overlap) or flicker; 6-hourly matches literature cadence |

**Separate the two axes explicitly:**
- **Space effect** = A vs B and C vs D (same cadence, different grid). Drives *detection*: amplitude, count, node structure.
- **Time effect** = A vs C and B vs D (same grid, different cadence). Drives *tracking*: duration, propagation, merge/split frequency, overlap linking.

Plot each metric as a 2×2 heatmap or grouped bar (A/B/C/D) into `plots/resolution/`; overlay all four amplitude histograms on one axis.

**Caveat to note in the report:** WAPER runs on **raw** `v` with no spectral truncation, whereas Souders/Pandey/Hunt pre-smooth (T21/T63). Finer grids therefore feed WAPER more small-scale structure. If amplitude/count inflate sharply at 0.25°, flag whether a light smoothing/truncation step should be standardized before the long evaluation.

**Acceptance & decision:** state, per metric, whether it is (i) resolution-insensitive, (ii) space-sensitive only, (iii) time-sensitive only, or (iv) both. **Recommend the cheapest variant that preserves the statistics the long evaluation needs**, and record the expected amplitude bias of going coarse. This recommendation is the main deliverable of Phase 0's first half. Use the recommended variant (widened to ~2 months) as the **reference config** for Tasks 2.2–2.6.

### Task 2.2 — Distributional sanity (Step 0.1)
**Method:** loop all timesteps, extract per-RWP rows (§1.3) into `per_rwp.csv` with columns:
`timestep, rwp_id, peak_amp, n_nodes, n_edges, centroid_lon, centroid_lat, zonal_extent_deg, implied_wavenumber`.
- `zonal_extent_deg` = span of the packet's node longitudes, wraparound-aware. Compute as the **maximum pairwise `_longitude_separation`** among the path's `lons` (robust to dateline).
- `implied_wavenumber` = `180 / mean_adjacent_node_spacing_deg`, where adjacent spacing = `_longitude_separation` between consecutive nodes sorted by longitude. (Half a wavelength ≈ one max-to-min spacing, so wavenumber ≈ 180/spacing.) If `n_nodes < 2`, leave NaN.

Then extract per-track rows (§1.5) into `per_track.csv`: `track_id, t_start, t_end, duration_steps, duration_hours, start_lon, end_lon, propagation_deg`.
- `duration_hours = duration_steps * 6`.
- `propagation_deg` = eastward longitude displacement from `start_lon` to `end_lon` (handle wraparound; eastward positive).

Compute summary statistics and compare to these **target bands** (treat each as order-of-magnitude / shape, not exact — see caveats in §3):

| Statistic | Target band | Source |
|---|---|---|
| Mean peak amplitude | ~20–30 m/s | Souders §3a |
| Amplitude distribution shape | unimodal, peak ~20 m/s, right-skewed | Souders Fig. 1 |
| Fraction peak amp > 30 m/s ("extreme") | ~0.05–0.10 | Souders §3a |
| Implied zonal wavenumber (median) | 5–8 | Chang & Yu 1999 |
| Mean track duration | ~4–8 days; ~70% < 8 days | Souders Table 1 |
| Mean eastward propagation | ~100–140°; ~80% < 180° | Souders Table 1 |
| RWPs per timestep | a handful (≈1–6), not 1 giant or dozens | qualitative |

**Acceptance:** produce a table in `phase0_report.md` with your computed value next to each target band and a ✓/✗. Histograms of `peak_amp`, `duration_hours`, `propagation_deg`, `implied_wavenumber` saved to `plots/`.

### Task 2.3 — Oversized-packet diagnostic (Step 0.2)
**Method:** from `per_rwp.csv`:
- Histogram of `zonal_extent_deg` across all RWPs. **Flag** any RWP with `zonal_extent_deg > 120`.
- Per timestep, compute the fraction of total nodes contained in the single largest packet; histogram it. A value near 1.0 at most timesteps = greedy over-connection winning.
- Histogram of `n_nodes` per packet. Flag packets with `n_nodes ≥ 15` spanning >120° as likely merge artifacts.
- Save 4–6 overlay sanity figures: use `waper.plot_rwp_polygons(t)` and `waper.plot_pruned_graph(t)` (these exist on the object) for a spread of timesteps, into `plots/overlays/`.

**Acceptance:** report (a) % of RWPs exceeding 120° extent, (b) median largest-packet node fraction, (c) a one-line human judgement after eyeballing the overlays: *are packets compact and multiple, or is one blob eating each timestep?* Gate passes if packets are mostly compact (most extents < 120°, largest-packet fraction typically < 0.6) and overlays look reasonable.

### Task 2.4 — Threshold knife-edge sweep (Step 0.3)
**Method:** re-run identification (no need to re-track) over two 1-D sweeps, reusing the cached subset Dataset:
- **GT sweep:** `edge_pruning_threshold ∈ {0.0, 0.01, 0.02, 0.04, 0.06, 0.08}`, holding `node_pruning_threshold=20`.
- **ST sweep:** `node_pruning_threshold ∈ {14, 20, 25, 30, 40, 50}`, holding `edge_pruning_threshold=0.02`.

For each setting compute, averaged over timesteps: mean RWP count per timestep, mean `zonal_extent_deg`, mean `n_nodes` per packet. Plot each metric vs the swept parameter (two figures, three curves each) into `plots/`.

**Acceptance:** identify and report a **plateau** region where the three metrics are roughly stable for each parameter. State the chosen defaults. Gate passes if a plateau exists; **fails** (record loudly) if every GT value is either "one blob" (low GT) or "confetti" (high GT) with no stable middle — that means the connection step is too sensitive.

### Task 2.5 — Split/merge tracking-graph probe (Step 0.4)
**Method:**
1. On the baseline run, compute `merges` and `splits` from the **raw** graph (§1.5). Report counts and their `(time, feature)` locations.
2. Pick **2–3 timestep sequences** around a few of those merge/split nodes. For each, render the packets at `t−1`, `t`, `t+1` (`waper.plot_rwp_polygons(t)`), and confirm by eye whether the graph event corresponds to a real visual merge/split, or whether a packet merely fragmented for one step and was re-stitched.
3. Tabulate for the inspected cases: merges detected / missed, splits detected / missed, spurious events.

**Acceptance:** a small table of inspected events with a ✓/✗ per case, plus a one-paragraph verdict: *does the graph capture real merges/splits and re-stitch threshold flicker?* Also confirm and report the `track_pruning_threshold` km-units bug from §1.5.

### Task 2.6 — Gate decision (Step 0.5)
Write the final verdict in `phase0_report.md`: PASS only if **all** hold —
- 2.2 statistics within the target bands (shape right; modest bias OK),
- 2.3 packets compact and multiple (no chronic giant-packet pathology),
- 2.4 a stable threshold plateau exists,
- 2.5 the tracking graph captures real splits/merges and re-stitches flicker.

Record the chosen default parameters (`node_pruning_threshold`, `edge_pruning_threshold`, plus the subset/resolution/cadence you used) and every caveat encountered. If any criterion fails, state precisely which metric failed and your hypothesis, and recommend the next investigation (likely feeding `superpowers/plans/design/clustering_investigation_plan.md`). Do **not** declare PASS to be agreeable — an honest FAIL here is the whole point of the gate.

---

## 3. Interpreting results — read before judging anything

- **Bias vs shape.** A uniform offset (all amplitudes ~10% high, all durations ~15% long) is *calibration*, not a bug — the distribution *shape* is what matters. A wrong shape (bimodal durations, westward-skewed propagation, one hemispheric blob per timestep) is a real algorithm problem. Never "calibrate away" a wrong shape.
- **Targets are bands, not exact numbers.** Souders used NCEP–NCAR at 2.5°; this is ERA5 coarsened to ~1°, a different period, NH only. Expect offsets. Judge by whether you land in the band and whether shapes/relative behaviors match.
- **Amplitude semantics.** WAPER's node `scalar` is the *instantaneous peak* `|v|` at an extremum; Souders' WPA is a *smoothed envelope* amplitude. WAPER's peak will read somewhat higher than the envelope WPA. That's expected — compare orders of magnitude and shape, not exact values.
- **You restricted to NH and a winter-ish period**, so use the NH columns of every target table; the winter season should be near the high end of activity (Souders Fig. 4).

## 4. Reference numbers (NH, from the papers — for convenience)

- Mean duration 5.8 d (CI 4.9–7.0); 71.8% last < 8 days; max ~47 d (rare). [Souders Table 1]
- Mean eastward propagation 119° (CI 104–139); 82.7% < 180°. [Souders Table 1]
- Mean/median peak amplitude 27.1 / 23.8 m/s; 95th pct 29.7 m/s; P(amp>30) ≈ 0.065. [Souders §3a]
- Dominant zonal wavenumber 5–8. [Chang & Yu 1999]
- Group velocity ~15–25 m/s (eastward), exceeds phase speed ~12 m/s (downstream development). [Chang & Yu 1999; Souders Fig. 7] — *informational; not required in Phase 0 unless you can extract velocities easily.*
- Tracker quality reference: Souders hand-verification gave POD ≈ 93%, FAR ≈ 20%, with ~75% of false alarms during merges/splits.

---

## 5. Checklist (tick in your report)
- [ ] 2.0 four resolution variants (A/B/C/D) built and run; per-variant counts + wall-clock printed
- [ ] 2.1 resolution sensitivity table + plots; space-vs-time effects separated; reference config recommended
- [ ] 2.2 `per_rwp.csv` + `per_track.csv` written; target-band table filled; histograms saved
- [ ] 2.3 oversized-packet metrics + overlay figures + human verdict
- [ ] 2.4 GT and ST sweeps plotted; plateau identified; defaults chosen
- [ ] 2.5 merge/split table + verdict; `track_pruning_threshold` bug confirmed
- [ ] 2.6 PASS/FAIL verdict with per-criterion reasoning and chosen defaults
- [ ] session entry added to `datasets/experiments/`
