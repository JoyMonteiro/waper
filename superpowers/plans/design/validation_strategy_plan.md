# WAPER Validation Strategy

> **Context:** This plan should be picked up after the current refactoring work (Phases 4+) is complete and the pipeline runs end-to-end on real data. It is gated: **Phase 0 is a small-data sanity check on `validation.nc` that must pass before any of the larger, more expensive layers are run.** Only once Phase 0 looks sane do we roll out the full multi-year climatological validation (Layer 2), which needs real thought about performance and parallelization.

## Background

There is no universally agreed-upon ground truth for RWPs. The definition is inherently ambiguous — spatial extent, wavenumber, and amplitude all vary during the lifecycle. Manual annotation is subjective, envelope methods lose phase information, and no labeled dataset exists.

Instead of ground truth, we validate against the **published climatology**. Souders et al. (2014), Chang & Yu (1999), and Lee & Held (1993) provide quantitative, falsifiable, citable distributions of RWP amplitude, duration, propagation, group velocity, seasonal cycle, and spatial frequency. Our pipeline does not have to reproduce these numbers exactly — datasets and resolutions differ (see "Reanalysis caveat" below) — but it must reproduce the **shapes, tolerance bands, and relative behaviors**.

### Two failure modes (keep these separate throughout)

Every check below should be read with this distinction in mind, because they demand opposite responses:

- **Systematic bias** — every amplitude is ~10% high, every packet ~15% too long. The *shape* of the distribution is right. This is a calibration issue: rescale or re-tune a threshold, not a bug.
- **Wrong shape** — bimodal duration distribution, propagation skewed westward, a single hemispheric-scale packet dominating most timesteps. This is a genuine algorithm problem and must be investigated, never "calibrated away."

Phase 0 exists primarily to catch wrong-shape failures cheaply, before we spend compute on a full climatology.

### Reanalysis caveat (applies to every quantitative target)

Souders used NCEP–NCAR (2.5°); Pandey used ERA-Interim; we use ERA5. Resolution and dataset differences shift absolute counts, and Souders flag a slow-wind bias in NCEP SH data. **Therefore every published number below is a distribution-shape / tolerance-band target, not an exact match.** Where a single number appears, treat it as the center of a band (±20–30% unless stated otherwise).

---

## Phase 0: Small-data sanity check (`validation.nc`)  — **GATE**

### Goal

Run the full pipeline on `validation.nc` (2024 global ERA5 300 hPa meridional wind) and confirm that the resulting statistics land in the published ballpark **and** that the known structural pathologies of the algorithm are under control. Phase 0 also runs a **time/space resolution-sensitivity study** (hourly vs 6-hourly × 0.25° vs 1.0°) to decide what dataset the longer evaluation actually needs — see `superpowers/plans/implementation/phase0_implementation_plan.md`. **Do not proceed to Layers 1–4 until this gate passes.**

Why this comes first: WAPER builds packets by connecting individual highs and lows (nodes joined by edges). Two failure modes are baked into that design and must be ruled out on small data before they pollute a multi-year run:

1. **Greedy over-connection → oversized packets.** The connection step tends to greedily join everything, producing a few hemisphere-spanning packets instead of several distinct ones.
2. **Threshold knife-edge.** Raising the node amplitude threshold (`node_pruning_threshold` / ST) to break spurious connections risks dropping a *genuine* node when one high/low in a real packet momentarily weakens below threshold — fragmenting a real RWP into pieces. The polygon-envelope + overlap tracking step partly compensates (a packet that fragments for a timestep can be re-stitched by footprint overlap), but **genuine splits and merges then become hard to distinguish from threshold artifacts.** The tracking graph records these events but has **not been tested**.

Phase 0 probes all three.

### Step 0.1: Distributional sanity (the ballpark check)

Run the pipeline on `validation.nc` with default parameters and compute the single-snapshot and (if tracking is on) per-track statistics. Compare against the published bands:

| Statistic | Published target (band) | Source |
|---|---|---|
| Mean peak amplitude | ~27 m/s (median ~24) | Souders §3a |
| Amplitude distribution | unimodal, peak ~20 m/s, skewed to high values | Souders Fig. 1 |
| Detection threshold (min WPA) | 14 m/s | Souders §2 |
| "Extreme" fraction P(WPA > 30 m/s) | ~0.05–0.07 (NH) | Souders §3a |
| Dominant zonal wavenumber | 5–8 | Chang & Yu 1999 |
| Mean duration (if tracking) | ~6 days, ~70% < 8 days | Souders Table 1 |
| Mean eastward propagation (if tracking) | ~120° lon (NH), 83% < 180° | Souders Table 1 |
| Count per timestep | a handful of distinct packets, not 1 giant or 50 fragments | qualitative |

A couple of months is too short for the seasonal-cycle or spatial-frequency checks — those wait for Layer 2. Phase 0 is about order-of-magnitude correctness and shape, not precision.

> **Two unit caveats before comparing to Souders:**
> 1. **Field units.** WAPER's `node_pruning_threshold` is in the *units of the input field*. Confirm `validation.nc` `v` is in m/s (not standardized anomalies, not absolute value — note some fixtures are named `*_abs_*`). If it isn't m/s, the published m/s targets don't apply directly.
> 2. **Amplitude semantics.** WAPER's node `scalar` is the *instantaneous peak* `|v|` at an extremum; Souders' WPA is a *smoothed Hilbert-envelope* amplitude. They are correlated and the same order of magnitude, but WAPER's peak will run somewhat higher than the envelope WPA. Treat the amplitude rows as order-of-magnitude / shape checks, not exact matches.

### Step 0.2: Oversized-packet diagnostic (greedy connection)

This is the primary structural risk. Compute and inspect:

- **Distribution of per-packet zonal extent (degrees longitude) at each snapshot.** A wavenumber 5–8 field should yield instantaneous packets spanning a fraction of the hemisphere, not 180°+. Flag any timestep where one packet spans more than ~120–150° of longitude.
- **Fraction of timesteps dominated by a single packet** — i.e. one packet contains the large majority of nodes. If this is common, greedy connection is winning.
- **Distribution of nodes (highs/lows) per packet.** Compare implied node spacing to wavenumber 5–8. A packet with 15+ alternating nodes spanning the hemisphere is almost certainly a merge artifact.
- **Sanity overlay plots** (`datasets/visualize.py` style): plot the v-field with detected packet polygons for a handful of timesteps and eyeball whether the partitioning matches what a human would draw.

**Expected outcome:** packets are spatially compact and multiple distinct packets coexist per timestep. If instead a few giant packets dominate, the connection logic — not the thresholds — needs work; record this as a blocker and feed it back to the clustering/connection investigation (`clustering_investigation_plan.md`) before continuing.

### Step 0.3: Threshold knife-edge sweep

Two different knobs drive the two failure modes — sweep both on `validation.nc` (cheap):

- **`edge_pruning_threshold` (GT)** controls greedy over-connection. The api default `3e-5` (m/s)/km is effectively *zero* and keeps almost every edge → giant merged packets. Sweep **0.0–0.08 (m/s)/km** (the tuned `visualize.py` value `0.02` is a good starting point). Watch the Step 0.2 oversized-packet metrics fall as GT rises.
- **`node_pruning_threshold` (ST)** is the amplitude knife-edge. Too high and a real packet fragments when one high/low momentarily drops below threshold: total packet count spikes, mean nodes-per-packet and mean zonal extent collapse, mean duration drops. Sweep around the default `20` (m/s), in the Pandey 25–50 neighborhood.

For each, plot mean zonal extent, packet count, and mean nodes-per-packet vs the threshold. **We are looking for a plateau** where the statistics are stable — that is the defensible default. If no plateau exists (every setting is either "one blob" or "confetti"), the connection step is too sensitive and must be fixed before a climatology is meaningful. This is a focused 2-parameter preview of Layer 3; keep the script reusable.

### Step 0.4: Split/merge tracking-graph probe

The tracking graph is untested, and Souders report that **~75% of their tracker's false alarms occurred during merging and splitting** (overall POD ≈ 93%, FAR ≈ 20%). So split/merge handling is exactly where to look.

Graph mechanics (verified in `waper/tracking/tracking_graph.py`): the tracking graph is a `networkx.DiGraph` whose nodes are `(time_index, feature_id)` and whose edges connect overlapping footprints in consecutive timesteps. **Merges and splits are already encoded as node degree:** a node with **in-degree > 1** is a merge (two prior packets → one), a node with **out-degree > 1** is a split. Note that `get_track_paths()` greedily collapses the graph into independent single-strand paths and *throws this structure away* — so inspect the **raw `waper._tracking_graph`**, not the extracted track paths.

- Pick 2–3 timestep sequences in `validation.nc` that contain an obvious visual merge or split (or reuse a documented merge event if `validation.nc` covers one).
- Enumerate the in-degree>1 / out-degree>1 nodes in the raw graph and check by eye against the v-field overlays: does each real merge/split show up as the right degree event, and is a momentary threshold-induced fragmentation (Step 0.3) *re-stitched* by footprint overlap rather than logged as a spurious split?
- Tabulate, for the inspected cases: merges detected / missed, splits detected / missed, and any spurious split/merge events. This is a tiny manual POD/FAR specifically for graph events — it tells us whether the graph is trustworthy enough to compute duration/propagation statistics from in Layer 2.

> **Known wrinkle to verify:** `track_rwps()` prunes with `track_pruning_threshold` (default `0.3`), but `prune_tracking_graph` keeps edges where `distance < threshold` and `distance` is a haversine value **in km**. A 0.3 km cutoff removes essentially every edge, leaving an empty graph; `visualize.py` sidesteps this by calling `prune_tracking_graph(..., threshold=8000)` directly. Confirm this behavior and use a sensible km threshold (e.g. 8000) for Phase 0; flag the default as a likely bug.

**Expected outcome:** the graph distinguishes real splits/merges from threshold flicker. If it does not, duration and propagation statistics derived from tracks (Layer 2, Table 1 comparison) will be biased, and that must be flagged loudly.

### Step 0.5: Gate decision

Proceed to Layers 1–4 only if:
- Step 0.1 statistics are within the published bands (shape correct; modest bias OK).
- Step 0.2 shows compact, multiple packets (no chronic giant-packet pathology).
- Step 0.3 shows a stable threshold plateau.
- Step 0.4 shows the tracking graph captures real splits/merges and re-stitches threshold flicker.

Document the chosen default parameters and any caveats here before moving on. A failure at this gate is cheaper to fix now than after a multi-year run.

---

## Layer 1: Case-Study Regression Tests

### Goal

A pytest suite where each test runs the full pipeline on a real synoptic event and asserts **literature-anchored, physically motivated invariants**. These catch regressions: if a code change makes the algorithm miss a documented RWP or produce a physically impossible one, the test fails. Every invariant cites the published quantity it is derived from — no invented bounds.

### Step 1.1: Select events

5–8 events covering the range of RWP behaviors:

| Event | Dates | Behavior | Source |
|-------|-------|----------|--------|
| Japan cyclogenesis RWP | 21–23 Jan 2007 | Genesis forced by deepening cyclone, downstream amplification over Pacific | Pandey et al. 2020 Fig. 5; Souders et al. 2014b Figs. 6–7 |
| Cutoff low RWP genesis | 6–10 Jan 2007 | Weak genesis at 120°E, **merges** with existing RWP, grows Pacific→Europe | Pandey et al. 2020 Fig. 6 |
| April 2011 forecast bust | 12–17 Apr 2011 | Non-wavelike RWP over Atlantic, **merge** of Pacific and Russian RWPs | Pandey et al. 2020 Fig. 7; Ghinassi et al. 2018 |
| South Asia extreme wet-bulb | May–Jun composite | Weak-amplitude RWP over central Asia | Pandey et al. 2020 Fig. 8–9; Monteiro & Caballero 2019 |
| Strong winter RWP | DJF, clear case | Classic wavelike RWP, 3+ nodes, peak WPA in extreme band (>30 m/s) | From ERA5 |
| Summer weak RWP | JJA | Lower-amplitude RWP that should still be detected | From ERA5 |
| Dateline-crossing RWP | any | Tests 180° wraparound handling | From ERA5 |
| Quiet period | 2–3 days | Few or no RWPs | From ERA5 |

The first four are documented and include the two **merge** cases — reuse them for the split/merge invariants and for the Layer 4 POD/FAR sample. The last four are identified during real-data testing.

### Step 1.2: Prepare test fixtures

For each event:
1. Download 300 hPa meridional wind from ERA5 (or ERA-Interim for paper-comparable results) for the dates/domain.
2. Store as compressed NetCDF in `tests/fixtures/events/` (e.g. `jan2007_genesis.nc`). Crop regional events to keep files small.
3. Add a YAML metadata file recording source, date range, domain bounds, description, literature reference, and the invariants (Step 1.3).

### Step 1.3: Define invariants (literature-anchored)

Invariants must be **robust to algorithm changes**, **falsifiable**, and **cite the published quantity** they encode. Tie each to a Souders/Chang number rather than a guessed bound.

**Existence invariants:**
```python
# At least one RWP spanning the Pacific on 23 Jan 2007 (Pandey Fig. 5)
assert any(rwp.spans_pacific() for rwp in results["2007-01-23T06"])

# No RWP activity in the tropics — RWPs live on the midlatitude waveguide (Chang & Yu 1999)
for ts in results:
    for rwp in results[ts]:
        assert rwp.mean_lat > 15, f"Spurious tropical RWP at {ts}"
```

**Structural invariants (anchored to wavenumber 5–8 and the oversized-packet diagnostic):**
```python
# Implied zonal wavenumber from node spacing must be in the baroclinic band (Chang & Yu 1999)
for rwp in all_rwps:
    assert 4 <= rwp.implied_wavenumber <= 10

# Instantaneous footprint must not span the hemisphere — longer is a greedy-merge artifact
# (wavenumber 5-8 => compact packets; see Phase 0 Step 0.2)
for rwp in all_rwps:
    assert rwp.zonal_extent_deg < 150

# Strong winter case: peak amplitude in the "extreme" band, not just "many edges"
assert max(r.peak_wpa for r in strong_winter_rwps) >= 30  # Souders "extreme" threshold
```

**Physical-signature invariants (new — the defining RWP physics):**
```python
# Downstream development: group velocity exceeds phase speed (Chang & Yu 1999: c_g 25-30, c_p ~12 m/s)
for track in tracks:
    if track.duration_hours >= 24:
        assert track.group_velocity_ms > track.phase_speed_ms

# Group velocity tracks the jet: 15-25 m/s in jets, up to 25-40 just upwind (Souders Fig. 7)
for track in tracks:
    if track.duration_hours >= 24:
        assert 5 < track.mean_group_velocity_ms < 45
        assert track.mean_group_velocity_ms > 0  # eastward; westward is unphysical for RWPs
```

**Temporal invariants (once tracking works; bounds from Souders Table 1):**
```python
# The Jan 2007 genesis RWP should be trackable for >= 2 days (Souders min "significant" duration)
assert any(t.duration_hours >= 48 for t in pacific_tracks)

# Upper-tail guard: a track lasting > 25 days is almost certainly a tracking error
# (Souders: only 13 such in 32 yr in the NH, ~10% of those false). Flag, don't pass silently.
for track in tracks:
    assert track.duration_days < 25, "Suspiciously long track — likely a merge/tracking artifact"
```

**Split/merge invariants (use the two documented merge events):**
```python
# Jan 2007 cutoff-low RWP merges with the existing RWP (Pandey Fig. 6)
assert tracking_graph.has_merge_event(region=east_asia, window=("2007-01-06", "2007-01-10"))

# April 2011: Pacific and Russian RWPs merge over the Atlantic (Pandey Fig. 7)
assert tracking_graph.has_merge_event(region=atlantic, window=("2011-04-12", "2011-04-17"))
```

**Non-detection invariants:**
```python
for ts in quiet_period:
    assert len(results[ts]) <= 2, "Too many RWPs during known quiet period"
```

### Step 1.4: Test structure

```
tests/
  fixtures/events/
    jan2007_genesis.nc / .yaml
    apr2011_bust.nc / .yaml
    ...
  test_case_studies.py     # parametrized over events
  conftest.py              # pipeline runner fixture
```

```python
# conftest.py
@pytest.fixture(scope="module")
def pipeline_results(request):
    event = request.param
    data = xr.open_dataset(f"tests/fixtures/events/{event}.nc")
    waper = Waper(WaperConfig())          # default params from Phase 0
    return waper.identify_rwps(data)

# test_case_studies.py
@pytest.mark.parametrize("event,invariants", load_event_invariants())
def test_event_invariants(pipeline_results, event, invariants):
    for inv in invariants:
        inv.check(pipeline_results)
```

> Run pytest directly (the conda env is pre-activated) — do not wrap in `conda run`.

### Step 1.5: Run and calibrate

The first real run will fail some invariants. Apply the bias-vs-shape rule: if the algorithm clearly finds the RWP but a bound is too tight (bias), relax and document with a comment citing why; if it misses a documented RWP or violates a physical-signature invariant (wrong shape), that is a genuine bug to investigate.

---

## Layer 2: Climatological Distribution Validation  (full rollout)

### Goal

The most decisive correctness test. Run WAPER over a multi-year period and check that aggregate distributions match the published climatology in **shape, tolerance band, and relative (seasonal/spatial) behavior**. Phase 0 was the two-month preview of exactly these checks; this is the full version, and it is where performance and parallelization matter.

### Step 2.1: Distributional targets

Run over several years of ERA5 NH 300 hPa meridional wind (`v_winds_300mb_nh_2022_2023.nc` to start; extend as needed). Compare:

| Statistic | NH target | SH target | Source |
|---|---|---|---|
| Mean duration | 5.8 d (CI 4.9–7.0) | 7.9 d (CI 6.7–9.6) | Table 1 |
| % duration < 8 days | 71.8% | 69.2% | Table 1 |
| Max duration | ~47 d (rare; ~13 events/32 yr NH) | ~56 d | Table 1 |
| Mean eastward propagation | 119° (104–139) | 157° (142–177) | Table 1 |
| % propagation < 180° | 82.7% | 66.3% | Table 1 |
| Mean / median peak WPA | 27.1 / 23.8 m/s | — | §3a |
| 95th-pct amplitude | 29.7 m/s | — | §3a |
| P(WPA > 30 m/s) | 0.065 | 0.041 | §3a |
| Total count | ~200/yr globally | — | §3a |

### Step 2.2: Shape and relative-behavior checks (bias-immune)

These survive systematic calibration offsets and are the strongest evidence of correctness:

- **Amplitude PDF** unimodal, peak ~20 m/s, skewed to large values (Souders Fig. 1).
- **Duration PDF** ~70% short-lived (<8 d) with a long tail past 25 d (Souders Table 1).
- **Seasonal cycle:** NH strong (≈80% drop in formations in JJA), SH weak (≈40% drop in DJF) (Souders Fig. 4). Relative, so immune to absolute calibration — a key check.
- **Spatial frequency maxima** collocated with the Pacific / W-Atlantic / S-Indian storm tracks; NH formation hotspots at 140°E–170°W and 80°–60°W (Souders Figs. 2–3).
- **Extreme RWPs** originate in the N Pacific >70% of the time but reach peak intensity over the W Atlantic (Souders §3a).

### Step 2.3: Cross-statistic correlation checks (cheap, strong)

Pairwise relationships from Souders Fig. 8 — easy to compute, catch subtle bugs, and include a useful negative control:

- amplitude ↔ size: r ≈ 0.75
- duration ↔ propagation: r ≈ 0.75
- duration ↔ max amplitude: r ≈ 0.59
- amplitude ↔ group velocity: **r ≈ 0.09 (no relation)** — negative control. If the pipeline shows strong amplitude–speed coupling, something is wrong.

### Step 2.4: Performance and parallelization

A multi-year run is where compute becomes real. Design notes:

- **Detection is embarrassingly parallel across timesteps** — fan out per-timestep RWP identification with joblib/Dask; one worker per timestep chunk.
- **Tracking is sequential** but chunkable: split the time axis into overlapping windows (overlap ≥ max plausible track length so no track is cut), track each window, then stitch tracks across window boundaries by footprint overlap.
- **Stream, don't hold:** never load the full multi-year field into memory. Process per-timestep, persist RWP objects/footprints to disk (e.g. Parquet/Zarr), then aggregate statistics in a second pass.
- **Cache intermediate products** (per-timestep detections) so re-running only the statistics layer is cheap.
- Benchmark on one season first; extrapolate before committing to the full archive.

### Step 2.5: Automate as a script

`scripts/climatology_validation.py`: takes a dataset path, year range, and output dir; runs detection+tracking (parallelized per Step 2.4); writes per-RWP records to disk; computes the Step 2.1–2.3 statistics; emits comparison plots overlaying WAPER distributions on the published targets. Diagnostic tool, not a pytest test.

---

## Layer 3: Parameter Sensitivity Analysis

### Goal

Reproduce Pandey et al. (2020) Section 3 with the refactored pipeline and extend to the parameters introduced in Phase 3 (`cluster_eps_km`, `min_longitude_separation`). Serves as validation (curves should be qualitatively similar to the paper) and as the tool for choosing defaults. Layer 3 generalizes the Phase 0 Step 0.3 threshold sweep to all parameters and the full Pandey statistics.

### Step 3.1: Reproduce Fig. 4

- 300 hPa meridional wind, 4 DJF seasons (paper used 1990/1995/2000/2005 ERA-Interim; use ERA5 equivalents).
- Sweep **GT** and **ST**, now reconciled against the code (verified in `waper/interface/api.py` and `rwp_graph.py`):
  - **ST = `node_pruning_threshold`** — scalar amplitude in the *field's units* (m/s for raw `v`). A max/min node-pair survives if `min_scalar ≥ ST`. Code default **20**; Pandey ST range **25–50**; sweep in m/s directly.
  - **GT = `edge_pruning_threshold`** — an edge weight `(max_scalar − min_scalar) / distance_km × zonal_fraction`, units **(m/s)/km**. Code default in `api.py` is **`3e-5`**, which is effectively *zero* — it prunes almost no edges and is a prime suspect for the greedy-over-connection pathology (Phase 0 Step 0.2). The tuned `datasets/visualize.py` instead uses **`0.02`**, which sits squarely in Pandey's **0.0–0.08 (m/s)/km** range. **Sweep GT 0.0–0.08; do not use 3e-5 or the old "0.3".**
- For each (GT, ST): mean edges per RWP, mean/median edge length (km), mean RWP extent (km), number of timesteps with ≥1 RWP.
- Plot the 8-panel figure; compare to Pandey Fig. 4.

**Expected outcome:** qualitatively similar curves. The DBSCAN change happens upstream of pruning and should not dramatically alter these. If curves differ, check whether the distance-scaling fix (Task 3.6) or the centroid change (Task 3.3) is responsible. Cross-check absolute edge-length/extent scales against the wavenumber 5–8 expectation and Souders' size–amplitude relation.

### Step 3.2: Extend to new parameters

Sweep the actual clustering/connection parameters in `WaperConfig` (the old draft referenced a `cluster_eps_km` that does not exist — clustering is OPTICS, not fixed-eps DBSCAN):
- `cluster_max_eps_km` (default 3000) and `cluster_xi` (default 0.15) — the OPTICS reachability controls.
- `min_longitude_separation` (default 6.0) — minimum zonal gap between connected extrema; sweep 3–10.
- `max_aspect_ratio` (default 1.5) — discards near-vertical edges; relevant to the oversized/spurious-connection question.

Hold GT/ST at the reconciled defaults. Plot the same 4 statistics; identify stability plateaus (the same plateau logic as Phase 0 Step 0.3, now across the full parameter set).

### Step 3.3: Automate as a script

`scripts/parameter_sensitivity.py`: dataset path, parameter ranges, output dir; runs the sweep (parallelizable across combinations); saves CSV + summary plots. Diagnostic, not a test. Reuse for the clustering investigation (DBSCAN vs OPTICS) — run before and after any clustering change.

---

## Layer 4: Cross-Method Comparison

### Goal

Compare WAPER against (a) a standard Hilbert-envelope method and (b) the Souders feature-based tracker, and adopt Souders' verification protocol to produce a single literature-comparable quality number. Neither method is "correct"; agreement on clear cases builds confidence, disagreement on ambiguous cases is expected and scientifically interesting.

### Step 4.1: Implement the standard envelope (match the published spec exactly)

```python
def compute_rwp_envelope(v_field, wavenumber_range=(3, 11)):
    """Zimin et al. (2003/2006) Hilbert envelope of 300 hPa meridional wind."""
    # 1. FFT along longitude
    # 2. Zero out wavenumbers outside range
    # 3. Inverse FFT
    # 4. Hilbert transform -> envelope
    # 5. Return envelope field
```

Match Souders so the comparison is fair: 300 hPa v, **zonal wavenumbers 3–11** (Souders) — *not* the 4–15 in the old draft — Hilbert envelope, and a **14 m/s** tracking threshold (the old draft's 20 m/s does not match Souders' object definition). ~30 lines of numpy/scipy.

### Step 4.2: Comparison metrics

- **Detection agreement:** threshold the envelope at 14 m/s; fraction of timesteps where both methods detect ≥1 RWP (expect > 0.8 for DJF); fraction where only one detects (the interesting cases).
- **Spatial overlap:** for co-detected timesteps, IoU between WAPER footprint polygons and thresholded envelope regions; report mean and distribution.
- **Amplitude correlation:** mean envelope amplitude within each WAPER footprint vs WAPER's scalar weight; expect r > 0.5 for well-defined RWPs.

### Step 4.3: Souders-tracker comparison and POD/FAR protocol

You already have `souders_v_1.nc` / `souders_v_2.nc` in `datasets/` — the Souders tracker is the closest analog to WAPER (both feature/object-based, not Hovmöller), so it is the most informative external reference.

- Adopt **Souders' verification protocol** directly: manually label a sample of timesteps (reuse the Layer 1 case-study events, especially the two merge events), then compute WAPER's **probability of detection** and **false-alarm ratio** against that labeling. Targets to match: POD ≈ 93%, FAR ≈ 20%, with merge/split events expected to dominate the false alarms (~75% in Souders).
- This single POD/FAR pair, plus the Phase 0 Step 0.4 graph probe, is the quantitative statement of how trustworthy the tracker is — and it specifically stress-tests the merge/split handling that `clustering_investigation_plan.md` cares about.

### Step 4.4: Run on a full season

Run all methods on one DJF season. Report: overall detection agreement, spatial-overlap distribution, amplitude correlation, POD/FAR, and a catalogue of high-disagreement cases for manual inspection (e.g. the April 2011 case where WAPER captures non-wavelike structure the envelope misses).

### Step 4.5: Automate as a script

`scripts/cross_method_comparison.py`: dataset in, runs WAPER + envelope (+ Souders comparison where data exist), outputs metrics and diagnostic plots.

---

## Implementation order

1. **Phase 0** — gate. Sanity-check on `validation.nc`: ballpark statistics, oversized-packet diagnostic, threshold plateau, split/merge graph probe. **Do not proceed until this passes.**
2. **Layer 1** — case-study regression tests; integrates into the existing suite. Start with Jan 2007 and Apr 2011 (also the merge cases).
3. **Layer 2** — climatological distribution validation; the most decisive test. Needs the Layer 2.4 performance/parallelization design. Run on `v_winds_300mb_nh_2022_2023.nc` first, then extend.
4. **Layer 3** — parameter sensitivity; generalizes the Phase 0 threshold sweep. Also drives the clustering investigation.
5. **Layer 4** — cross-method comparison and POD/FAR; the heaviest lift (envelope + Souders comparison), more scientific-validation than regression.

## Relationship to other plans

- **Western-disturbance applied extension** (`superpowers/plans/design/western_disturbance_validation_plan.md`): **Layer 5** — benchmarks WAPER against Hunt et al. (2018) on subtropical RWPs / western disturbances and uses WAPER's explicit trough–crest links to *update and exceed* Hunt (per-event RWP embedding, upstream provenance linked to the Layer 2 global climatology, downstream-development energetics, precursor lead times) on modern global ERA5. Reuses Phase 0 Step 0.3 for subtropical threshold tuning; builds on Layers 2 and 4.
- **Regimes → RWP structure** (`superpowers/plans/design/regime_rwp_structure_plan.md`): **Layer 5** sibling — maps the heatwave DSE regimes of Shah & Monteiro (2025) to explicit WAPER packet structures over the same NW-South-Asia hotspot, testing whether the same outcome arises from different packet structures (the Monteiro & Caballero analogue). Shares the subtropical tuning and region with the WD plan.
- **Serialization / query / visualization** (`superpowers/plans/archive/design/serialization_query_viz_plan.md`): the infrastructure under everything — a portable "RWP catalogue" on disk (implements Layer 2.4 streaming), a `Catalogue` query API that returns the reference quantities (amplitude/duration/wavenumber/seasonal-cycle/merges/provenance/region-phase) with no boilerplate so the validation layers just *call* it, and a HoloViz explorer (white-at-zero diverging colorbar, time slider/animation, toggleable nodes/edges/polygons/tracks) reading straight from the catalogue.
- **Clustering investigation** (`superpowers/plans/design/clustering_investigation_plan.md`): Phase 0 Steps 0.2–0.4 (oversized packets, threshold plateau, split/merge graph) directly feed it, and Layer 3 supports its spatial-scale-variability step. Run sensitivity before and after any clustering change.
- **Phase 4 tracking improvements:** Layer 1 temporal/split-merge invariants and the Layer 4 POD/FAR protocol depend on tracking working. The Phase 0 Step 0.4 graph probe is the first real test of the tracking graph — wire it in as Phase 4 tasks land.
- **Experiment log:** record Phase 0 tuning runs in `datasets/experiments/` (one md file per session), per existing convention.
