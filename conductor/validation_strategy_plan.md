# WAPER Validation Strategy

> **Context:** This plan should be picked up after the current refactoring work (Phases 4+) is complete and the pipeline runs end-to-end on real data. The three layers build on each other — start with Layer 1, which is the most immediately actionable.

## Background

There is no universally agreed-upon ground truth for RWPs. The definition is inherently ambiguous — spatial extent, wavenumber, and amplitude all vary during the lifecycle. Manual annotation is subjective, envelope methods lose phase information, and no labeled dataset exists.

Instead of ground truth, we validate through three complementary layers:
1. **Case-study regression tests** — physically motivated invariants on known synoptic events
2. **Parameter sensitivity analysis** — reproducing and extending the Pandey et al. (2020) Fig. 4 analysis
3. **Cross-method comparison** — automated agreement metrics against an envelope method

---

## Layer 1: Case-Study Regression Tests

### Goal

Build a pytest-based test suite where each test runs the full WAPER pipeline on a real synoptic event and asserts structural invariants. These tests catch regressions: if a code change causes the algorithm to miss a well-documented RWP or produce a physically impossible one, the test fails.

### Step 1.1: Select events

Choose 5–8 events that cover the range of RWP behaviors. Candidates from the paper and literature:

| Event | Dates | Behavior | Source |
|-------|-------|----------|--------|
| Japan cyclogenesis RWP | 21–23 Jan 2007 | Genesis forced by a deepening cyclone, downstream amplification over Pacific | Pandey et al. 2020, Fig. 5; Souders et al. 2014b Figs. 6–7 |
| Cutoff low RWP genesis | 6–10 Jan 2007 | Weak RWP genesis at 120°E, merges with existing RWP, grows Pacific-to-Europe | Pandey et al. 2020, Fig. 6; Souders et al. 2014b |
| April 2011 forecast bust | 12–17 Apr 2011 | Non-wavelike RWP over Atlantic, merge of Pacific and Russian RWPs | Pandey et al. 2020, Fig. 7; Ghinassi et al. 2018 |
| South Asia extreme wet-bulb | May–Jun composite | Weak-amplitude RWP over central Asia associated with extreme temperatures | Pandey et al. 2020, Fig. 8–9; Monteiro and Caballero 2019 |
| Strong winter RWP | DJF, pick a clear case | Classic well-defined wavelike RWP with 3+ nodes, clean structure | To be identified from ERA5 |
| Summer weak RWP | JJA, pick a case | Lower-amplitude RWP that should still be detected | To be identified from ERA5 |
| Dateline-crossing RWP | Pick a case | RWP that crosses 180° to test wraparound handling | To be identified from ERA5 |
| Quiet period | Pick 2–3 days with no significant RWP activity | Should produce few or no RWPs | To be identified from ERA5 |

The first four are documented in the paper and provide immediate test cases. The last four should be identified during initial real-data testing and added as they are encountered.

### Step 1.2: Prepare test fixtures

For each event:

1. Download 300 hPa meridional wind from ERA5 (or ERA-Interim for paper-comparable results) for the relevant dates and domain.
2. Store as compressed NetCDF in `tests/fixtures/events/` (e.g., `tests/fixtures/events/jan2007_genesis.nc`).
3. Keep file sizes manageable — crop to the relevant domain if the event is regional (e.g., the South Asia case can use 0°–140°E, 10°–70°N). For global events, use the full NH domain.
4. Add a YAML metadata file per event (`tests/fixtures/events/jan2007_genesis.yaml`) recording:
   - Source dataset and variable
   - Date range and domain bounds
   - Brief description and literature reference
   - The invariants (see Step 1.3)

### Step 1.3: Define invariants

Invariants are physically motivated assertions — not pixel-perfect expected outputs. They should be:
- **Robust to algorithm changes** — a better clustering algorithm should still satisfy them
- **Falsifiable** — a broken algorithm should violate at least some of them
- **Documentable** — each invariant should cite why it must hold

Categories of invariants:

**Existence invariants** — an RWP must (or must not) exist in a region/time:
```python
# At least one RWP spanning the Pacific on 23 Jan 2007
assert any(
    rwp.west_lon < 150 and rwp.east_lon > -120  # approximate Pacific extent
    for rwp in results["2007-01-23T06"]
)

# No RWP activity in the tropics
for timestep in results:
    for rwp in results[timestep]:
        assert rwp.mean_lat > 15, f"Spurious tropical RWP at {timestep}"
```

**Structural invariants** — properties of detected RWPs:
```python
# RWPs should have alternating structure (at least 2 edges = 3 nodes)
# for the well-defined winter case
for rwp in strong_winter_rwps:
    assert rwp.num_edges >= 2, "RWP too short to be physically meaningful"

# Detected RWPs should not span more than ~180° in longitude
# (longer than a hemisphere is almost certainly a merge artifact)
for rwp in all_rwps:
    assert rwp.zonal_extent_deg < 180
```

**Temporal invariants** (once tracking is working):
```python
# The Jan 2007 genesis RWP should be trackable for at least 2 days
pacific_tracks = [t for t in tracks if t.overlaps_region(pacific_box)]
assert any(t.duration_hours >= 48 for t in pacific_tracks)

# Tracked RWPs should propagate eastward on average
for track in tracks:
    if track.duration_hours >= 24:
        assert track.mean_propagation_speed_kmh > 0, "Westward propagation is unphysical for RWPs"
```

**Non-detection invariants** — quiet periods should be quiet:
```python
# During quiet period, fewer than N RWPs per timestep
for timestep in quiet_period:
    assert len(results[timestep]) <= 2, f"Too many RWPs during known quiet period"
```

### Step 1.4: Implement test structure

```
tests/
  fixtures/
    events/
      jan2007_genesis.nc
      jan2007_genesis.yaml
      apr2011_bust.nc
      apr2011_bust.yaml
      ...
  test_case_studies.py          # parametrized over events
  conftest.py                   # shared fixtures, pipeline runner
```

In `conftest.py`:
```python
@pytest.fixture(scope="module")
def pipeline_results(request):
    """Run WAPER pipeline on a test event, cached per module."""
    event = request.param
    data = xr.open_dataset(f"tests/fixtures/events/{event}.nc")
    config = WaperConfig()  # default parameters
    waper = Waper(config)
    return waper.identify_rwps(data)
```

In `test_case_studies.py`, parametrize tests over events and their invariants:
```python
@pytest.mark.parametrize("event,invariants", load_event_invariants())
def test_event_invariants(pipeline_results, event, invariants):
    for invariant in invariants:
        invariant.check(pipeline_results)
```

### Step 1.5: Run and calibrate

The first time you run these tests on real data, some invariants will fail — either because the invariant is too strict or because the algorithm has a genuine issue. This calibration step is essential:
- If the algorithm clearly identifies the RWP but the invariant bounds are too tight, relax them.
- If the algorithm misses a well-documented RWP, that's a genuine bug to investigate.
- Document any relaxations with a comment explaining why.

---

## Layer 2: Parameter Sensitivity Analysis

### Goal

Reproduce the Pandey et al. (2020) Section 3 analysis with the refactored pipeline, and extend it to the new parameters introduced during Phase 3 (cluster_eps_km, min_longitude_separation). This serves both as validation (results should be qualitatively similar to the paper) and as a tool for choosing default parameters.

### Step 2.1: Reproduce Fig. 4

- Use 300 hPa meridional winds for 4 DJF seasons (the paper used 1990, 1995, 2000, 2005 from ERA-Interim; use ERA5 equivalents or the same ERA-Interim data).
- Sweep GT (gradient threshold, = `edge_pruning_threshold`) from 0.0 to 0.08.
- Sweep ST (scalar threshold, = `node_pruning_threshold`) from 25 to 50.
- For each (GT, ST) pair, compute:
  - Mean number of edges per RWP
  - Mean and median edge length (km)
  - Mean RWP extent (km)
  - Number of timesteps containing at least one RWP
- Plot the 8-panel figure and compare to Fig. 4.

**Expected outcome:** Qualitatively similar curves. The DBSCAN change should not dramatically alter these statistics since clustering happens upstream of pruning. If the curves differ significantly, investigate whether the distance scaling fix (Task 3.6) or the centroid change (Task 3.3) is responsible.

### Step 2.2: Extend to new parameters

Add sweeps for:
- `cluster_eps_km`: 200, 300, 400, 500, 600, 800, 1000
- `min_longitude_separation`: 3, 4, 5, 6, 8, 10

Plot the same 4 statistics against these new parameters (holding GT=0.3, ST=30 fixed). Identify stability plateaus.

### Step 2.3: Automate as a script

Create `scripts/parameter_sensitivity.py` that:
- Takes a dataset path, parameter ranges, and output directory as arguments
- Runs the sweep (parallelizable across parameter combinations)
- Saves results as a CSV and generates the summary plots
- This script is not a pytest test — it's a diagnostic tool to be run manually when parameters change

---

## Layer 3: Cross-Method Comparison

### Goal

Compare WAPER's output against a Hilbert-transform envelope method to measure agreement. Neither method is "correct" — agreement on clear cases builds confidence, disagreement on ambiguous cases is expected.

### Step 3.1: Implement a minimal envelope method

Implement the Zimin et al. (2003) envelope as a standalone function:
```python
def compute_rwp_envelope(v_field, wavenumber_range=(4, 15)):
    """Hilbert transform envelope of meridional wind, filtered to wavenumber range."""
    # 1. FFT along longitude
    # 2. Zero out wavenumbers outside range
    # 3. Inverse FFT
    # 4. Hilbert transform to get envelope
    # 5. Return envelope field
```

This is ~30 lines of numpy/scipy. No need for a full package — just enough to get an envelope field for comparison.

### Step 3.2: Define comparison metrics

For each timestep, given WAPER's RWP paths and the envelope field:

**Detection agreement:**
- Threshold the envelope at a standard value (e.g., 20 m/s as in Souders et al. 2014b) to get envelope-detected RWP regions.
- Compute the fraction of timesteps where both methods detect at least one RWP (expect > 0.8 for DJF).
- Compute the fraction where only one method detects an RWP — these are the interesting cases.

**Spatial overlap:**
- For co-detected timesteps, compute the IoU (intersection over union) between WAPER's footprint polygons and the thresholded envelope regions.
- Report mean and distribution of IoU across timesteps.

**Amplitude correlation:**
- For each WAPER-detected RWP, compute the mean envelope amplitude within its footprint.
- Correlate WAPER's scalar weight with this envelope amplitude.
- Expect a positive correlation (r > 0.5) for well-defined RWPs.

### Step 3.3: Run on a full season

Run both methods on one DJF season. Report:
- Overall detection agreement rate
- Spatial overlap distribution
- Amplitude correlation
- Catalogue of high-disagreement cases for manual inspection (these are scientifically interesting — see the April 2011 case where WAPER captures non-wavelike structure that the envelope misses)

### Step 3.4: Automate as a script

Create `scripts/cross_method_comparison.py` with the same structure as the sensitivity script — takes a dataset, runs both methods, outputs comparison metrics and diagnostic plots.

---

## Implementation order

1. **Layer 1** first — it's the most immediately useful and integrates into the existing test suite. Start with the January 2007 and April 2011 events from the paper.
2. **Layer 2** after Layer 1 — it requires the pipeline to run on larger datasets. The sensitivity script is valuable for evaluating the clustering investigation (DBSCAN vs OPTICS) as well.
3. **Layer 3** last — it requires implementing the envelope method and is more of a scientific validation tool than a regression test.

## Relationship to other plans

- **Clustering investigation** (`conductor/clustering_investigation_plan.md`): Layer 2 directly supports Step 2 (characterize spatial scale variability) of that plan. The sensitivity analysis should be run before and after any clustering algorithm change.
- **Phase 4 tracking improvements**: Layer 1 temporal invariants depend on tracking working correctly. Add tracking-specific invariants as Phase 4 tasks are completed.
