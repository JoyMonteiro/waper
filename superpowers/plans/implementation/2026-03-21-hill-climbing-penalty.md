# Hill-Climbing Penalty for Extremum Clustering

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dead zero-crossing penalty in `cluster_extrema` with a fractional-descent hill-climbing penalty that actually activates when the clip value is positive, properly separating extrema that belong to different wave crests/troughs within the same connected component.

**Architecture:** The distance matrix between extrema pairs already walks the Dijkstra path and samples field values (lines 100-138 of `topology.py`). We replace the zero-crossing check with a fractional descent calculation: `f = (endpoint_amplitude - min_along_path) / endpoint_amplitude`. This dimensionless fraction is then scaled by a characteristic length (`L_char`, default ~2000 km) and added to the geodesic distance. The penalty is dimensionless in its intermediate form, avoiding the unit mismatch between m/s and km.

**Tech Stack:** Python, NumPy, scikit-learn (DBSCAN), VTK (Dijkstra paths), PyVista, pytest

---

## Background

### The bug

The current penalty code (lines 128-134 of `waper/identification/topology.py`) checks whether the minimum value along a Dijkstra path between two maxima is **negative** (`penalty_v < 0`). However, the Dijkstra path is computed on the **clipped** mesh — a mesh where all points satisfy `|v| > clip_value`. Within a single connected component of the clipped mesh, all values have the same sign. Therefore, for any `clip_value > 0`, the penalty **never activates**. The distance matrix is just raw geodesic distance, with no field-aware component.

### The fix

Replace the sign-reversal check with a **fractional descent** (hill-climbing) penalty. Instead of asking "does the path cross zero?", ask "how much does the path descend relative to the extrema amplitude?"

For two maxima with values `val_i` and `val_j`, and `min_along_path` being the minimum field value sampled along the Dijkstra path between them:

```
reference = min(val_i, val_j)           # weaker of the two endpoints
f = (reference - min_along_path) / reference   # fractional descent, dimensionless
f = max(0, f)                           # clamp (path might not dip at all)
penalty = f * L_char                    # convert to km-equivalent
```

Properties:
- `f = 0`: path stays at or above the weaker extremum — no penalty (same ridge)
- `f = 0.5`: path dips to half the amplitude — significant penalty
- `f = 1.0`: path dips to zero — full `L_char` penalty
- `f > 1.0`: path crosses zero — penalty exceeds `L_char` (subsumes old behavior)

For minima (sign < 0), the logic is mirrored: find the maximum along the path, compute fractional ascent relative to the weaker (least negative) minimum.

### Key files

- **Modify:** `waper/identification/topology.py` — the `cluster_extrema` function (lines 14-193)
- **Modify:** `waper/interface/api.py` — add `penalty_length_scale_km` to `WaperConfig` (line 43-49)
- **Test:** `tests/test_clustering.py` — existing test file with clustering tests

### Important implementation details

- The field values along the Dijkstra path are sampled via `scalar_field.GetPointData().GetArray(scalar_name)` (the clipped mesh point data). If that array doesn't exist, it falls back to `cell_v` from `base_field.GetCellData()`. The original code has the `point_scalar_arr` lookup inside the inner loop (line 113), but since it doesn't change per iteration, we intentionally move it before the loop. This is a trivial cleanup, not a behavioral change. **Note:** the `cell_v` fallback uses a point ID to index into cell data — this is a pre-existing bug but out of scope for this plan.
- `SCALE_FACTOR = RADIUS_EARTH_KM / RADIUS_SPHERE` converts from VTK sphere units to km. It is applied to the final distance at line 136. The penalty (in km) should be added **before** this scaling, since the geodesic `dist` is also in VTK sphere units at that point. So `L_char` must be converted to sphere units: `L_char_sphere = L_char / SCALE_FACTOR`. Alternatively, add the penalty **after** scaling — this is simpler and equivalent. We'll add it after scaling.
- The extrema field values (`val_i`, `val_j`) are needed. These can be obtained from `extrema_points` point data for the scalar field. The scalar values are available via `extrema_points.GetPointData().GetArray(scalar_name)` or by looking up the point ID in the base field. Check which is available in Task 1.

---

## Task 1: Verify how to access extremum field values

**Files:**
- Read: `waper/identification/topology.py:59-80`
- Read: `tests/test_clustering.py:26-37`

This is a research task. Before writing code, confirm how to obtain the scalar field value at each extremum point.

- [ ] **Step 1: Write a diagnostic test**

Add a test that creates a simple field, runs the pipeline up to `cluster_extrema`, and prints the available point data arrays on `extrema_points`. This test exists only to confirm data availability and will be removed.

```python
def test_extrema_have_scalar_values():
    """Diagnostic: confirm extrema points carry the scalar field values."""
    lons = np.arange(0, 360, 5)
    lats = np.arange(20, 80.1, 5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    v = 30 * np.exp(-((lon2d - 180) ** 2 + (lat2d - 50) ** 2) / 100)

    da = xr.DataArray(
        v, dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons}, name="v",
    )
    data_with_max = max_min.add_maxima_data(da, "v", lons, lats)
    clipped = max_min.clip_dataset(data_with_max, "v", threshold=10)
    connectivity = topology.identify_connected_regions(clipped)
    maxima_points = max_min.extract_maxima_points(connectivity, 10, "v")

    assert maxima_points.GetNumberOfPoints() >= 1

    # Check what arrays are available
    pd = maxima_points.GetPointData()
    array_names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    print(f"Available arrays on extrema_points: {array_names}")

    # Check if scalar values are directly accessible
    scalar_arr = pd.GetArray("v")
    if scalar_arr is not None:
        val = scalar_arr.GetTuple1(0)
        print(f"Scalar value at first extremum: {val}")
        assert val > 10  # should be above clip threshold
    else:
        # Try via vtkOriginalPointIds lookup into connectivity
        orig_id = pd.GetArray("vtkOriginalPointIds").GetTuple1(0)
        print(f"Original point ID: {orig_id}, will need to look up value from connectivity or base_field")
```

- [ ] **Step 2: Run the diagnostic test**

Run: `pytest tests/test_clustering.py::test_extrema_have_scalar_values -v -s`

Read the output to determine which approach works for accessing scalar values at extremum locations. Record the finding (which array name, which lookup method). You will use this in Task 2.

- [ ] **Step 3: Remove the diagnostic test**

Delete `test_extrema_have_scalar_values` from `tests/test_clustering.py`. It was only for research.

---

## Task 2: Add the `penalty_length_scale_km` config parameter

**Files:**
- Modify: `waper/interface/api.py:43-49` (add field to `WaperConfig`)
- Modify: `waper/interface/api.py:134-136` (pass to `cluster_extrema` call for maxima)
- Modify: `waper/interface/api.py:176-178` (pass to `cluster_extrema` call for minima)
- Modify: `waper/identification/topology.py:14-22` (add parameter to function signature)

- [ ] **Step 1: Add config field**

In `waper/interface/api.py`, add to `WaperConfig` (after line 45):

```python
    penalty_length_scale_km: float = 2000.0
```

- [ ] **Step 2: Pass parameter through to `cluster_extrema` calls**

In `waper/interface/api.py`, update both `cluster_extrema` calls (lines ~134-136 and ~176-178) to pass the new parameter:

```python
    clustered_points = topology.cluster_extrema(
        data_with_maxima, connectivity, maxima_points, config.scalar_name, sign=1,
        max_eps_km=config.cluster_max_eps_km, min_samples=config.cluster_min_samples,
        xi=config.cluster_xi, penalty_length_scale_km=config.penalty_length_scale_km,
    )
```

(Same pattern for the minima call with `sign=-1`.)

- [ ] **Step 3: Add parameter to function signature**

In `waper/identification/topology.py`, update `cluster_extrema` signature (line 14-22):

```python
def cluster_extrema(
    base_field,
    connectivity_clipped_scalar_field,
    extrema_points,
    scalar_name,
    sign,
    max_eps_km=1500,
    min_samples=2,
    xi=0.05,
    penalty_length_scale_km=2000.0,
):
```

- [ ] **Step 4: Update test helper to accept the new parameter**

In `tests/test_clustering.py`, update `_create_and_process_field` (line 8):

```python
def _create_and_process_field(v, lons, lats, threshold=5, max_eps_km=1500, xi=0.05, penalty_length_scale_km=2000.0):
```

And pass it through in the `cluster_extrema` call inside that function (line 19-22):

```python
    clustered = topology.cluster_extrema(
        data_with_max, connectivity, maxima_points, "v",
        sign=1, max_eps_km=max_eps_km, xi=xi,
        penalty_length_scale_km=penalty_length_scale_km,
    )
```

- [ ] **Step 5: Run existing tests to confirm nothing breaks**

Run: `pytest tests/test_clustering.py -v`

Expected: All existing tests pass. The new parameter has a default value so no call sites break.

- [ ] **Step 6: Commit**

```bash
git add waper/interface/api.py waper/identification/topology.py tests/test_clustering.py
git commit -m "feat: add penalty_length_scale_km parameter to cluster_extrema

Threads the new parameter from WaperConfig through to the clustering
function. Default value 2000 km. No behavioral change yet — the penalty
logic is updated in the next commit.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Write failing tests for the hill-climbing penalty

**Files:**
- Modify: `tests/test_clustering.py`

These tests encode the desired behavior: two maxima connected by a path that dips significantly (but stays positive) should be placed in **different** clusters.

- [ ] **Step 1: Write test for two maxima separated by a valley (same sign)**

This is the core test case. Two strong maxima (+30 m/s) connected by a broad base that dips to a low positive value (~6 m/s). They're in the same connected component but the fractional descent is high, so the penalty should push their effective distance beyond `max_eps_km`, placing them in different clusters.

```python
def test_hill_climbing_penalty_separates_dipped_maxima():
    """Two maxima connected by a same-sign valley should be split by hill-climbing penalty."""
    lons = np.arange(0, 360, 2.5)
    lats = np.arange(20, 80.1, 2.5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Two strong maxima 60 degrees apart, connected by a low-amplitude bridge
    peak1 = 30 * np.exp(-((lon2d - 160) ** 2 + (lat2d - 50) ** 2) / 50)
    peak2 = 30 * np.exp(-((lon2d - 220) ** 2 + (lat2d - 50) ** 2) / 50)
    # Bridge that keeps them connected (above clip=5) but dips to ~6 m/s
    bridge = 6 * np.exp(-((lon2d - 190) ** 2 + (lat2d - 50) ** 2) / 2000)
    v = peak1 + peak2 + bridge

    clustered = _create_and_process_field(
        v, lons, lats, threshold=5, max_eps_km=3000,
        penalty_length_scale_km=2000.0,
    )

    clusters = np.unique(clustered.point_data["Cluster ID"])
    # The penalty should separate them: fractional descent ~(30-6)/30 = 0.8
    # penalty = 0.8 * 2000 = 1600 km added to geodesic distance
    assert len(clusters) >= 2, (
        f"Expected >= 2 clusters (hill-climbing should separate dipped maxima), got {len(clusters)}"
    )
```

- [ ] **Step 2: Write test confirming no penalty when path stays high**

Two maxima on the same ridge with no significant dip between them should remain in the same cluster.

```python
def test_hill_climbing_no_penalty_when_ridge_stays_high():
    """Two maxima on the same ridge (no significant dip) should stay in one cluster."""
    lons = np.arange(0, 360, 2.5)
    lats = np.arange(20, 80.1, 2.5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Two peaks close together with overlapping high-amplitude regions
    peak1 = 30 * np.exp(-((lon2d - 180) ** 2 + (lat2d - 50) ** 2) / 80)
    peak2 = 25 * np.exp(-((lon2d - 195) ** 2 + (lat2d - 50) ** 2) / 80)
    v = peak1 + peak2

    clustered = _create_and_process_field(
        v, lons, lats, threshold=10, max_eps_km=3000,
        penalty_length_scale_km=2000.0,
    )

    if clustered.n_points >= 2:
        # If two maxima are detected, they should be in the same cluster
        clusters = np.unique(clustered.point_data["Cluster ID"])
        assert len(clusters) == 1, (
            f"Expected 1 cluster (ridge stays high, no penalty), got {len(clusters)}"
        )
```

- [ ] **Step 3: Write test for two minima separated by a ridge (same sign)**

This mirrors the maxima test but for minima (sign < 0), exercising the `max(val_i, val_j)` / `path_extreme_v - reference` branch. The test helper `_create_and_process_field` uses `sign=1`, so we need to build a minima-specific variant.

```python
def _create_and_process_minima_field(v, lons, lats, threshold=5, max_eps_km=1500, xi=0.05, penalty_length_scale_km=2000.0):
    """Like _create_and_process_field but for minima (sign=-1)."""
    da = xr.DataArray(
        v, dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons}, name="v",
    )
    data_with_min = max_min.add_minima_data(da, "v", lons, lats)
    clipped = max_min.clip_dataset(data_with_min, "v", threshold=-threshold)
    connectivity = topology.identify_connected_regions(clipped)
    minima_points = max_min.extract_minima_points(connectivity, -threshold, "v")
    clustered = topology.cluster_extrema(
        data_with_min, connectivity, minima_points, "v",
        sign=-1, max_eps_km=max_eps_km, xi=xi,
        penalty_length_scale_km=penalty_length_scale_km,
    )
    return clustered


def test_hill_climbing_penalty_separates_ridged_minima():
    """Two minima connected by a same-sign ridge should be split by hill-climbing penalty."""
    lons = np.arange(0, 360, 2.5)
    lats = np.arange(20, 80.1, 2.5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Two strong troughs 60 degrees apart, connected by a weak negative bridge
    trough1 = -30 * np.exp(-((lon2d - 160) ** 2 + (lat2d - 50) ** 2) / 50)
    trough2 = -30 * np.exp(-((lon2d - 220) ** 2 + (lat2d - 50) ** 2) / 50)
    bridge = -6 * np.exp(-((lon2d - 190) ** 2 + (lat2d - 50) ** 2) / 2000)
    v = trough1 + trough2 + bridge

    clustered = _create_and_process_minima_field(
        v, lons, lats, threshold=5, max_eps_km=3000,
        penalty_length_scale_km=2000.0,
    )

    clusters = np.unique(clustered.point_data["Cluster ID"])
    assert len(clusters) >= 2, (
        f"Expected >= 2 clusters (hill-climbing should separate ridged minima), got {len(clusters)}"
    )
```

**Note:** This test uses `add_minima_data`, `extract_minima_points`, and negative thresholds. If these functions don't exist or work differently, check `waper/identification/max_min.py` for the actual minima API and adapt accordingly. The key requirement is to run `cluster_extrema` with `sign=-1` on a field with two negative troughs connected by a weak bridge.

- [ ] **Step 4: Run the new tests to confirm they fail**

Run: `pytest tests/test_clustering.py::test_hill_climbing_penalty_separates_dipped_maxima tests/test_clustering.py::test_hill_climbing_no_penalty_when_ridge_stays_high tests/test_clustering.py::test_hill_climbing_penalty_separates_ridged_minima -v`

Expected: `test_hill_climbing_penalty_separates_dipped_maxima` and `test_hill_climbing_penalty_separates_ridged_minima` FAIL (currently the penalty is dead code, so the extrema end up in the same cluster). `test_hill_climbing_no_penalty_when_ridge_stays_high` may pass or fail depending on geodesic distance — either outcome is acceptable at this stage.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/test_clustering.py
git commit -m "test: add failing tests for hill-climbing penalty

Three tests encoding desired behavior:
- Maxima separated by a same-sign valley should be split
- Minima separated by a same-sign ridge should be split
- Maxima on a continuous ridge should stay together

The separation tests fail because the penalty code never activates
when clip_value > 0 (the zero-crossing check is dead code).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Implement the fractional descent penalty

**Files:**
- Modify: `waper/identification/topology.py:96-138`

This is the core change. Replace the zero-crossing penalty with the fractional descent calculation.

- [ ] **Step 1: Get extremum scalar values**

Based on findings from Task 1, add code to extract the scalar value at each extremum point. Add this **before** the distance matrix loop (before line 81), after the `point_coords` block:

```python
    # Extract scalar values at each extremum for hill-climbing penalty.
    extrema_scalar_values = np.zeros(num_points)
    # Try point data on extrema_points first
    extrema_scalar_arr = extrema_points.GetPointData().GetArray(scalar_name)
    if extrema_scalar_arr is not None:
        for i in range(num_points):
            extrema_scalar_values[i] = extrema_scalar_arr.GetTuple1(i)
    else:
        # Fall back: look up via original point ID in the clipped scalar field
        sf_scalar_arr = scalar_field.GetPointData().GetArray(scalar_name)
        for i in range(num_points):
            orig_id = int(extrema_point_id.GetTuple1(i))
            extrema_scalar_values[i] = sf_scalar_arr.GetTuple1(orig_id)
```

**Note:** If Task 1 revealed a different access pattern, adapt this code accordingly. The key requirement is that `extrema_scalar_values[i]` holds the field value (in m/s) at extremum `i`.

- [ ] **Step 2: Replace the penalty calculation**

Replace lines 100-136 (from `penalty_v = 1000 * sign` through `final_dist = (dist + penalty) * SCALE_FACTOR`) with:

```python
            # Track the extreme value along the Dijkstra path for hill-climbing penalty.
            # For maxima (sign>0): find minimum along path.
            # For minima (sign<0): find maximum along path.
            path_extreme_v = extrema_scalar_values[i]  # initialize to endpoint value

            for ptId in range(pts.GetNumberOfPoints() - 1):
                pts.GetPoint(ptId, p0)
                pts.GetPoint(ptId + 1, p1)
                dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))

            # Look up point_scalar_arr once (moved out of inner loop — same result,
            # since the array doesn't change per iteration).
            point_scalar_arr = scalar_field.GetPointData().GetArray(scalar_name)
            for ptIdx in range(id_list.GetNumberOfIds()):
                vid = id_list.GetId(ptIdx)
                if point_scalar_arr:
                    val = point_scalar_arr.GetTuple1(vid)
                else:
                    val = cell_v.GetTuple1(vid)

                if sign > 0:
                    if val < path_extreme_v:
                        path_extreme_v = val
                else:
                    if val > path_extreme_v:
                        path_extreme_v = val

            # Hill-climbing penalty: fractional descent from weaker endpoint.
            #
            # For maxima (sign>0): reference is the weaker (smaller) peak value.
            #   descent = reference - path_minimum. Positive when path dips below
            #   the weaker peak. Example: peaks at 30 and 25, path dips to 10.
            #   reference=25, descent=15, f=0.6.
            #
            # For minima (sign<0): reference is the weaker (least negative) trough.
            #   descent = path_maximum - reference. Positive when path rises above
            #   the weaker trough. Example: troughs at -20 and -18, path rises to -5.
            #   reference=-18, descent=(-5)-(-18)=13, f=13/18=0.72.
            val_i = extrema_scalar_values[i]
            val_j = extrema_scalar_values[j]

            if sign > 0:
                reference = min(val_i, val_j)
                descent = reference - path_extreme_v
            else:
                reference = max(val_i, val_j)
                descent = path_extreme_v - reference

            abs_ref = abs(reference)
            if abs_ref > 0:
                f = max(0.0, descent / abs_ref)
            else:
                f = 0.0

            penalty_km = f * penalty_length_scale_km

            final_dist = dist * SCALE_FACTOR + penalty_km
```

- [ ] **Step 3: Run the hill-climbing tests**

Run: `pytest tests/test_clustering.py::test_hill_climbing_penalty_separates_dipped_maxima tests/test_clustering.py::test_hill_climbing_no_penalty_when_ridge_stays_high tests/test_clustering.py::test_hill_climbing_penalty_separates_ridged_minima -v`

Expected: All three PASS.

- [ ] **Step 4: Run all clustering tests**

Run: `pytest tests/test_clustering.py -v`

Expected: All tests pass, including pre-existing ones. If any pre-existing test fails, it means the penalty is too aggressive for that test's field configuration. Adjust the test's `penalty_length_scale_km` parameter or the field setup — do NOT weaken the penalty implementation.

- [ ] **Step 5: Commit**

```bash
git add waper/identification/topology.py
git commit -m "fix: replace dead zero-crossing penalty with fractional-descent hill-climbing

The old penalty only activated when the Dijkstra path crossed zero, which
never happens on a clipped mesh (all values same sign). The new penalty
measures the fractional descent from the weaker endpoint:

  f = (reference - path_extreme) / |reference|
  penalty = f * L_char_km

This is dimensionless in intermediate form, avoiding the m/s vs km unit
mismatch. A characteristic length scale (default 2000 km) converts
the fraction to a km-equivalent distance added to the geodesic distance.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Run integration tests and validate with datasets

**Files:**
- Run: `tests/` (full test suite)
- Run: `datasets/visualize.py` (regenerate figures)

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`

Expected: All tests pass.

- [ ] **Step 2: Regenerate dataset figures**

Run: `python datasets/visualize.py`

This regenerates the cluster plots. Visually inspect the output images in `datasets/figures/` to confirm:
1. Large zonally-extended clusters are now split where the field dips between crests
2. Meridionally-extended clusters with no significant dip are preserved as single clusters
3. No obvious regressions (features that were correctly clustered before are still correct)

- [ ] **Step 3: If tests or visual inspection reveal issues**

Common issues and fixes:
- **Too aggressive** (splitting clusters that should be together): Increase `penalty_length_scale_km` default from 2000 to 3000 in both `topology.py` and `api.py`
- **Not aggressive enough** (still merging clusters that should be split): Decrease `penalty_length_scale_km` to 1500, or decrease `max_eps_km`
- **Minima logic wrong**: Double-check the sign convention in Task 4 Step 2 — `path_extreme_v` for minima is the **maximum** (least negative) value along the path

- [ ] **Step 4: Commit any adjustments**

```bash
git add -A
git commit -m "fix: tune penalty_length_scale_km based on dataset validation

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
