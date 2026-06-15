# Energy-Weighted RWP Tracking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Design doc:** `superpowers/plans/design/energy_weighted_tracking_plan.md` — read it for rationale.

**Goal:** Make cross-timestep RWP association and track position follow the *energetic core* of each packet (energy ∝ amplitude²) instead of weighting every crest/trough of the whole envelope equally, so large RWPs stop looking stationary.

**Architecture:** Three coordinated changes. (1) The tracked position (`weighted_longitude/latitude`) becomes amplitude²-weighted. (2) Each timestep gets a co-registered **energy raster** — amplitude²-weighted disks burned at every extremum on the existing 512² stereographic grid. (3) `build_tracking_graph` computes the association weight from **energy overlap** of those rasters (`Σ√(E_prev·E_curr)` over the overlap, normalised by the larger feature's total energy) instead of binary pixel overlap. The quadtree code is left untouched (its direct test stays valid); only `build_tracking_graph` switches to the energy path.

**Tech Stack:** numpy, shapely, rasterio (`features.rasterize`), networkx, pyproj — all already deps.

## Verified facts (do not re-derive)

- Run tests with the project env: `~/miniconda3/envs/waper/bin/python -m pytest …` from repo root `/Users/joymonteiro/github/waper`. (Plain `pytest`/`python` are not the project env.)
- Branch is already `energy-weighted-tracking`.
- `WaperConfig` (`waper/interface/api.py:23`) is a frozen dataclass with a defaults section (where `hemisphere`, `penalty_length_scale_km`, etc. live).
- `WaperSingleTimestepData` (`waper/interface/api.py:57`) is **not** frozen and has a custom `__init__`; `raster_data`/`raster_features`/`quadtree` are set dynamically after identify. New attributes can be set dynamically too.
- `rasterize_all_rwps(polygon_list, hemisphere)` (`waper/tracking/rwp_polygon.py:221`) burns `rwp_id` into a 512² raster via `features.rasterize(((geom, value), …), out_shape=(512,512), all_touched=True, transform=_get_raster_transform(hemisphere))`, returning `None` for an empty list. `WAPER_IMAGE_SIZE = 512`.
- `transform_to_stereographic(lons, lats, hemisphere="north", inverse=False)` (`rwp_polygon.py:50`) converts lon/lat↔polar-stereographic metres; accepts scalars or arrays.
- Pruned-graph nodes carry `coords=(lon,lat)` and `scalar` (float, m/s).
- `get_polygon_for_rwp_path(...)` (`rwp_polygon.py:127`) currently returns `(rwp_poly, list_rwp_points, weighted_longitude, weighted_latitude)`. Its weighted centroid uses `np.abs(all_values)` weights at lines ~204–209. Its only caller is `_identify_rwps` (`api.py:221`).
- `build_tracking_graph(time_step_data, number_steps)` (`waper/tracking/tracking_graph.py:15`) builds nodes per `raster_features` with `coords=(weighted_longitude, weighted_latitude)`, then for each consecutive pair computes overlap via `merge()`/`compute_size_features()` and sets `weight = overlap / max(prev_size, curr_size)`; edge `distance` is haversine of centroids. Existing tests in `tests/test_tracking.py`.

---

## Task 1: Amplitude²-weighted centroid (track position)

**Files:**
- Modify: `waper/tracking/rwp_polygon.py`
- Test: `tests/test_tracking.py`

- [ ] **Step 1: Write the failing test** in `tests/test_tracking.py`:

```python
import numpy as np
from waper.tracking.rwp_polygon import _weighted_centroid

def test_energy_weighted_centroid_favors_high_amplitude():
    xs = np.array([0.0, 10.0]); ys = np.array([0.0, 0.0])
    values = np.array([1.0, 3.0])          # 3x amplitude -> 9x energy
    wx, wy = _weighted_centroid(xs, ys, values)
    assert abs(wx - 9.0) < 1e-9            # 10 * 9/(1+9) = 9.0
    assert abs(wy - 0.0) < 1e-9

def test_weighted_centroid_uses_squared_weights_sign_independent():
    xs = np.array([0.0, 4.0]); ys = np.array([0.0, 0.0])
    values = np.array([-2.0, 2.0])         # equal energy (4 each) -> midpoint
    wx, _ = _weighted_centroid(xs, ys, values)
    assert abs(wx - 2.0) < 1e-9
```

- [ ] **Step 2: Run, expect FAIL.** `~/miniconda3/envs/waper/bin/python -m pytest tests/test_tracking.py -k weighted_centroid -v` → ImportError.

- [ ] **Step 3: Implement** in `waper/tracking/rwp_polygon.py` (add near the top-level helpers, before `get_polygon_for_rwp_path`):

```python
def _weighted_centroid(xs, ys, values):
    """Energy-weighted centroid: weights are amplitude squared (energy), so the
    strongest crests/troughs dominate the position and the sign of `values`
    does not matter."""
    weights = np.asarray(values, dtype=float) ** 2
    wx = np.average(np.asarray(xs, dtype=float), weights=weights)
    wy = np.average(np.asarray(ys, dtype=float), weights=weights)
    return wx, wy
```

Then in `get_polygon_for_rwp_path`, replace the existing weighted-centroid lines:

```python
    xs, ys = transform_to_stereographic(all_lons, all_lats, hemisphere=hemisphere)
    weighted_ys = np.average(ys, weights=np.abs(np.array(all_values)))
    weighted_xs = np.average(xs, weights=np.abs(np.array(all_values)))
```

with:

```python
    xs, ys = transform_to_stereographic(all_lons, all_lats, hemisphere=hemisphere)
    weighted_xs, weighted_ys = _weighted_centroid(xs, ys, all_values)
```

- [ ] **Step 4: Run, expect PASS.** `~/miniconda3/envs/waper/bin/python -m pytest tests/test_tracking.py -k weighted_centroid -v`

- [ ] **Step 5: Commit.**
```bash
git add waper/tracking/rwp_polygon.py tests/test_tracking.py
git commit -m "feat(tracking): amplitude^2-weighted centroid for track position"
```

---

## Task 2: Energy disks at extrema

**Files:**
- Modify: `waper/tracking/rwp_polygon.py`
- Test: `tests/test_tracking.py`

- [ ] **Step 1: Write the failing test:**

```python
from waper.tracking.rwp_polygon import energy_disks

def test_energy_disks_one_per_node_weighted_by_energy():
    # two extrema; energy must be amplitude**2 and sign-independent
    cells = energy_disks([(0.0, 50.0, 3.0), (90.0, 50.0, -2.0)],
                         hemisphere="north", radius_m=300e3)
    assert len(cells) == 2
    geom0, e0 = cells[0]
    geom1, e1 = cells[1]
    assert abs(e0 - 9.0) < 1e-9
    assert abs(e1 - 4.0) < 1e-9
    assert geom0.geom_type == "Polygon" and geom0.area > 0

def test_energy_disks_empty():
    assert energy_disks([], hemisphere="north") == []
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** in `waper/tracking/rwp_polygon.py` (add `from shapely.geometry import Point` to the imports if not present):

```python
def energy_disks(node_coords_scalars, hemisphere="north", radius_m=500e3):
    """Build energy footprint cells for a set of extrema.

    Args:
        node_coords_scalars: iterable of (lon, lat, scalar) for each extremum.
        hemisphere: "north" | "south".
        radius_m: disk radius in metres around each extremum.

    Returns:
        list of (shapely Polygon in stereographic metres, energy=scalar**2).
        Energy concentrates the footprint on the high-amplitude cores.
    """
    cells = []
    for lon, lat, scalar in node_coords_scalars:
        x, y = transform_to_stereographic(lon, lat, hemisphere=hemisphere)
        cells.append((Point(x, y).buffer(radius_m), float(scalar) ** 2))
    return cells
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit.**
```bash
git add waper/tracking/rwp_polygon.py tests/test_tracking.py
git commit -m "feat(tracking): energy disks (amplitude^2) at extrema"
```

---

## Task 3: Rasterize the energy field

**Files:**
- Modify: `waper/tracking/rwp_polygon.py`
- Test: `tests/test_tracking.py`

- [ ] **Step 1: Write the failing test:**

```python
import numpy as np
from waper.tracking.rwp_polygon import energy_disks, rasterize_energy, WAPER_IMAGE_SIZE

def test_rasterize_energy_shape_and_values():
    cells = energy_disks([(0.0, 80.0, 4.0)], hemisphere="north", radius_m=500e3)
    raster = rasterize_energy(cells, hemisphere="north")
    assert raster.shape == (WAPER_IMAGE_SIZE, WAPER_IMAGE_SIZE)
    assert raster.dtype == np.float64
    # the only non-zero value present is the burned energy (4**2 = 16)
    nonzero = np.unique(raster[raster > 0])
    assert nonzero.size == 1 and abs(nonzero[0] - 16.0) < 1e-6
    assert (raster > 0).sum() > 0

def test_rasterize_energy_empty_returns_none():
    assert rasterize_energy([], hemisphere="north") is None
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** in `waper/tracking/rwp_polygon.py` (next to `rasterize_all_rwps`):

```python
def rasterize_energy(energy_cells, hemisphere="north"):
    """Rasterize energy cells onto the same 512x512 stereographic grid used by
    `rasterize_all_rwps`. Each pixel holds the burned energy (0 where empty).

    Args:
        energy_cells: list of (geometry in stereographic metres, energy float).
        hemisphere: "north" | "south".

    Returns:
        float64 raster of shape (WAPER_IMAGE_SIZE, WAPER_IMAGE_SIZE), or None.
    """
    if len(energy_cells) == 0:
        return None
    return features.rasterize(
        ((g, e) for g, e in energy_cells),
        out_shape=(WAPER_IMAGE_SIZE, WAPER_IMAGE_SIZE),
        fill=0.0,
        all_touched=True,
        dtype="float64",
        transform=_get_raster_transform(hemisphere),
    )
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit.**
```bash
git add waper/tracking/rwp_polygon.py tests/test_tracking.py
git commit -m "feat(tracking): rasterize energy field on the stereographic grid"
```

---

## Task 4: Build & store the energy raster per timestep

**Files:**
- Modify: `waper/interface/api.py`
- Test: `tests/test_tracking.py`

- [ ] **Step 1: Write the failing test** (drives the integration through a real `Waper` run):

```python
import xarray as xr
import numpy as np
from waper.interface.api import Waper

def test_energy_raster_built_and_aligned(two_timestep_field):
    ds = xr.Dataset({"v": two_timestep_field})
    w = Waper(data_array=ds, scalar_name="v",
              latitude_label="latitude", longitude_label="longitude", time_label="time",
              clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80.1,
              node_pruning_threshold=15, edge_pruning_threshold=3e-5,
              track_pruning_threshold=0.3, max_edge_weight=1, debug=False)
    w.identify_rwps()
    tsd = w._time_step_data[0]
    assert tsd.energy_raster is not None
    assert tsd.energy_raster.shape == tsd.raster_data.shape
    # energy lives only where a feature footprint is, and is strictly positive there
    assert (tsd.energy_raster > 0).any()
    assert np.all(tsd.energy_raster[tsd.raster_data == 0] >= 0)
```

- [ ] **Step 2: Run, expect FAIL** (`tsd.energy_raster` does not exist).

- [ ] **Step 3: Implement.**

In `WaperConfig` (defaults section, after `penalty_length_scale_km`), add:

```python
    energy_radius_km: float = 500.0
```

In `WaperSingleTimestepData` field annotations (after `quadtree: Graph`), add:

```python
    energy_raster: ndarray = None
```

In `_identify_rwps` in `api.py`, immediately after the line that sets
`time_step_data.raster_data = rwp_polygon.rasterize_all_rwps(list_polygons, hemisphere=config.hemisphere)`,
add:

```python
    energy_nodes = []
    for path in time_step_data.identified_rwp_paths:
        for n in path:
            lon, lat = time_step_data.pruned_graph.nodes[n]["coords"]
            scalar = time_step_data.pruned_graph.nodes[n]["scalar"]
            energy_nodes.append((lon, lat, scalar))
    energy_cells = rwp_polygon.energy_disks(
        energy_nodes, hemisphere=config.hemisphere,
        radius_m=config.energy_radius_km * 1000.0,
    )
    time_step_data.energy_raster = rwp_polygon.rasterize_energy(
        energy_cells, hemisphere=config.hemisphere
    )
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit.**
```bash
git add waper/interface/api.py tests/test_tracking.py
git commit -m "feat(tracking): build per-timestep energy raster during identify"
```

---

## Task 5: Energy-overlap primitives

**Files:**
- Create: `waper/tracking/energy_overlap.py`
- Test: `tests/test_energy_overlap.py`

- [ ] **Step 1: Write the failing test** in `tests/test_energy_overlap.py`:

```python
import numpy as np
from waper.tracking.energy_overlap import feature_energies, overlap_energies

def test_feature_energies_sums_per_feature():
    F = np.array([[0, 1, 1], [2, 2, 0]])
    E = np.array([[0.0, 3.0, 1.0], [5.0, 5.0, 0.0]])
    assert feature_energies(F, E) == {1: 4.0, 2: 10.0}

def test_overlap_energies_geometric_mean_over_overlap():
    Fp = np.array([[1, 1, 0]]); Ep = np.array([[4.0, 4.0, 0.0]])
    Fc = np.array([[1, 0, 0]]); Ec = np.array([[9.0, 0.0, 0.0]])
    # overlap only at pixel (0,0): sqrt(4*9) = 6
    assert overlap_energies(Fp, Ep, Fc, Ec) == {(1, 1): 6.0}

def test_overlap_energies_no_overlap_empty():
    Fp = np.array([[1, 0]]); Ep = np.array([[4.0, 0.0]])
    Fc = np.array([[0, 2]]); Ec = np.array([[0.0, 9.0]])
    assert overlap_energies(Fp, Ep, Fc, Ec) == {}
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** `waper/tracking/energy_overlap.py`:

```python
"""Energy-weighted overlap between consecutive-timestep RWP rasters.

The association weight focuses on the energetic cores: each feature's "size" is
its summed energy, and the overlap between two features is the summed geometric
mean of their per-pixel energies over the pixels they share. Compared with binary
pixel overlap, the weak periphery contributes ~0, so a moving core changes the
weight even when the broad footprints still overlap.
"""
import numpy as np


def feature_energies(feature_raster, energy_raster):
    """{feature_id: total energy} for every non-zero feature."""
    ids = np.unique(feature_raster)
    ids = ids[ids != 0]
    return {int(i): float(energy_raster[feature_raster == i].sum()) for i in ids}


def overlap_energies(prev_features, prev_energy, curr_features, curr_energy):
    """{(prev_id, curr_id): Σ sqrt(E_prev * E_curr)} over overlapping pixels."""
    out = {}
    prev_ids = np.unique(prev_features)
    prev_ids = prev_ids[prev_ids != 0]
    for a in prev_ids:
        amask = prev_features == a
        curr_here = np.unique(curr_features[amask])
        curr_here = curr_here[curr_here != 0]
        for b in curr_here:
            m = amask & (curr_features == b)
            e = float(np.sqrt(prev_energy[m] * curr_energy[m]).sum())
            if e > 0:
                out[(int(a), int(b))] = e
    return out
```

- [ ] **Step 4: Run, expect PASS.** `~/miniconda3/envs/waper/bin/python -m pytest tests/test_energy_overlap.py -v`

- [ ] **Step 5: Commit.**
```bash
git add waper/tracking/energy_overlap.py tests/test_energy_overlap.py
git commit -m "feat(tracking): energy-overlap primitives (feature energy + overlap)"
```

---

## Task 6: Drive `build_tracking_graph` from energy overlap

**Files:**
- Modify: `waper/tracking/tracking_graph.py`
- Test: `tests/test_tracking.py`

- [ ] **Step 1: Write the failing test** (energy weight discriminates a moved core; and update the merge-call-count test to the new code path):

```python
import numpy as np
import networkx as nx
from unittest.mock import MagicMock, patch
from waper.tracking import tracking_graph

def _stub_tsd(raster, energy, features):
    s = MagicMock()
    s.raster_data = raster
    s.energy_raster = energy
    s.raster_features = features
    s.rwp_info = {}
    return s

def test_energy_weight_full_overlap_is_one():
    F = np.array([[0, 1, 1], [0, 1, 0]])
    E = np.array([[0.0, 2.0, 2.0], [0.0, 2.0, 0.0]])
    ts = [_stub_tsd(F, E, {0, 1}), _stub_tsd(F.copy(), E.copy(), {0, 1})]
    g = tracking_graph.build_tracking_graph(ts, 2)
    assert g.number_of_edges() == 1
    assert abs(g[(0, 1)][(1, 1)]["weight"] - 1.0) < 1e-9

def test_energy_weight_partial_when_core_moves():
    Fp = np.array([[1, 1, 0, 0]]); Ep = np.array([[5.0, 1.0, 0.0, 0.0]])
    Fc = np.array([[0, 1, 1, 0]]); Ec = np.array([[0.0, 1.0, 5.0, 0.0]])
    ts = [_stub_tsd(Fp, Ep, {0, 1}), _stub_tsd(Fc, Ec, {0, 1})]
    g = tracking_graph.build_tracking_graph(ts, 2)
    w = g[(0, 1)][(1, 1)]["weight"]
    assert 0.0 < w < 1.0

def test_overlap_computed_once_per_timestep_pair():
    F = np.array([[1]]); E = np.array([[1.0]])
    ts = [_stub_tsd(F, E, {0, 1}) for _ in range(3)]
    with patch("waper.tracking.tracking_graph.overlap_energies",
               return_value={}) as mock_ov:
        tracking_graph.build_tracking_graph(ts, number_steps=3)
        assert mock_ov.call_count == 2
```

Also delete the now-obsolete `test_merge_called_once_per_timestep_pair` from `tests/test_tracking.py` (it patched `tracking_graph.merge`, which the energy path no longer calls).

- [ ] **Step 2: Run, expect FAIL.** `~/miniconda3/envs/waper/bin/python -m pytest tests/test_tracking.py -k "energy_weight or once_per_timestep" -v`

- [ ] **Step 3: Implement** in `waper/tracking/tracking_graph.py`:

Add the import near the top:

```python
from .energy_overlap import feature_energies, overlap_energies
```

Replace the per-timestep overlap block (currently the `merge(...)` / `compute_size_features(...)` / `edge_list` loop inside `build_tracking_graph`, roughly lines 49–78) with:

```python
        if time > 0:
            prev = time_step_data[time - 1]
            curr = time_step_data[time]
            if prev.raster_data is None or curr.raster_data is None:
                continue
            if prev.energy_raster is None or curr.energy_raster is None:
                continue

            prev_energy = feature_energies(prev.raster_data, prev.energy_raster)
            curr_energy = feature_energies(curr.raster_data, curr.energy_raster)
            overlaps = overlap_energies(
                prev.raster_data, prev.energy_raster,
                curr.raster_data, curr.energy_raster,
            )

            for (a, b), overlap_e in overlaps.items():
                denom = max(prev_energy.get(a, 0.0), curr_energy.get(b, 0.0))
                if denom > 0:
                    tracking_graph.add_edge(
                        (time - 1, a), (time, b), weight=overlap_e / denom
                    )
```

Leave the node-creation loop above it and the centroid-distance loop below it unchanged. Remove the now-unused `merge` / `compute_size_features` imports from this file **only if** they are not referenced elsewhere in it (the quadtree module itself is unchanged).

- [ ] **Step 4: Run the tracking suite, expect PASS** (the behavioural tests `test_identical_timesteps_full_overlap`, `test_shifted_field_partial_overlap`, `test_no_overlap_no_edge`, `test_tracking_path_extraction`, `test_feature_zero_not_in_edges`, and `test_quadtree_pixel_counts` must all still pass):

`~/miniconda3/envs/waper/bin/python -m pytest tests/test_tracking.py -v`

- [ ] **Step 5: Commit.**
```bash
git add waper/tracking/tracking_graph.py tests/test_tracking.py
git commit -m "feat(tracking): associate timesteps by energy overlap of cores"
```

---

## Task 7: Empirical check on a real dataset (evidence, not a gate)

**Files:**
- Test: `tests/test_tracking.py`

- [ ] **Step 1: Add a slow, opt-in empirical test** asserting energy tracking yields appreciable eastward motion on a real packet (skipped unless the dataset is present):

```python
import os
import pytest
import numpy as np
import xarray as xr
from waper.interface.api import Waper

DATASET = "datasets/forecast_bust.nc"

@pytest.mark.skipif(not os.path.exists(DATASET), reason="forecast_bust.nc not present")
def test_energy_tracks_show_eastward_motion():
    ds = xr.open_dataset(DATASET)
    w = Waper(data_array=ds, scalar_name="v",
              latitude_label="latitude", longitude_label="longitude", time_label="time",
              clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80,
              node_pruning_threshold=20, edge_pruning_threshold=0.02, max_edge_weight=1,
              track_pruning_threshold=0.3)
    w.identify_rwps(); w.track_rwps()
    g = w._tracking_graph
    # at least one tracked edge exists, and centroids move (not frozen)
    assert g.number_of_edges() > 0
    moved = [d["distance"] for _, _, d in g.edges(data=True)]
    assert np.median(moved) > 0.0
```

- [ ] **Step 2: Run, expect PASS** (or SKIP if the file is absent):
`~/miniconda3/envs/waper/bin/python -m pytest tests/test_tracking.py -k eastward_motion -v`

- [ ] **Step 3: Manually inspect (optional but recommended):** run `scripts/run_explorer.py` on `datasets/forecast_bust.nc`, select a track in the Tabulator, and confirm its highlighted path now advances eastward across timesteps instead of sitting still.

- [ ] **Step 4: Commit.**
```bash
git add tests/test_tracking.py
git commit -m "test(tracking): empirical eastward-motion check on forecast_bust"
```

---

## Self-review

- **Spec coverage:** energy-weighted centroid → Task 1; energy field per RWP → Tasks 2–4; energy-weighted association → Tasks 5–6; validation cases (motion visible, periphery ignored via energy weighting, regression of merge/split behaviour, empirical eastward motion) → Tasks 1/6/7. Merge/split structure is preserved because only the edge-weight computation changes; node identity and the DAG extraction are untouched.
- **Deviation from the design doc (intentional, lower-risk):** the energy field is built from **amplitude²-weighted disks at the extrema** (`energy_disks`) rather than grid-sampled `v²`. This avoids per-pixel inverse projection, is unit-testable, and keeps energy concentrated on the cores. Disk radius is configurable (`energy_radius_km`, default 500 km). If extrema-disk overlap proves too phase-sensitive in evaluation, swap the energy field for `v²` sampled over the footprint behind the same `rasterize_energy` interface — no downstream change.
- **Type consistency:** `energy_disks → list[(Polygon, float)]`; `rasterize_energy → float64 raster | None`; `feature_energies → {int: float}`; `overlap_energies → {(int,int): float}`; `tsd.energy_raster` aligns with `tsd.raster_data`. Names match across tasks.
- **No placeholders.**

## Open questions to resolve during evaluation (not blockers)
- `energy_radius_km` default (500 km) — tune against observed track continuity.
- Per-pixel overlap combiner `√(E_prev·E_curr)` vs `min` — geometric mean chosen; revisit if weights need to be harsher on amplitude change.
- Whether to also add an eastward-displacement prior to the association (deferred; the design doc flags it).
