# Feature-Track Layer (SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Design doc:** `superpowers/plans/design/feature_track_layer_plan.md` — read it for rationale.

**Goal:** Track each individual crest/trough (not the unstable RWP group) across time as a continuous trajectory, robust to a feature briefly weakening (recovery) and to the RWP grouping merging/splitting around it.

**Architecture:** A new read-only post-processing module `waper/tracking/feature_tracks.py` over a completed `identify_rwps()` run. Per timestep it extracts *every* extremum from `tsd.association_graph` (the unpruned pool) with a region-based footprint; then it matches strong features of the same type across consecutive steps by footprint overlap (one-to-one, greedy), recovering momentarily-weak features from the weak pool. Output is a list of `FeatureTrack`s. Identification and all existing tracking code are untouched.

**Tech Stack:** numpy, shapely, networkx, pyvista (only via existing helpers), matplotlib/cartopy (viz only) — all already deps.

## Global Constraints

- Run tests with the project env: `~/miniconda3/envs/waper/bin/python -m pytest …` from repo root `/Users/joymonteiro/github/waper`. Plain `pytest`/`python` are the wrong interpreter.
- Work on branch `energy-weighted-tracking`. New code only in `waper/tracking/feature_tracks.py`; tests in `tests/test_feature_tracks.py`; viz in `scripts/feature_tracks_gif.py`. **Do not modify identification, the association/pruned graph, or existing tracking code.**
- `tsd.association_graph` node keys are `("max"|"min", id)` tuples; node attrs: `coords=(lon,lat)`, `scalar` (float, signed), `node_type` (`"max"`/`"min"`), `cluster_id` (int), `spherical_coords`, `cluster_extrema`.
- Per-timestep scalar field is `tsd.vtk_data`. Clipped connected regions are built exactly as `get_polygon_for_rwp_path` does:
  `topology.identify_connected_regions(tsd.vtk_data.clip_scalar(scalars=name, value=+clip, invert=False).clean())` for maxima and `… value=-clip, invert=True …` for minima.
- Region points for one extremum: `waper.tracking.rwp_polygon.get_region_points_and_values(assoc_graph, node, clipped_region, clip_value, scalar_name)` → `(lons, lats, values)` or `None`.
- `waper.tracking.rwp_polygon.transform_to_stereographic(lons, lats, hemisphere="north")` → stereographic metres (accepts arrays).
- Footprints are shapely geometries in **stereographic metres**. A feature is `strong` iff `abs(scalar) >= amplitude_threshold` (absolute m/s), else `weak` — a label, never a filter.
- `git add` only the explicit files named in each commit step. **Never `git add -A`/`git add .`** (the repo has large untracked data files that must never be committed).

## File Structure

- `waper/tracking/feature_tracks.py` — `Feature`, `TrackStep`, `FeatureTrack` dataclasses; `feature_overlap`, `match_features`, `track_features`, `extract_features`, `feature_tracks_to_dataframe`, `phase_velocity`.
- `tests/test_feature_tracks.py` — unit tests (synthetic `Feature`s) + one integration test (small `Waper` run).
- `scripts/feature_tracks_gif.py` — full-hemisphere GIF of the tracks.

---

## Task 1: Feature model + footprint matching primitive

**Files:**
- Create: `waper/tracking/feature_tracks.py`
- Test: `tests/test_feature_tracks.py`

**Interfaces:**
- Produces: `Feature(time:int, cluster_id:int, node_type:str, lon:float, lat:float, scalar:float, footprint, strength:str)`; `feature_overlap(a:Feature, b:Feature)->float`; `match_features(prev:list[Feature], curr:list[Feature])->dict[int,int]` (prev-index → curr-index, one-to-one).

- [ ] **Step 1: Write the failing test** in `tests/test_feature_tracks.py`:

```python
from shapely.geometry import box
from waper.tracking.feature_tracks import Feature, feature_overlap, match_features


def _feat(t, ntype, x0, strength="strong", scalar=20.0):
    # footprint is a 10x10 stereographic box starting at x0
    return Feature(time=t, cluster_id=0, node_type=ntype, lon=0.0, lat=40.0,
                   scalar=scalar, footprint=box(x0, 0, x0 + 10, 10), strength=strength)


def test_overlap_same_type_area():
    a = _feat(0, "max", 0); b = _feat(1, "max", 5)
    assert feature_overlap(a, b) == 50.0          # 5 wide x 10 tall


def test_overlap_zero_for_different_type():
    a = _feat(0, "max", 0); b = _feat(1, "min", 0)   # identical box, different type
    assert feature_overlap(a, b) == 0.0


def test_match_is_one_to_one_greedy_by_overlap():
    prev = [_feat(0, "max", 0), _feat(0, "max", 100)]
    curr = [_feat(1, "max", 2), _feat(1, "max", 4)]   # both overlap prev[0]; prev[1] overlaps neither
    m = match_features(prev, curr)
    assert m == {0: 0}                                # prev[0] takes its best (curr[0], overlap 8>6); curr[1] free but prev exhausted
```

- [ ] **Step 2: Run, expect FAIL.** `~/miniconda3/envs/waper/bin/python -m pytest tests/test_feature_tracks.py -q` → ImportError.

- [ ] **Step 3: Implement** in `waper/tracking/feature_tracks.py`:

```python
"""Feature-track layer: track individual crests/troughs across time.

Read-only post-processing over a completed `identify_rwps()` run. The tracked
primitive is a single extremum (crest/trough), which moves continuously, rather
than the RWP group, whose membership flips between timesteps.
"""
from dataclasses import dataclass, field


@dataclass
class Feature:
    time: int
    cluster_id: int
    node_type: str          # "max" | "min"
    lon: float
    lat: float
    scalar: float           # signed amplitude
    footprint: object       # shapely geometry in stereographic metres
    strength: str           # "strong" | "weak"


def feature_overlap(a: Feature, b: Feature) -> float:
    """Footprint intersection area; 0 for different node_type."""
    if a.node_type != b.node_type:
        return 0.0
    return float(a.footprint.intersection(b.footprint).area)


def match_features(prev, curr) -> dict:
    """One-to-one assignment prev-index -> curr-index, greedy by descending
    footprint overlap. Only positive-overlap, same-type pairs are eligible."""
    scored = []
    for i, a in enumerate(prev):
        for j, b in enumerate(curr):
            ov = feature_overlap(a, b)
            if ov > 0.0:
                scored.append((ov, i, j))
    scored.sort(reverse=True)
    used_prev, used_curr, match = set(), set(), {}
    for _, i, j in scored:
        if i in used_prev or j in used_curr:
            continue
        match[i] = j
        used_prev.add(i)
        used_curr.add(j)
    return match
```

- [ ] **Step 4: Run, expect PASS.** `~/miniconda3/envs/waper/bin/python -m pytest tests/test_feature_tracks.py -q`

- [ ] **Step 5: Commit.**
```bash
git add waper/tracking/feature_tracks.py tests/test_feature_tracks.py
git commit -m "feat(feature-tracks): Feature model + footprint overlap matching"
```

---

## Task 2: track_features — births, extension, deaths (no recovery yet)

**Files:**
- Modify: `waper/tracking/feature_tracks.py`
- Test: `tests/test_feature_tracks.py`

**Interfaces:**
- Consumes: `Feature`, `match_features` (Task 1).
- Produces: `TrackStep(time:int, lon:float, lat:float, scalar:float, node_type:str, recovered:bool)`; `FeatureTrack(track_id:int, steps:list[TrackStep])`; `track_features(features_by_time:list[list[Feature]])->list[FeatureTrack]`.

- [ ] **Step 1: Write the failing test:**

```python
from waper.tracking.feature_tracks import track_features, FeatureTrack


def test_shifted_feature_becomes_one_track():
    fb = [[_feat(0, "max", 0)], [_feat(1, "max", 4)], [_feat(2, "max", 8)]]  # overlaps step to step
    tracks = track_features(fb)
    assert len(tracks) == 1
    assert [s.time for s in tracks[0].steps] == [0, 1, 2]


def test_unmatched_curr_feature_is_a_birth():
    fb = [[_feat(0, "max", 0)], [_feat(1, "max", 4), _feat(1, "max", 100)]]
    tracks = track_features(fb)
    assert len(tracks) == 2                       # the continued one + the newborn
    assert max(len(t.steps) for t in tracks) == 2
    assert min(len(t.steps) for t in tracks) == 1


def test_unmatched_prev_feature_dies():
    fb = [[_feat(0, "max", 0)], [_feat(1, "max", 100)]]   # no overlap -> no match
    tracks = track_features(fb)
    assert len(tracks) == 2
    assert all(len(t.steps) == 1 for t in tracks)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** (append to `waper/tracking/feature_tracks.py`):

```python
@dataclass
class TrackStep:
    time: int
    lon: float
    lat: float
    scalar: float
    node_type: str
    recovered: bool


@dataclass
class FeatureTrack:
    track_id: int
    steps: list = field(default_factory=list)


def _step(feature: Feature, recovered: bool) -> TrackStep:
    return TrackStep(time=feature.time, lon=feature.lon, lat=feature.lat,
                     scalar=feature.scalar, node_type=feature.node_type,
                     recovered=recovered)


def _strong(features):
    return [f for f in features if f.strength == "strong"]


def track_features(features_by_time) -> list:
    """Build continuous feature tracks across timesteps. A track is seeded from a
    strong feature and extended each step to the maximally-overlapping strong
    feature of the same type; unmatched strong features at t are deaths, unmatched
    strong features at t+1 are births."""
    tracks = []
    # active = list of [track, head_feature]; seed from the first non-empty step
    active = []
    next_id = 0
    if features_by_time:
        for f in _strong(features_by_time[0]):
            tr = FeatureTrack(next_id, [_step(f, recovered=False)]); next_id += 1
            tracks.append(tr); active.append([tr, f])

    for t in range(1, len(features_by_time)):
        curr_strong = _strong(features_by_time[t])
        heads = [a[1] for a in active]
        match = match_features(heads, curr_strong)
        new_active = []
        matched_curr = set(match.values())
        for hi, a in enumerate(active):
            if hi in match:
                f = curr_strong[match[hi]]
                a[0].steps.append(_step(f, recovered=False))
                new_active.append([a[0], f])
            # else: track dies (dropped from active)
        # births: strong curr features not matched to any head
        for j, f in enumerate(curr_strong):
            if j not in matched_curr:
                tr = FeatureTrack(next_id, [_step(f, recovered=False)]); next_id += 1
                tracks.append(tr); new_active.append([tr, f])
        active = new_active

    return tracks
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit.**
```bash
git add waper/tracking/feature_tracks.py tests/test_feature_tracks.py
git commit -m "feat(feature-tracks): track_features with births/extension/deaths"
```

---

## Task 3: Recovery from the weak pool + latitude-band termination

**Files:**
- Modify: `waper/tracking/feature_tracks.py`
- Test: `tests/test_feature_tracks.py`

**Interfaces:**
- Modifies: `track_features(features_by_time, max_recover_steps:int=2, lat_bounds:tuple|None=None)->list[FeatureTrack]` (same return type; two new optional params).

- [ ] **Step 1: Write the failing test:**

```python
def test_feature_recovered_through_weak_step():
    # strong at t0, only a WEAK overlapping feature at t1, strong again at t2
    fb = [
        [_feat(0, "max", 0)],
        [_feat(1, "max", 4, strength="weak")],
        [_feat(2, "max", 8)],
    ]
    tracks = track_features(fb, max_recover_steps=2)
    assert len(tracks) == 1
    steps = tracks[0].steps
    assert [s.time for s in steps] == [0, 1, 2]
    assert steps[1].recovered is True and steps[2].recovered is False


def test_track_dies_after_recovery_budget_exhausted():
    fb = [
        [_feat(0, "max", 0)],
        [_feat(1, "max", 4, strength="weak")],
        [_feat(2, "max", 8, strength="weak")],
        [_feat(3, "max", 12, strength="weak")],   # 3rd consecutive weak > budget(2)
    ]
    tracks = track_features(fb, max_recover_steps=2)
    assert len(tracks) == 1
    assert [s.time for s in tracks[0].steps] == [0, 1, 2]   # terminated before t3


def test_track_ends_when_feature_leaves_lat_band():
    f0 = _feat(0, "max", 0); f0.lat = 60.0
    f1 = _feat(1, "max", 4); f1.lat = 85.0          # outside band
    tracks = track_features([[f0], [f1]], lat_bounds=(20.0, 80.0))
    assert len(tracks) == 1 and len(tracks[0].steps) == 1
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — replace the whole `track_features` function with this version:

```python
def _weak(features):
    return [f for f in features if f.strength == "weak"]


def _in_band(feature: Feature, lat_bounds) -> bool:
    if lat_bounds is None:
        return True
    lo, hi = lat_bounds
    return lo <= feature.lat <= hi


def track_features(features_by_time, max_recover_steps: int = 2,
                   lat_bounds=None) -> list:
    """Build continuous feature tracks across timesteps.

    A track is seeded from a strong feature and extended each step to the
    maximally-overlapping strong feature of the same type. If no strong match
    exists, the track may be continued through an overlapping *weak* feature
    (recovery), flagged on that step; it terminates after `max_recover_steps`
    consecutive recovered steps, or when its head leaves `lat_bounds`
    (``(min_lat, max_lat)`` or ``None``).
    """
    tracks = []
    active = []  # list of [track, head_feature, weak_streak]
    next_id = 0
    if features_by_time:
        for f in _strong(features_by_time[0]):
            tr = FeatureTrack(next_id, [_step(f, recovered=False)]); next_id += 1
            tracks.append(tr); active.append([tr, f, 0])

    for t in range(1, len(features_by_time)):
        curr_strong = _strong(features_by_time[t])
        curr_weak = _weak(features_by_time[t])
        heads = [a[1] for a in active]

        strong_match = match_features(heads, curr_strong)
        unmatched_heads = [hi for hi in range(len(active)) if hi not in strong_match]
        weak_match = match_features([active[hi][1] for hi in unmatched_heads], curr_weak)
        # remap weak_match local indices back to head indices
        weak_match = {unmatched_heads[k]: v for k, v in weak_match.items()}

        new_active = []
        for hi, a in enumerate(active):
            if hi in strong_match:
                f = curr_strong[strong_match[hi]]
                if not _in_band(f, lat_bounds):
                    continue                       # leaves band -> terminate
                a[0].steps.append(_step(f, recovered=False))
                new_active.append([a[0], f, 0])
            elif hi in weak_match:
                f = curr_weak[weak_match[hi]]
                if not _in_band(f, lat_bounds) or a[2] + 1 > max_recover_steps:
                    continue                       # band exit or budget exhausted -> terminate
                a[0].steps.append(_step(f, recovered=True))
                new_active.append([a[0], f, a[2] + 1])
            # else: no match at all -> terminate

        matched_curr = set(strong_match.values())
        for j, f in enumerate(curr_strong):
            if j not in matched_curr:
                if not _in_band(f, lat_bounds):
                    continue
                tr = FeatureTrack(next_id, [_step(f, recovered=False)]); next_id += 1
                tracks.append(tr); new_active.append([tr, f, 0])
        active = new_active

    return tracks
```

- [ ] **Step 4: Run, expect PASS.** Run the whole file: `~/miniconda3/envs/waper/bin/python -m pytest tests/test_feature_tracks.py -q`

- [ ] **Step 5: Commit.**
```bash
git add waper/tracking/feature_tracks.py tests/test_feature_tracks.py
git commit -m "feat(feature-tracks): weak-pool recovery + latitude-band termination"
```

---

## Task 4: Output — flat table + phase velocity

**Files:**
- Modify: `waper/tracking/feature_tracks.py`
- Test: `tests/test_feature_tracks.py`

**Interfaces:**
- Consumes: `FeatureTrack`, `TrackStep`.
- Produces: `feature_tracks_to_dataframe(tracks)->pandas.DataFrame` (cols `track_id,time,lon,lat,scalar,node_type,recovered`); `phase_velocity(track, dt_hours)->float` (mean eastward deg/hour along the track, wraparound-safe).

- [ ] **Step 1: Write the failing test:**

```python
import pandas as pd
from waper.tracking.feature_tracks import feature_tracks_to_dataframe, phase_velocity


def test_dataframe_has_one_row_per_step():
    fb = [[_feat(0, "max", 0)], [_feat(1, "max", 4)]]
    df = feature_tracks_to_dataframe(track_features(fb))
    assert set(df.columns) >= {"track_id", "time", "lon", "lat", "scalar", "node_type", "recovered"}
    assert len(df) == 2


def test_phase_velocity_eastward_degrees_per_hour():
    f0 = _feat(0, "max", 0); f0.lon = 10.0
    f1 = _feat(1, "max", 4); f1.lon = 16.0     # +6 deg over 6 h -> 1.0 deg/h east
    (track,) = track_features([[f0], [f1]])
    assert abs(phase_velocity(track, dt_hours=6.0) - 1.0) < 1e-9


def test_phase_velocity_handles_dateline():
    f0 = _feat(0, "max", 0); f0.lon = 179.0
    f1 = _feat(1, "max", 4); f1.lon = -179.0   # +2 deg east across dateline
    (track,) = track_features([[f0], [f1]])
    assert abs(phase_velocity(track, dt_hours=2.0) - 1.0) < 1e-9
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** (append to `waper/tracking/feature_tracks.py`):

```python
import numpy as np
import pandas as pd


def feature_tracks_to_dataframe(tracks) -> pd.DataFrame:
    rows = []
    for tr in tracks:
        for s in tr.steps:
            rows.append(dict(track_id=tr.track_id, time=s.time, lon=s.lon, lat=s.lat,
                             scalar=s.scalar, node_type=s.node_type, recovered=s.recovered))
    return pd.DataFrame(rows, columns=["track_id", "time", "lon", "lat", "scalar",
                                       "node_type", "recovered"])


def phase_velocity(track, dt_hours: float) -> float:
    """Mean eastward propagation in degrees/hour along the track (wraparound-safe).
    Returns nan for a single-step track."""
    steps = track.steps
    if len(steps) < 2:
        return float("nan")
    east = 0.0
    for a, b in zip(steps[:-1], steps[1:]):
        east += ((b.lon - a.lon + 180.0) % 360.0) - 180.0
    span_hours = (steps[-1].time - steps[0].time) * dt_hours
    return east / span_hours if span_hours else float("nan")
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit.**
```bash
git add waper/tracking/feature_tracks.py tests/test_feature_tracks.py
git commit -m "feat(feature-tracks): flat-table export + phase velocity"
```

---

## Task 5: extract_features — region footprints for all extrema

**Files:**
- Modify: `waper/tracking/feature_tracks.py`
- Test: `tests/test_feature_tracks.py`

**Interfaces:**
- Consumes: `Feature` (Task 1); WAPER region helpers (Global Constraints).
- Produces: `extract_features(tsd, time:int, scalar_name:str, clip_value:float, amplitude_threshold:float, hemisphere:str="north")->list[Feature]`.

- [ ] **Step 1: Write the failing test** (integration — drives a small real run via the top-level `two_timestep_field` fixture):

```python
import xarray as xr
from waper.interface.api import Waper
from waper.tracking.feature_tracks import extract_features, Feature


def test_extract_features_from_real_timestep(two_timestep_field):
    ds = xr.Dataset({"v": two_timestep_field})
    w = Waper(data_array=ds, scalar_name="v",
              latitude_label="latitude", longitude_label="longitude", time_label="time",
              clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80.1,
              node_pruning_threshold=15, edge_pruning_threshold=3e-5,
              track_pruning_threshold=0.3, max_edge_weight=1, debug=False)
    w.identify_rwps()
    feats = extract_features(w._time_step_data[0], time=0, scalar_name="v",
                             clip_value=2, amplitude_threshold=10)
    assert len(feats) > 0
    assert all(isinstance(f, Feature) for f in feats)
    assert all(f.node_type in ("max", "min") for f in feats)
    assert all(f.strength in ("strong", "weak") for f in feats)
    assert all(f.footprint.area > 0 for f in feats)        # real region hulls, not points
    # there are at least as many features as pruned RWP nodes (nothing was dropped)
    n_pruned_nodes = sum(len(p) for p in w._time_step_data[0].identified_rwp_paths)
    assert len(feats) >= n_pruned_nodes
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** (append to `waper/tracking/feature_tracks.py`; add imports at the top of the file):

```python
from shapely.geometry import MultiPoint
from ..identification import topology
from .rwp_polygon import get_region_points_and_values, transform_to_stereographic


def _footprint_from_region(lons, lats, hemisphere):
    xs, ys = transform_to_stereographic(np.asarray(lons), np.asarray(lats),
                                        hemisphere=hemisphere)
    pts = list(zip(np.atleast_1d(xs), np.atleast_1d(ys)))
    geom = MultiPoint(pts)
    return geom.convex_hull if len(pts) >= 3 else geom.buffer(1e4)


def extract_features(tsd, time, scalar_name, clip_value, amplitude_threshold,
                     hemisphere="north"):
    """All extrema of one timestep as Features, footprint = convex hull of the
    extremum's connected-region sample points at a single global `clip_value`."""
    scalar_data = tsd.vtk_data
    g = tsd.association_graph
    max_region = topology.identify_connected_regions(
        scalar_data.clip_scalar(scalars=scalar_name, value=clip_value, invert=False).clean())
    min_region = topology.identify_connected_regions(
        scalar_data.clip_scalar(scalars=scalar_name, value=-clip_value, invert=True).clean())

    features = []
    for node in g.nodes():
        attrs = g.nodes[node]
        region = max_region if attrs["node_type"] == "max" else min_region
        out = get_region_points_and_values(g, node, region, clip_value, scalar_name)
        if out is None:
            continue
        lons, lats, _ = out
        if len(np.atleast_1d(lons)) == 0:
            continue
        lon, lat = attrs["coords"]
        scalar = float(attrs["scalar"])
        features.append(Feature(
            time=time, cluster_id=int(attrs["cluster_id"]), node_type=attrs["node_type"],
            lon=float(lon), lat=float(lat), scalar=scalar,
            footprint=_footprint_from_region(lons, lats, hemisphere),
            strength=("strong" if abs(scalar) >= amplitude_threshold else "weak"),
        ))
    return features
```

- [ ] **Step 4: Run, expect PASS.** `~/miniconda3/envs/waper/bin/python -m pytest tests/test_feature_tracks.py -q`

- [ ] **Step 5: Commit.**
```bash
git add waper/tracking/feature_tracks.py tests/test_feature_tracks.py
git commit -m "feat(feature-tracks): extract_features with region-hull footprints"
```

---

## Task 6: Visualization GIF + empirical continuity check

**Files:**
- Create: `scripts/feature_tracks_gif.py`
- Test: `tests/test_feature_tracks.py`

**Interfaces:**
- Consumes: `extract_features`, `track_features`, `feature_tracks_to_dataframe` (Tasks 3–5).

- [ ] **Step 1: Write the failing empirical test** (opt-in; skipped if the dataset is absent):

```python
import os
import pytest


DATASET = "datasets/forecast_bust.nc"


@pytest.mark.skipif(not os.path.exists(DATASET), reason="forecast_bust.nc not present")
def test_feature_tracks_are_continuous_on_real_data():
    import numpy as np, xarray as xr
    from waper.interface.api import Waper
    from waper.tracking.feature_tracks import extract_features, track_features

    ds = xr.open_dataset(DATASET)
    av = np.abs(ds["v"].values).ravel()
    thr = float(np.percentile(av, 90))
    w = Waper(data_array=ds, scalar_name="v",
              latitude_label="latitude", longitude_label="longitude", time_label="time",
              clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80,
              node_pruning_threshold=20, edge_pruning_threshold=0.02, max_edge_weight=1,
              track_pruning_threshold=0.3)
    w.identify_rwps()
    fb = [extract_features(w._time_step_data[t], t, "v", 2, thr)
          for t in range(ds.sizes["time"])]
    tracks = track_features(fb, max_recover_steps=2, lat_bounds=(20.0, 80.0))
    # at least one feature is tracked across several steps (not all singletons)
    assert max(len(t.steps) for t in tracks) >= 5
```

- [ ] **Step 2: Run, expect PASS** (or SKIP). `~/miniconda3/envs/waper/bin/python -m pytest tests/test_feature_tracks.py -k continuous -v`

- [ ] **Step 3: Implement** `scripts/feature_tracks_gif.py` (full-hemisphere PlateCarree GIF, one colour per track):

```python
"""GIF of feature tracks over forecast_bust: each tracked crest/trough is one
coloured trajectory, so continuity (and whether neighbours move together) is
visible. Run: ~/miniconda3/envs/waper/bin/python scripts/feature_tracks_gif.py
"""
import glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from PIL import Image
from waper.interface.api import Waper
from waper.tracking.feature_tracks import extract_features, track_features
from waper.interface.colormaps import joy_nl8

ds = xr.open_dataset("datasets/forecast_bust.nc")
av = np.abs(ds["v"].values).ravel(); vmax = float(np.percentile(av, 99))
thr = float(np.percentile(av, 90))
w = Waper(data_array=ds, scalar_name="v", latitude_label="latitude",
          longitude_label="longitude", time_label="time", clip_value=2,
          extrema_threshold=10, min_latitude=20, max_latitude=80,
          node_pruning_threshold=20, edge_pruning_threshold=0.02, max_edge_weight=1,
          track_pruning_threshold=0.3)
w.identify_rwps()
nt = ds.sizes["time"]
fb = [extract_features(w._time_step_data[t], t, "v", 2, thr) for t in range(nt)]
tracks = track_features(fb, max_recover_steps=2, lat_bounds=(20.0, 80.0))

cmap = plt.cm.tab20
colors = {tr.track_id: cmap(tr.track_id % 20) for tr in tracks}
proj = ccrs.PlateCarree(central_longitude=180); pc = ccrs.PlateCarree()
frames = []
for t in range(nt):
    fig = plt.figure(figsize=(12, 6)); ax = plt.axes(projection=proj)
    ax.set_extent([-180, 180, 12, 88], crs=pc)
    cf = ds["v"].isel(time=t).plot.contourf(ax=ax, transform=pc, levels=15, cmap=joy_nl8,
                                            vmin=-vmax, vmax=vmax, add_colorbar=False)
    ax.coastlines(linewidth=0.5, color="0.4")
    fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.6, aspect=40, pad=0.07
                 ).set_label("v (m s$^{-1}$)")
    for tr in tracks:
        pts = [(s.lon, s.lat) for s in tr.steps if s.time <= t]
        if len(pts) >= 2:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], transform=ccrs.Geodetic(),
                    color=colors[tr.track_id], linewidth=1.6, zorder=5)
        here = [s for s in tr.steps if s.time == t]
        for s in here:
            ax.plot(s.lon, s.lat, transform=pc, marker=("s" if s.recovered else "o"),
                    markersize=6, markerfacecolor=colors[tr.track_id],
                    markeredgecolor="k", markeredgewidth=0.4, zorder=7)
    ax.set_title(f"feature tracks  t={t}/{nt-1}  (square = recovered step)", fontsize=10)
    fn = f"/tmp/feature_tracks_{t:02d}.png"; fig.savefig(fn, dpi=110, bbox_inches="tight")
    plt.close(fig); frames.append(fn)

imgs = [Image.open(f).convert("RGB") for f in sorted(glob.glob("/tmp/feature_tracks_*.png"))]
imgs[0].save("/tmp/feature_tracks.gif", save_all=True, append_images=imgs[1:], duration=550, loop=0)
print("wrote /tmp/feature_tracks.gif")
```

- [ ] **Step 4: Run the script to produce the GIF** (manual review aid; not a gated test):
`~/miniconda3/envs/waper/bin/python scripts/feature_tracks_gif.py` → `/tmp/feature_tracks.gif`.

- [ ] **Step 5: Commit.**
```bash
git add scripts/feature_tracks_gif.py tests/test_feature_tracks.py
git commit -m "feat(feature-tracks): hemisphere GIF + empirical continuity test"
```

---

## Self-Review

- **Spec coverage:** extraction with region-hull footprints at a global clip → Task 5; strong/weak label → Tasks 1/5; same-type max-overlap one-to-one matching → Tasks 1–2; recovery from weak pool + max_recover_steps + lat-band termination → Task 3; FeatureTrack output + phase velocity → Tasks 2/4; GIF + empirical continuity → Task 6. Identification untouched (no task modifies it). Group velocity / RWP identity correctly absent (Layer 3, deferred).
- **Type consistency:** `Feature`/`TrackStep`/`FeatureTrack` field names are identical across Tasks 1–6; `track_features` final signature `(features_by_time, max_recover_steps=2, lat_bounds=None)` is the one used in Tasks 4 and 6; `feature_overlap`/`match_features`/`extract_features`/`feature_tracks_to_dataframe`/`phase_velocity` names match between definition and use.
- **No placeholders.** Every code step is complete.
- **Deferred (not blockers):** amplitude threshold value (driver passes a percentile/absolute — sweepable); tie-break when several same-type extrema share one connected region at the global clip (identical footprints) — start with greedy-by-overlap; revisit from the GIF.
