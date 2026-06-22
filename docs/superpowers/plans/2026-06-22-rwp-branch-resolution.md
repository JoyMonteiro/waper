# RWP Branch Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `get_ranked_paths` from emitting length-1 RWPs and spatially-overlapping in-band RWPs, by skipping `source == sink`, rejecting in-band zonally-interleaving paths during selection, and reassigning leftover (orphan) features to the stronger branch under a latitude gate.

**Architecture:** All changes are in `waper/identification/rwp_graph.py` plus a config field in `waper/interface/api.py`. `get_ranked_paths` gains a `lat_gate` parameter; it does pass-1 selection (with `source==sink` skip + in-band-interleave rejection) then calls a new `reassign_orphans` helper that absorbs/drops orphans by branch resolution. The return type (list of node-list paths) is unchanged, so downstream code is untouched.

**Tech Stack:** Python 3.12, networkx, numpy, pytest.

## Global Constraints

- An RWP is and remains a **simple, monotonic-east, sign-alternating path**; no node in two RWPs; **no RWP of length 1**.
- **Latitude gate default = 15.0 degrees**, exposed as `WaperConfig.lat_gate` and `Waper.__init__(..., lat_gate=15.0)`.
- **In-band interleave** = longitude arcs overlap AND latitude ranges within `lat_gate`. Different-waveguide packets (latitude ranges > `lat_gate` apart) may share longitudes and must be kept.
- **Branch strength = summed edge weight** of the sub-path; a lone orphan's arm is its single connecting edge.
- Edges connect a max cluster to a min cluster only; node attrs are `coords=(lon, lat)`, `node_type in {"max","min"}`, `scalar`; edges carry `weight`.
- Existing module helpers available in `rwp_graph.py`: `is_to_the_east(lon1, lon2)`, `_longitude_separation(lon1, lon2)`, `_is_monotonic_east`, `_unwrap_path`; `numpy as np` and `networkx as nx` are imported.
- Run tests with `pytest <path> -q` — env is pre-activated; do NOT use `conda run`. If `pytest` is not on PATH use `~/miniconda3/envs/waper/bin/pytest`.
- Acceptance dataset: `datasets/forecast_bust_hourly.nc`, coarsened to 1° as in `scripts/feature_tracks_gif.py`, timestep `t = 95` (2011-04-04 23Z).

---

## File Structure

- `waper/identification/rwp_graph.py` — add `_path_lon_span`, `_arc_bins`, `_arcs_overlap`, `_path_lat_range`, `_lat_ranges_within`, `_paths_interleave_in_band`, `reassign_orphans`; modify `get_ranked_paths`.
- `waper/interface/api.py` — add `lat_gate: float = 15.0` to `WaperConfig`; add `lat_gate=15.0` kwarg to `Waper.__init__` and forward it; pass `config.lat_gate` at the `get_ranked_paths` call site.
- `tests/test_rwp_branch_resolution.py` — new unit + acceptance tests.

---

### Task 1: Skip `source == sink` (remove length-1 paths)

**Files:**
- Modify: `waper/identification/rwp_graph.py` (`get_ranked_paths`, the `for source/for sink` loop ~line 383)
- Test: `tests/test_rwp_branch_resolution.py`

**Interfaces:**
- Consumes: existing `get_ranked_paths(assoc_graph, max_weight)`.
- Produces: same signature; never returns a length-1 path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rwp_branch_resolution.py
import networkx as nx
from waper.identification.rwp_graph import get_ranked_paths


def _g():
    """A→B edge (a valid 2-node RWP) plus an isolated node C."""
    g = nx.Graph()
    g.add_node(("max", 0), coords=(10.0, 50.0), node_type="max", scalar=30.0)
    g.add_node(("min", 0), coords=(30.0, 50.0), node_type="min", scalar=-30.0)
    g.add_node(("max", 1), coords=(200.0, 50.0), node_type="max", scalar=25.0)  # isolated
    g.add_edge(("max", 0), ("min", 0), weight=1.0)
    return g


def test_no_length_one_paths():
    paths = get_ranked_paths(_g(), max_weight=10.0)
    assert all(len(p) >= 2 for p in paths)
    # the isolated node must not appear as its own RWP
    assert not any(p == [("max", 1)] for p in paths)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rwp_branch_resolution.py::test_no_length_one_paths -q`
Expected: FAIL — a length-1 path `[('max', 1)]` is returned (and possibly other singletons).

- [ ] **Step 3: Implement the skip**

In `get_ranked_paths`, inside the `for sink in end_leaves:` loop, add a guard as the first statement:

```python
    for source in start_leaves:
        for sink in end_leaves:
            if source == sink:
                continue
            # eliminate sinks to the west of source node
            if is_to_the_east(
                assoc_graph.nodes[source]["coords"][0], assoc_graph.nodes[sink]["coords"][0]
            ):
                continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rwp_branch_resolution.py::test_no_length_one_paths -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add waper/identification/rwp_graph.py tests/test_rwp_branch_resolution.py
git commit -m "fix(identification): skip source==sink in get_ranked_paths (no length-1 RWPs)"
```

---

### Task 2: Longitude/latitude span + interleave helpers

**Files:**
- Modify: `waper/identification/rwp_graph.py` (add module-level helpers, e.g. after `_unwrap_path`)
- Test: `tests/test_rwp_branch_resolution.py` (append)

**Interfaces:**
- Consumes: `_longitude_separation` (existing).
- Produces:
  - `_path_lon_span(assoc_graph, path) -> tuple[float, float]` — `(start_lon, arc_length_deg)`.
  - `_arc_bins(start, length, full=360.0, step=1.0) -> set[int]` — integer-degree bins an eastward arc covers.
  - `_arcs_overlap(start_a, len_a, start_b, len_b) -> bool`.
  - `_path_lat_range(assoc_graph, path) -> tuple[float, float]` — `(min_lat, max_lat)`.
  - `_lat_ranges_within(range_a, range_b, gate) -> bool`.
  - `_paths_interleave_in_band(assoc_graph, path_a, path_b, lat_gate) -> bool`.

- [ ] **Step 1: Write the failing tests (append)**

```python
# tests/test_rwp_branch_resolution.py  (append)
from waper.identification.rwp_graph import (
    _path_lon_span, _arcs_overlap, _path_lat_range,
    _lat_ranges_within, _paths_interleave_in_band,
)


def _node(g, name, lon, lat):
    g.add_node(name, coords=(lon, lat), node_type=name[0], scalar=10.0)


def test_lon_span_and_lat_range():
    g = nx.Graph()
    _node(g, ("max", 0), 10.0, 40.0)
    _node(g, ("min", 0), 30.0, 55.0)
    _node(g, ("max", 1), 60.0, 45.0)
    path = [("max", 0), ("min", 0), ("max", 1)]
    start, length = _path_lon_span(g, path)
    assert start == 10.0
    assert abs(length - 50.0) < 1e-6          # 20 + 30 degrees eastward
    assert _path_lat_range(g, path) == (40.0, 55.0)


def test_arcs_overlap():
    assert _arcs_overlap(10.0, 40.0, 30.0, 40.0) is True     # [10,50] vs [30,70]
    assert _arcs_overlap(10.0, 20.0, 100.0, 20.0) is False   # [10,30] vs [100,120]
    assert _arcs_overlap(350.0, 30.0, 10.0, 10.0) is True    # wrap: [350,20] vs [10,20]


def test_lat_ranges_within():
    assert _lat_ranges_within((40, 55), (45, 60), 15) is True    # overlapping
    assert _lat_ranges_within((30, 35), (45, 50), 15) is True    # gap 10 <= 15
    assert _lat_ranges_within((20, 25), (45, 50), 15) is False   # gap 20 > 15


def test_paths_interleave_in_band():
    g = nx.Graph()
    _node(g, ("max", 0), 10.0, 50.0); _node(g, ("min", 0), 40.0, 52.0)   # band A 50-52
    _node(g, ("max", 1), 20.0, 51.0); _node(g, ("min", 1), 50.0, 53.0)   # overlaps lon, same band
    _node(g, ("max", 2), 20.0, 25.0); _node(g, ("min", 2), 50.0, 27.0)   # overlaps lon, far band
    a = [("max", 0), ("min", 0)]
    b = [("max", 1), ("min", 1)]
    c = [("max", 2), ("min", 2)]
    assert _paths_interleave_in_band(g, a, b, 15.0) is True    # same band, overlapping lon
    assert _paths_interleave_in_band(g, a, c, 15.0) is False   # far band -> allowed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rwp_branch_resolution.py -q -k "span or arcs or lat_ranges or interleave"`
Expected: FAIL — `ImportError: cannot import name '_path_lon_span'`.

- [ ] **Step 3: Implement the helpers (append to `rwp_graph.py`)**

```python
def _path_lon_span(assoc_graph, path):
    """Eastward longitude arc of a monotonic-east path: (start_lon, arc_length_deg)."""
    start = assoc_graph.nodes[path[0]]["coords"][0]
    length = 0.0
    for i in range(len(path) - 1):
        a = assoc_graph.nodes[path[i]]["coords"][0]
        b = assoc_graph.nodes[path[i + 1]]["coords"][0]
        length += _longitude_separation(a, b)
    return start, length


def _arc_bins(start, length, full=360.0, step=1.0):
    """Integer-degree bins covered by the eastward arc [start, start+length] (mod 360)."""
    n = int(length // step) + 1
    return {int(round((start + k * step) % full)) % 360 for k in range(n + 1)}


def _arcs_overlap(start_a, len_a, start_b, len_b):
    """True if two eastward longitude arcs share any longitude (wrap-aware)."""
    return not _arc_bins(start_a, len_a).isdisjoint(_arc_bins(start_b, len_b))


def _path_lat_range(assoc_graph, path):
    lats = [assoc_graph.nodes[n]["coords"][1] for n in path]
    return min(lats), max(lats)


def _lat_ranges_within(range_a, range_b, gate):
    """True if the gap between two [min,max] latitude ranges is <= gate (overlap -> 0)."""
    lo = max(range_a[0], range_b[0])
    hi = min(range_a[1], range_b[1])
    if lo <= hi:
        return True
    return (lo - hi) <= gate


def _paths_interleave_in_band(assoc_graph, path_a, path_b, lat_gate):
    """True if two paths overlap in longitude AND lie within lat_gate of each other."""
    sa, la = _path_lon_span(assoc_graph, path_a)
    sb, lb = _path_lon_span(assoc_graph, path_b)
    if not _arcs_overlap(sa, la, sb, lb):
        return False
    return _lat_ranges_within(
        _path_lat_range(assoc_graph, path_a),
        _path_lat_range(assoc_graph, path_b),
        lat_gate,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rwp_branch_resolution.py -q -k "span or arcs or lat_ranges or interleave"`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add waper/identification/rwp_graph.py tests/test_rwp_branch_resolution.py
git commit -m "feat(identification): longitude/latitude span + in-band interleave helpers"
```

---

### Task 3: Pass-1 in-band zonal exclusivity

**Files:**
- Modify: `waper/identification/rwp_graph.py` (`get_ranked_paths` greedy-selection loop + signature)
- Test: `tests/test_rwp_branch_resolution.py` (append)

**Interfaces:**
- Consumes: `_paths_interleave_in_band` (Task 2).
- Produces: `get_ranked_paths(assoc_graph, max_weight, lat_gate=15.0)` — rejects a candidate path that interleaves an already-accepted path within `lat_gate`. (Pass-2 reassignment is wired in Task 5; for now the function still returns the pass-1 result.)

- [ ] **Step 1: Write the failing tests (append)**

```python
# tests/test_rwp_branch_resolution.py  (append)
def test_pass1_rejects_in_band_interleaver():
    g = nx.Graph()
    # strong train, lat ~50, lon 10..70
    _node(g, ("max", 0), 10.0, 50.0); _node(g, ("min", 0), 40.0, 51.0); _node(g, ("max", 2), 70.0, 50.0)
    g.add_edge(("max", 0), ("min", 0), weight=5.0)
    g.add_edge(("min", 0), ("max", 2), weight=5.0)
    # weak interleaver, lat ~52 (same band), lon 20..50
    _node(g, ("min", 1), 20.0, 52.0); _node(g, ("max", 1), 50.0, 52.0)
    g.add_edge(("min", 1), ("max", 1), weight=0.5)
    paths = get_ranked_paths(g, max_weight=10.0, lat_gate=15.0)
    node_sets = [set(p) for p in paths]
    assert {("max", 0), ("min", 0), ("max", 2)} in node_sets        # strong train kept
    assert {("min", 1), ("max", 1)} not in node_sets                # weak interleaver rejected


def test_pass1_keeps_different_waveguide():
    g = nx.Graph()
    # midlat train lat ~50, lon 10..70
    _node(g, ("max", 0), 10.0, 50.0); _node(g, ("min", 0), 40.0, 50.0); _node(g, ("max", 2), 70.0, 50.0)
    g.add_edge(("max", 0), ("min", 0), weight=5.0)
    g.add_edge(("min", 0), ("max", 2), weight=5.0)
    # subtropical train lat ~25 (>15 away), overlapping lon 20..60
    _node(g, ("min", 1), 20.0, 25.0); _node(g, ("max", 1), 60.0, 25.0)
    g.add_edge(("min", 1), ("max", 1), weight=4.0)
    paths = get_ranked_paths(g, max_weight=10.0, lat_gate=15.0)
    node_sets = [set(p) for p in paths]
    assert {("max", 0), ("min", 0), ("max", 2)} in node_sets        # midlat kept
    assert {("min", 1), ("max", 1)} in node_sets                    # subtropical also kept
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rwp_branch_resolution.py -q -k "pass1"`
Expected: FAIL — `get_ranked_paths()` got an unexpected keyword `lat_gate` (and the interleaver is not yet rejected).

- [ ] **Step 3: Implement pass-1 exclusivity**

Change the signature and the greedy loop in `get_ranked_paths`:

```python
def get_ranked_paths(assoc_graph, max_weight, lat_gate=15.0):
```

Replace the final greedy-selection loop with:

```python
    top_paths = []
    used_nodes = set()

    for path in sorted_paths:
        path_nodes = set(path)
        if not path_nodes.isdisjoint(used_nodes):
            continue
        if any(_paths_interleave_in_band(assoc_graph, path, ap, lat_gate)
               for ap in top_paths):
            continue
        top_paths.append(path)
        used_nodes.update(path_nodes)

    return top_paths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rwp_branch_resolution.py -q -k "pass1"`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add waper/identification/rwp_graph.py tests/test_rwp_branch_resolution.py
git commit -m "feat(identification): in-band zonal exclusivity in get_ranked_paths pass-1"
```

---

### Task 4: Orphan reassignment by branch resolution

**Files:**
- Modify: `waper/identification/rwp_graph.py` (add `reassign_orphans`)
- Test: `tests/test_rwp_branch_resolution.py` (append)

**Interfaces:**
- Consumes: `is_to_the_east` (existing).
- Produces: `reassign_orphans(assoc_graph, top_paths, lat_gate=15.0, max_iter=50) -> list[list]` — each input path is a node list; returns refined paths (orphans absorbed at chain ends, weaker branches dropped, length-<2 paths removed).

- [ ] **Step 1: Write the failing tests (append)**

```python
# tests/test_rwp_branch_resolution.py  (append)
from waper.identification.rwp_graph import reassign_orphans


def test_orphan_extends_chain_end():
    # orphan max east of the path's eastmost min -> clean extension (no branch)
    g = nx.Graph()
    _node(g, ("max", 0), 10.0, 50.0); _node(g, ("min", 0), 40.0, 50.0)
    _node(g, ("max", 1), 70.0, 50.0)  # orphan, east of min0
    g.add_edge(("max", 0), ("min", 0), weight=5.0)
    g.add_edge(("min", 0), ("max", 1), weight=4.0)   # discarded edge available
    out = reassign_orphans(g, [[("max", 0), ("min", 0)]], lat_gate=15.0)
    assert out == [[("max", 0), ("min", 0), ("max", 1)]]


def test_orphan_weak_branch_dropped():
    # interior junction: orphan competes with a strong existing east arm and loses
    g = nx.Graph()
    _node(g, ("max", 0), 10.0, 50.0); _node(g, ("min", 0), 40.0, 50.0); _node(g, ("max", 2), 80.0, 50.0)
    g.add_edge(("max", 0), ("min", 0), weight=5.0)
    g.add_edge(("min", 0), ("max", 2), weight=5.0)          # strong east arm
    _node(g, ("max", 1), 55.0, 50.0)                         # orphan east of min0, weak edge
    g.add_edge(("min", 0), ("max", 1), weight=0.5)
    out = reassign_orphans(g, [[("max", 0), ("min", 0), ("max", 2)]], lat_gate=15.0)
    # orphan dropped; strong train intact
    assert out == [[("max", 0), ("min", 0), ("max", 2)]]


def test_orphan_outside_gate_dropped():
    # orphan within longitude but >15 deg latitude away, with an edge -> not absorbed, dropped
    g = nx.Graph()
    _node(g, ("max", 0), 10.0, 50.0); _node(g, ("min", 0), 40.0, 50.0)
    g.add_edge(("max", 0), ("min", 0), weight=5.0)
    _node(g, ("max", 1), 60.0, 25.0)                         # far band
    g.add_edge(("min", 0), ("max", 1), weight=9.0)           # strong edge, but gated out
    out = reassign_orphans(g, [[("max", 0), ("min", 0)]], lat_gate=15.0)
    assert out == [[("max", 0), ("min", 0)]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rwp_branch_resolution.py -q -k "orphan"`
Expected: FAIL — `ImportError: cannot import name 'reassign_orphans'`.

- [ ] **Step 3: Implement `reassign_orphans` (append to `rwp_graph.py`)**

```python
def reassign_orphans(assoc_graph, top_paths, lat_gate=15.0, max_iter=50):
    """Absorb leftover (orphan) nodes into the stronger branch, drop the weaker.

    An orphan attaches to an in-RWP neighbour within ``lat_gate`` degrees of
    latitude. If it would extend a chain end (the existing arm on its side is
    empty) it is absorbed. Otherwise it competes with that arm by summed edge
    weight: the weaker arm is dropped (its nodes re-orphan and may re-attach on a
    later iteration). Orphans with no eligible neighbour are dropped.
    """
    paths = [list(p) for p in top_paths]

    def arm_weight(path, j, direction):
        w = 0.0
        i = j
        while 0 <= i + direction < len(path):
            a, b = path[i], path[i + direction]
            w += assoc_graph[a][b]["weight"]
            i += direction
        return w

    dropped = set()
    for _ in range(max_iter):
        assigned = {n for p in paths for n in p}
        orphans = [n for n in assoc_graph.nodes()
                   if n not in assigned and n not in dropped]
        progressed = False

        for o in orphans:
            o_lon, o_lat = assoc_graph.nodes[o]["coords"]
            cands = [
                (nb, assoc_graph[o][nb]["weight"])
                for nb in assoc_graph.neighbors(o)
                if nb in assigned
                and abs(assoc_graph.nodes[nb]["coords"][1] - o_lat) <= lat_gate
            ]
            if not cands:
                continue

            nb, w_o = max(cands, key=lambda c: c[1])
            pi = next(i for i, p in enumerate(paths) if nb in p)
            path = paths[pi]
            j = path.index(nb)
            direction = 1 if is_to_the_east(o_lon, assoc_graph.nodes[nb]["coords"][0]) else -1
            existing = arm_weight(path, j, direction)

            if w_o <= existing:
                dropped.add(o)                      # orphan's branch is weaker -> drop it
            elif direction == 1:
                paths[pi] = path[: j + 1] + [o]     # drop weaker east arm, splice orphan
            else:
                paths[pi] = [o] + path[j:]          # drop weaker west arm, splice orphan
            progressed = True
            break                                   # recompute assignment after each change

        if not progressed:
            break

    return [p for p in paths if len(p) >= 2]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rwp_branch_resolution.py -q -k "orphan"`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add waper/identification/rwp_graph.py tests/test_rwp_branch_resolution.py
git commit -m "feat(identification): orphan reassignment by branch resolution"
```

---

### Task 5: Wire `lat_gate` through config + acceptance test

**Files:**
- Modify: `waper/identification/rwp_graph.py` (`get_ranked_paths` calls `reassign_orphans` before returning)
- Modify: `waper/interface/api.py` (`WaperConfig.lat_gate`; `Waper.__init__` kwarg + forward; `get_ranked_paths` call site)
- Test: `tests/test_rwp_branch_resolution.py` (append)

**Interfaces:**
- Consumes: `reassign_orphans` (Task 4); `get_ranked_paths(..., lat_gate=...)` (Task 3).
- Produces: end-to-end behaviour on real data; `Waper(..., lat_gate=15.0)`.

- [ ] **Step 1: Write the failing acceptance test (append)**

```python
# tests/test_rwp_branch_resolution.py  (append)
import pytest


def _load_t95():
    import warnings; warnings.filterwarnings("ignore")
    import xarray as xr
    raw = xr.open_dataset("datasets/forecast_bust_hourly.nc")
    da = (raw["v"].rename({"valid_time": "time"})
          .squeeze("pressure_level", drop=True)
          .coarsen(latitude=4, longitude=4, boundary="trim").mean()
          .assign_coords(longitude=lambda d: d.longitude % 360)
          .sortby("longitude"))
    return da.isel(time=[95])


@pytest.mark.slow
def test_acceptance_t95():
    from waper import Waper
    from waper.identification.rwp_graph import (
        _paths_interleave_in_band,
    )
    da = _load_t95()
    w = Waper(data_array=da.to_dataset(name="v"), scalar_name="v",
              latitude_label="latitude", longitude_label="longitude",
              time_label="time", clip_value=2, extrema_threshold=10,
              min_latitude=20, max_latitude=80,
              node_pruning_threshold=20, edge_pruning_threshold=0.02,
              lat_gate=15.0)
    w.identify_rwps()
    tsd = w._time_step_data[0]
    paths = tsd.identified_rwp_paths
    g = tsd.pruned_graph

    # no length-1 RWPs
    assert all(len(p) >= 2 for p in paths)
    # the strong western train survived (a long path remains)
    assert max(len(p) for p in paths) >= 8
    # no in-band spatial overlap between any two returned RWPs
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            assert not _paths_interleave_in_band(g, paths[i], paths[j], 15.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rwp_branch_resolution.py::test_acceptance_t95 -q -m slow`
Expected: FAIL — `Waper.__init__()` got an unexpected keyword `lat_gate`.

- [ ] **Step 3: Wire `lat_gate` through config and the call site**

In `waper/interface/api.py`, add to `WaperConfig` (with the other defaulted fields, e.g. after `energy_radius_km`):

```python
    lat_gate: float = 15.0
```

In `Waper.__init__`, add the parameter (e.g. after `penalty_length_scale_km=2000.0`):

```python
        penalty_length_scale_km=2000.0,
        lat_gate=15.0,
```

and forward it in the `WaperConfig(...)` construction (with the other fields):

```python
            penalty_length_scale_km=penalty_length_scale_km,
            lat_gate=lat_gate,
```

At the `get_ranked_paths` call site (`_identify_rwps`, ~line 213):

```python
    time_step_data.identified_rwp_paths = rwp_graph.get_ranked_paths(
        time_step_data.pruned_graph, config.max_edge_weight, lat_gate=config.lat_gate
    )
```

In `waper/identification/rwp_graph.py`, make `get_ranked_paths` run pass-2 before returning — replace `return top_paths` (the final line) with:

```python
    return reassign_orphans(assoc_graph, top_paths, lat_gate=lat_gate)
```

- [ ] **Step 4: Run the acceptance test and the full unit set**

Run: `pytest tests/test_rwp_branch_resolution.py::test_acceptance_t95 -q -m slow`
Expected: PASS (takes ~5–15 s for the single timestep)

Run: `pytest tests/test_rwp_branch_resolution.py -q`
Expected: PASS (all unit tests + the slow acceptance test)

Also confirm no regression in the existing identification tests:

Run: `pytest tests/ -q -k "rwp or identif or graph" `
Expected: PASS (or unchanged from baseline)

- [ ] **Step 5: Commit**

```bash
git add waper/identification/rwp_graph.py waper/interface/api.py tests/test_rwp_branch_resolution.py
git commit -m "feat(identification): wire lat_gate config + reassign_orphans into get_ranked_paths"
```

---

## Self-Review notes (addressed)

- **Spec coverage:** source==sink fix (Task 1) · span/interleave definitions (Task 2) · pass-1 in-band exclusivity, incl. different-waveguide keep (Task 3) · pass-2 branch resolution: extend / drop-weaker / gate-out (Task 4) · `lat_gate` config + acceptance test on 2011-04-04 23Z, incl. no-length-1 / strong-train-preserved / no-in-band-overlap invariants (Task 5).
- **Placeholder scan:** none — every step has complete code/commands.
- **Type consistency:** `get_ranked_paths(assoc_graph, max_weight, lat_gate=15.0)`, `reassign_orphans(assoc_graph, top_paths, lat_gate=15.0, max_iter=50)`, helper names, and node attr keys (`coords`, `node_type`, `scalar`, edge `weight`) are used identically across tasks and match the existing code.
- **Branch-strength metric** is summed edge weight (`arm_weight`), per the spec; if the acceptance test reveals it drops/keeps the wrong arm, this is the single localised place to revisit.
