# Serialization, Query Layer & Visualization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Design doc:** `superpowers/plans/design/serialization_query_viz_plan.md` (read it for rationale). This is the executable counterpart.
>
> **Audience note (Gemini-style handoff):** assume you know Python well but nothing about WAPER. Everything you need — the WAPER object model, exact attribute names, fixtures, run commands — is in *Verified Facts* below. Do not re-derive; do not invent attributes not listed there.

**Goal:** Give WAPER a durable on-disk "RWP catalogue" (Part 1), a boilerplate-free query API that returns the reference-paper quantities (Part 2), and an interactive HoloViz explorer with the user's white-near-zero diverging colorbar (Part 3).

**Architecture:** Decompose each per-timestep WAPER result + the tracking graph into columnar Parquet tables (+ optional Zarr fields) — never pickle. A `Catalogue` class wraps those tables (pandas/geopandas now; DuckDB later) and exposes science methods. A HoloViews/GeoViews/Panel module reads the catalogue directly.

**Tech Stack:** pandas, pyarrow, geopandas/shapely (WKB), zarr/xarray, networkx; then hvplot, holoviews, geoviews, panel, panel-material-ui, datashader, matplotlib.

**Hard ordering:** Part 1 → Part 2 → Part 3. Part 2 reads Part 1's tables; Part 3 reads Part 2's API. Do not start a part before the previous part's tests pass.

---

## Verified Facts (do not re-derive)

**Environment**
- The conda env is pre-activated. Run tests with `pytest` **directly** — never `conda run`. Run from the repo root `/Users/joymonteiro/github/waper`.
- New runtime deps to add to `pyproject.toml` before Part 1: `pyarrow`, `geopandas`, `zarr`. Before Part 3: `hvplot`, `holoviews`, `geoviews`, `panel`, `panel-material-ui`, `datashader`. (`matplotlib`, `shapely`, `networkx`, `xarray`, `numpy`, `pandas` are already deps.)

**WAPER API (verified in `waper/interface/api.py`)**
```python
from waper.interface.api import Waper
w = Waper(data_array=ds, scalar_name="v",
          latitude_label="latitude", longitude_label="longitude", time_label="time",
          clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80.1,
          node_pruning_threshold=15, edge_pruning_threshold=3e-5,
          track_pruning_threshold=0.3, max_edge_weight=1, debug=False)
w.identify_rwps()      # fills w._time_step_data (list, one per timestep); returns None
w.track_rwps()         # builds w._tracking_graph (networkx.DiGraph); returns None
```
- `ds` is an `xarray.Dataset` containing variable `v` with dims `(time, latitude, longitude)`.
- **Per-timestep** object `tsd = w._time_step_data[t]`:
  - `tsd.identified_rwp_paths` — list; each item is a **tuple of node-ids** = one RWP.
  - `tsd.pruned_graph` — `networkx.Graph`; node attrs: `coords=(lon,lat)`, `scalar` (float, m/s), `node_type` (`"max"`|`"min"`), `cluster_id` (int), `region_id` (int). Edges connect the path nodes.
  - `tsd.rwp_info[tuple(path)]` — dict: `polygon` (shapely `Polygon`/`MultiPolygon`), `rwp_id` (int, = index+1), `sample_points` (list of `(lon,lat)` tuples), `weighted_longitude` (float), `weighted_latitude` (float).
- **Tracking graph** `w._tracking_graph` — `networkx.DiGraph`:
  - nodes are `(time_index, feature_id)` tuples with attr `coords=(lon,lat)`.
  - edges connect `(t-1,f_i)→(t,f_j)` with attrs `weight` (float) and `distance` (float, km).
  - **Merges = in-degree>1, splits = out-degree>1.** Use the raw graph (do NOT use `get_track_paths`, which discards this).
- `node_type` is exactly the strings `"max"` and `"min"`.

**Test fixtures (verified in `tests/conftest.py`)** — pytest fixtures returning `xarray.DataArray` named `v`, dims `(latitude, longitude)` unless noted:
- `simple_wave_field` — single timestep, 3 crests/2 troughs, 2.5° grid, 20–80N.
- `two_timestep_field` — dims `(time, latitude, longitude)`, 2 timesteps, packet shifts 5° east (use for tracking).
- `date_line_wave_field` — packet straddling 0°/360° (antimeridian test).
- `flat_field` — all zeros (empty-result test).

**Helpers (verified in `waper/identification/utils.py`)**
```python
from waper.identification.utils import haversine_distance, _longitude_separation
# _longitude_separation(lon1, lon2) -> shortest angular gap in degrees, wraparound-safe
# haversine_distance(lat1, lon1, lat2, lon2) -> km
```

**Conventions for this plan**
- New code lives under `waper/io/` (Parts 1–2) and `waper/interface/` (Part 3). Tests under `tests/io/` and `tests/interface/`.
- A test helper builds a tiny catalogue once and is reused. Put it in `tests/io/conftest.py`.
- Tracking-graph node keys are serialized as a string `f"{time}:{feature}"` and two int columns `time`/`feature`.
- RWP polygons are stored as **WKB bytes** in a normal Parquet column (`geometry_wkb`) — robust with partitioning and reconstructs to a GeoDataFrame on load. (Equivalent to GeoParquet, which is WKB under the hood; we choose WKB-in-Parquet so time-partitioned writes stay trivial.)

---

## File Structure

- `waper/io/__init__.py` — exports `save_catalogue`, `load_catalogue`, `Catalogue`.
- `waper/io/extract.py` — pure functions turning a `Waper` run into pandas DataFrames (no I/O).
- `waper/io/catalogue.py` — `save_catalogue`, `load_catalogue`, the `Catalogue` class (Part 2 methods).
- `waper/interface/colormaps.py` — vendored NL diverging colormaps + `bokeh_palette`.
- `waper/interface/explorer.py` — HoloViz layer builders + `RWPExplorer` Panel app.
- `tests/io/conftest.py`, `tests/io/test_extract.py`, `tests/io/test_catalogue.py`, `tests/io/test_query.py`.
- `tests/interface/test_colormaps.py`, `tests/interface/test_explorer.py`.

---

# PART 1 — Serialization (`waper/io/`)

## Task 1: Package scaffold + dependencies + meta.json

**Files:**
- Create: `waper/io/__init__.py`, `waper/io/extract.py`, `waper/io/catalogue.py`
- Modify: `pyproject.toml` (add `pyarrow`, `geopandas`, `zarr`)
- Test: `tests/io/test_catalogue.py`

- [x] **Step 1: Add deps.** Edit `pyproject.toml` dependencies to include `"pyarrow"`, `"geopandas"`, `"zarr"`. Install: `pip install pyarrow geopandas zarr`.

- [x] **Step 2: Write the failing test** in `tests/io/test_catalogue.py`:

```python
import json
from waper.io.catalogue import write_meta, read_meta

def test_meta_roundtrip(tmp_path):
    meta = {"units": "m s**-1", "resolution_deg": 1.0, "cadence_hours": 6,
            "config": {"node_pruning_threshold": 15}, "waper_sha": "abc123"}
    write_meta(tmp_path, meta)
    back = read_meta(tmp_path)
    assert back["units"] == "m s**-1"
    assert back["config"]["node_pruning_threshold"] == 15
```

- [x] **Step 3: Run it, expect FAIL.** `pytest tests/io/test_catalogue.py::test_meta_roundtrip -v` → ImportError / AttributeError.

- [x] **Step 4: Implement** in `waper/io/catalogue.py`:

```python
import json
from pathlib import Path

def write_meta(path, meta: dict) -> None:
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

def read_meta(path) -> dict:
    return json.loads((Path(path) / "meta.json").read_text())
```
And `waper/io/__init__.py`:
```python
from .catalogue import save_catalogue, load_catalogue, Catalogue, write_meta, read_meta
__all__ = ["save_catalogue", "load_catalogue", "Catalogue", "write_meta", "read_meta"]
```
Leave `save_catalogue`/`load_catalogue`/`Catalogue` as names imported from later tasks — add temporary stubs so the import works:
```python
# placeholder names filled in later tasks
def save_catalogue(*a, **k): raise NotImplementedError
def load_catalogue(*a, **k): raise NotImplementedError
class Catalogue: ...
```

- [x] **Step 5: Run, expect PASS.** `pytest tests/io/test_catalogue.py::test_meta_roundtrip -v`

- [x] **Step 6: Commit.**
```bash
git add pyproject.toml waper/io tests/io/test_catalogue.py
git commit -m "feat(io): scaffold catalogue package + meta.json read/write"
```

## Task 2: Extract node & edge tables

**Files:**
- Modify: `waper/io/extract.py`
- Test: `tests/io/conftest.py`, `tests/io/test_extract.py`

- [x] **Step 1: Add a shared run fixture** in `tests/io/conftest.py`:

```python
import pytest, xarray as xr
from waper.interface.api import Waper

@pytest.fixture(scope="session")
def tracked_waper(two_timestep_field):
    ds = xr.Dataset({"v": two_timestep_field})
    w = Waper(data_array=ds, scalar_name="v",
              latitude_label="latitude", longitude_label="longitude", time_label="time",
              clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80.1,
              node_pruning_threshold=15, edge_pruning_threshold=3e-5,
              track_pruning_threshold=0.3, max_edge_weight=1, debug=False)
    w.identify_rwps()
    w.track_rwps()
    return w
```
(The `two_timestep_field` fixture comes from the top-level `tests/conftest.py`; pytest finds it automatically.)

- [x] **Step 2: Write the failing test** in `tests/io/test_extract.py`:

```python
from waper.io.extract import extract_nodes, extract_edges

def test_extract_nodes(tracked_waper):
    df = extract_nodes(tracked_waper)
    assert set(["time","rwp_id","node_id","node_type","lon","lat","scalar",
                "cluster_id","region_id"]).issubset(df.columns)
    assert len(df) > 0
    assert set(df["node_type"].unique()).issubset({"max","min"})

def test_extract_edges(tracked_waper):
    df = extract_edges(tracked_waper)
    assert set(["time","rwp_id","src_node_id","dst_node_id"]).issubset(df.columns)
    assert len(df) > 0
```

- [x] **Step 3: Run, expect FAIL.** `pytest tests/io/test_extract.py -v`

- [x] **Step 4: Implement** in `waper/io/extract.py`:

```python
import pandas as pd

def extract_nodes(waper) -> pd.DataFrame:
    rows = []
    for t, tsd in enumerate(waper._time_step_data):
        g = tsd.pruned_graph
        for path in tsd.identified_rwp_paths:
            rwp_id = tsd.rwp_info[tuple(path)]["rwp_id"]
            for n in path:
                nd = g.nodes[n]
                lon, lat = nd["coords"]
                rows.append(dict(time=t, rwp_id=rwp_id, node_id=int(n),
                                 node_type=nd["node_type"], lon=float(lon), lat=float(lat),
                                 scalar=float(nd["scalar"]), cluster_id=int(nd["cluster_id"]),
                                 region_id=int(nd["region_id"])))
    return pd.DataFrame(rows)

def extract_edges(waper) -> pd.DataFrame:
    rows = []
    for t, tsd in enumerate(waper._time_step_data):
        for path in tsd.identified_rwp_paths:
            rwp_id = tsd.rwp_info[tuple(path)]["rwp_id"]
            for a, b in zip(path[:-1], path[1:]):
                rows.append(dict(time=t, rwp_id=rwp_id,
                                 src_node_id=int(a), dst_node_id=int(b)))
    return pd.DataFrame(rows)
```

- [x] **Step 5: Run, expect PASS.** `pytest tests/io/test_extract.py -v`

- [x] **Step 6: Commit.**
```bash
git add tests/io/conftest.py tests/io/test_extract.py waper/io/extract.py
git commit -m "feat(io): extract node and edge tables from a Waper run"
```

## Task 3: Extract RWP (polygon, WKB) & sample-point tables

**Files:** Modify `waper/io/extract.py`; Test `tests/io/test_extract.py`

- [x] **Step 1: Write the failing test:**

```python
from waper.io.extract import extract_rwps, extract_samples
from shapely import wkb

def test_extract_rwps(tracked_waper):
    df = extract_rwps(tracked_waper)
    assert set(["time","rwp_id","weighted_lon","weighted_lat","peak_amp",
                "n_nodes","zonal_extent_deg","geometry_wkb"]).issubset(df.columns)
    assert (df["peak_amp"] > 0).all()
    geom = wkb.loads(df["geometry_wkb"].iloc[0])   # round-trips to a shapely geometry
    assert geom.geom_type in ("Polygon","MultiPolygon")

def test_extract_samples(tracked_waper):
    df = extract_samples(tracked_waper)
    assert set(["time","rwp_id","pt_idx","lon","lat"]).issubset(df.columns)
    assert len(df) > 0
```

- [x] **Step 2: Run, expect FAIL.**

- [x] **Step 3: Implement** in `waper/io/extract.py` (add imports `from shapely import wkb` and reuse `_longitude_separation`):

```python
from shapely import wkb
from waper.identification.utils import _longitude_separation

def _zonal_extent_deg(lons):
    if len(lons) < 2:
        return 0.0
    return max(_longitude_separation(a, b) for a in lons for b in lons)

def extract_rwps(waper) -> pd.DataFrame:
    rows = []
    for t, tsd in enumerate(waper._time_step_data):
        g = tsd.pruned_graph
        for path in tsd.identified_rwp_paths:
            info = tsd.rwp_info[tuple(path)]
            scalars = [abs(g.nodes[n]["scalar"]) for n in path]
            lons = [g.nodes[n]["coords"][0] for n in path]
            rows.append(dict(
                time=t, rwp_id=info["rwp_id"],
                weighted_lon=float(info["weighted_longitude"]),
                weighted_lat=float(info["weighted_latitude"]),
                peak_amp=float(max(scalars)), n_nodes=len(path),
                zonal_extent_deg=float(_zonal_extent_deg(lons)),
                geometry_wkb=wkb.dumps(info["polygon"]),
            ))
    return pd.DataFrame(rows)

def extract_samples(waper) -> pd.DataFrame:
    rows = []
    for t, tsd in enumerate(waper._time_step_data):
        for path in tsd.identified_rwp_paths:
            info = tsd.rwp_info[tuple(path)]
            for i, (lon, lat) in enumerate(info["sample_points"]):
                rows.append(dict(time=t, rwp_id=info["rwp_id"], pt_idx=i,
                                 lon=float(lon), lat=float(lat)))
    return pd.DataFrame(rows)
```

- [x] **Step 4: Run, expect PASS.**

- [x] **Step 5: Commit.**
```bash
git add waper/io/extract.py tests/io/test_extract.py
git commit -m "feat(io): extract rwp (WKB polygon) and sample-point tables"
```

## Task 4: Extract tracking-graph tables

**Files:** Modify `waper/io/extract.py`; Test `tests/io/test_extract.py`

- [x] **Step 1: Write the failing test:**

```python
from waper.io.extract import extract_track_nodes, extract_track_edges

def test_extract_track_tables(tracked_waper):
    nodes = extract_track_nodes(tracked_waper)
    edges = extract_track_edges(tracked_waper)
    assert set(["time","feature","lon","lat","key"]).issubset(nodes.columns)
    assert set(["src","dst","time_from","feat_from","time_to","feat_to",
                "weight","distance"]).issubset(edges.columns)
    # keys are the f"{time}:{feature}" strings used to rebuild the DiGraph
    assert edges["src"].iloc[0] == f'{edges["time_from"].iloc[0]}:{edges["feat_from"].iloc[0]}'
```

- [x] **Step 2: Run, expect FAIL.**

- [x] **Step 3: Implement** in `waper/io/extract.py`:

```python
def _key(time, feature):
    return f"{int(time)}:{int(feature)}"

def extract_track_nodes(waper) -> pd.DataFrame:
    g = waper._tracking_graph
    rows = []
    for (time, feature), nd in g.nodes(data=True):
        lon, lat = nd["coords"]
        rows.append(dict(time=int(time), feature=int(feature),
                         lon=float(lon), lat=float(lat), key=_key(time, feature)))
    return pd.DataFrame(rows)

def extract_track_edges(waper) -> pd.DataFrame:
    g = waper._tracking_graph
    rows = []
    for (a, b, ed) in g.edges(data=True):
        (t0, f0), (t1, f1) = a, b
        rows.append(dict(src=_key(t0, f0), dst=_key(t1, f1),
                         time_from=int(t0), feat_from=int(f0),
                         time_to=int(t1), feat_to=int(f1),
                         weight=float(ed.get("weight", 0.0)),
                         distance=float(ed.get("distance", 0.0))))
    return pd.DataFrame(rows)
```

- [x] **Step 4: Run, expect PASS.**

- [x] **Step 5: Commit.**
```bash
git add waper/io/extract.py tests/io/test_extract.py
git commit -m "feat(io): extract tracking-graph node and edge tables"
```

## Task 5: `save_catalogue` — orchestrate + partition writes

**Files:** Modify `waper/io/catalogue.py`; Test `tests/io/test_catalogue.py`

- [x] **Step 1: Write the failing test:**

```python
from pathlib import Path
from waper.io.catalogue import save_catalogue

def test_save_catalogue_writes_tables(tracked_waper, tmp_path):
    save_catalogue(tracked_waper, tmp_path, meta={"units": "m s**-1"})
    for name in ["nodes","edges","rwps","samples","track_nodes","track_edges"]:
        assert (Path(tmp_path)/name).exists(), f"missing {name}"
    assert (Path(tmp_path)/"meta.json").exists()
```

- [x] **Step 2: Run, expect FAIL.**

- [x] **Step 3: Implement** in `waper/io/catalogue.py` (replace the stub):

```python
import pandas as pd
from pathlib import Path
from . import extract

_TABLES = {
    "nodes": extract.extract_nodes,
    "edges": extract.extract_edges,
    "rwps": extract.extract_rwps,
    "samples": extract.extract_samples,
    "track_nodes": extract.extract_track_nodes,
    "track_edges": extract.extract_track_edges,
}

def _write_table(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # single file per save() call; multi-year runs call save_catalogue per chunk
    df.to_parquet(out_dir / "part.parquet", engine="pyarrow", index=False)

def save_catalogue(waper, path, *, meta=None) -> None:
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    for name, fn in _TABLES.items():
        _write_table(fn(waper), path / name)
    write_meta(path, meta or {})
```

> **Multi-year note (Layer 2.4):** call `save_catalogue` once per time-chunk into the *same* `path`; change `_write_table` to write `part_<chunk>.parquet` so partitions accumulate. For v1 (Phase 0 scale) one file per table is fine — `load_catalogue` globs `*.parquet`.

- [x] **Step 4: Run, expect PASS.**

- [x] **Step 5: Commit.**
```bash
git add waper/io/catalogue.py tests/io/test_catalogue.py
git commit -m "feat(io): save_catalogue writes all six tables + meta"
```

## Task 6: `load_catalogue` + `Catalogue` skeleton + round-trip

**Files:** Modify `waper/io/catalogue.py`; Test `tests/io/test_catalogue.py`

- [x] **Step 1: Write the failing round-trip test:**

```python
import networkx as nx
from shapely import wkb
from waper.io.catalogue import save_catalogue, load_catalogue

def test_roundtrip_reconstructs_graphs(tracked_waper, tmp_path):
    save_catalogue(tracked_waper, tmp_path)
    cat = load_catalogue(tmp_path)

    # tracking DiGraph rebuilds with identical edge count + merge/split structure
    te = cat.table("track_edges")
    tg = nx.from_pandas_edgelist(te, "src", "dst",
                                 edge_attr=["weight","distance"], create_using=nx.DiGraph)
    assert tg.number_of_edges() == tracked_waper._tracking_graph.number_of_edges()

    # polygon round-trips
    rwps = cat.table("rwps")
    geom = wkb.loads(rwps["geometry_wkb"].iloc[0])
    assert geom.geom_type in ("Polygon","MultiPolygon")

    # nodes table non-empty and typed
    assert len(cat.table("nodes")) > 0
```

- [x] **Step 2: Run, expect FAIL.**

- [x] **Step 3: Implement** in `waper/io/catalogue.py`:

```python
class Catalogue:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = read_meta(self.path) if (self.path/"meta.json").exists() else {}
        self._cache = {}

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            files = sorted((self.path / name).glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no parquet for table {name!r} in {self.path}")
            self._cache[name] = pd.concat(
                (pd.read_parquet(f, engine="pyarrow") for f in files), ignore_index=True)
        return self._cache[name]

def load_catalogue(path) -> "Catalogue":
    return Catalogue(path)
```

- [x] **Step 4: Run, expect PASS.** `pytest tests/io/test_catalogue.py -v`

- [x] **Step 5: Commit.**
```bash
git add waper/io/catalogue.py tests/io/test_catalogue.py
git commit -m "feat(io): load_catalogue + Catalogue.table + graph round-trip test"
```

---

# PART 2 — Query / science layer (`Catalogue` methods)

> All Part 2 methods are added to the `Catalogue` class in `waper/io/catalogue.py`. Tests go in `tests/io/test_query.py` using a module-scoped catalogue built from `tracked_waper`. Add this fixture to `tests/io/conftest.py`:
> ```python
> @pytest.fixture(scope="session")
> def cat(tracked_waper, tmp_path_factory):
>     from waper.io.catalogue import save_catalogue, load_catalogue
>     p = tmp_path_factory.mktemp("cat")
>     save_catalogue(tracked_waper, p, meta={"units": "m s**-1", "dt_hours": 6})
>     return load_catalogue(p)
> ```

## Task 7: `filter` + raw accessors

**Files:** Modify `waper/io/catalogue.py`; Test `tests/io/test_query.py`

- [ ] **Step 1: Write the failing test:**

```python
def test_filter_and_accessors(cat):
    assert len(cat.rwps()) == len(cat.table("rwps"))
    sub = cat.filter(time=0)
    assert (sub.rwps()["time"] == 0).all()
    assert (sub.nodes()["time"] == 0).all()
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** (add to `Catalogue`):

```python
import numpy as np

class Catalogue:   # ... continued
    def __init__(self, path, _filters=None):
        self.path = Path(path)
        self.meta = read_meta(self.path) if (self.path/"meta.json").exists() else {}
        self._cache = {}
        self._filters = _filters or {}   # e.g. {"time": 0, "min_amp": 14, "region": (w,e,s,n)}

    def filter(self, **kw):
        f = dict(self._filters); f.update(kw)
        c = Catalogue(self.path, _filters=f); c._cache = self._cache
        return c

    def _apply(self, df, has_amp=False, has_lonlat=False):
        f = self._filters
        if "time" in f and "time" in df:
            df = df[df["time"] == f["time"]]
        if has_amp and "min_amp" in f and "peak_amp" in df:
            df = df[df["peak_amp"] >= f["min_amp"]]
        if has_lonlat and "region" in f:
            w, e, s, n = f["region"]
            lon = df["lon"] if "lon" in df else df["weighted_lon"]
            lat = df["lat"] if "lat" in df else df["weighted_lat"]
            df = df[(lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)]
        return df.reset_index(drop=True)

    def rwps(self):    return self._apply(self.table("rwps"), has_amp=True, has_lonlat=True)
    def nodes(self):   return self._apply(self.table("nodes"), has_lonlat=True)
    def edges(self):   return self._apply(self.table("edges"))
    def samples(self): return self._apply(self.table("samples"), has_lonlat=True)
    def tracks(self):  return self.table("track_edges")
```
> Replace the earlier minimal `__init__` with this one; keep `table()`, `load_catalogue`, `save_catalogue`, `write_meta`, `read_meta` as-is.

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit.** `git commit -am "feat(query): Catalogue.filter + raw accessors"`

## Task 8: Structural metrics — amplitudes, zonal_extent, implied_wavenumber

**Files:** Modify `waper/io/catalogue.py`; Test `tests/io/test_query.py`

- [ ] **Step 1: Write the failing test:**

```python
def test_structural_metrics(cat):
    amp = cat.amplitudes()
    assert {"time","rwp_id","peak_amp"}.issubset(amp.columns)
    wn = cat.implied_wavenumber()
    assert "implied_wavenumber" in wn.columns
    # synthetic field is wavenumber-4: implied wavenumber should be in a plausible band
    assert wn["implied_wavenumber"].dropna().between(2, 12).mean() > 0.5
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** (add to `Catalogue`; reuse `_longitude_separation`):

```python
    def amplitudes(self):
        return self.rwps()[["time","rwp_id","peak_amp"]]

    def zonal_extent(self):
        return self.rwps()[["time","rwp_id","zonal_extent_deg"]]

    def implied_wavenumber(self):
        from waper.identification.utils import _longitude_separation
        out = []
        nodes = self.nodes()
        for (t, rid), g in nodes.groupby(["time","rwp_id"]):
            lons = sorted(g["lon"].tolist())
            if len(lons) < 2:
                wn = np.nan
            else:
                gaps = [_longitude_separation(a, b) for a, b in zip(lons[:-1], lons[1:])]
                spacing = float(np.mean(gaps))
                wn = 180.0 / spacing if spacing > 0 else np.nan
            out.append(dict(time=t, rwp_id=rid, implied_wavenumber=wn))
        return pd.DataFrame(out)
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit.** `git commit -am "feat(query): amplitudes, zonal_extent, implied_wavenumber"`

## Task 9: Track metrics — durations, propagation, group_velocity

**Files:** Modify `waper/io/catalogue.py`; Test `tests/io/test_query.py`

> **Scope note:** `group_velocity` is computed from track-centroid zonal displacement. `phase_speed` (individual node speed → the `c_g>c_p` downstream-development test) requires node-level tracking, which WAPER does not natively provide — **deferred to a follow-up plan**; do not stub it here.

- [ ] **Step 1: Write the failing test:**

```python
def test_track_metrics(cat):
    dur = cat.track_durations()
    assert {"track_id","duration_hours"}.issubset(dur.columns)
    prop = cat.track_propagation()
    assert {"track_id","propagation_deg"}.issubset(prop.columns)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** (add to `Catalogue`):

```python
    def _track_paths(self):
        """Rebuild the DiGraph and return longest-weight track paths (list of node keys)."""
        import networkx as nx
        from waper.tracking import tracking_graph as tg
        te = self.table("track_edges"); tn = self.table("track_nodes")
        if te.empty:
            return [], {}
        g = nx.from_pandas_edgelist(te, "src", "dst",
                                    edge_attr=["weight","distance"], create_using=nx.DiGraph)
        coords = {r.key: (r.lon, r.lat, r.time) for r in tn.itertuples()}
        for k, (lon, lat, t) in coords.items():
            if k in g: g.nodes[k]["coords"] = (lon, lat)
        return tg.get_track_paths(g), coords

    def track_durations(self):
        from waper.identification.utils import _longitude_separation  # noqa
        dt = float(self.meta.get("dt_hours", 6))
        paths, coords = self._track_paths()
        rows = []
        for i, p in enumerate(paths):
            t0 = coords[p[0]][2]; t1 = coords[p[-1]][2]
            rows.append(dict(track_id=i, duration_steps=t1 - t0,
                             duration_hours=(t1 - t0) * dt))
        return pd.DataFrame(rows)

    def track_propagation(self):
        from waper.identification.utils import _longitude_separation
        paths, coords = self._track_paths()
        rows = []
        for i, p in enumerate(paths):
            lon0 = coords[p[0]][0]; lon1 = coords[p[-1]][0]
            rows.append(dict(track_id=i, propagation_deg=_longitude_separation(lon1, lon0)))
        return pd.DataFrame(rows)

    def group_velocity(self):
        from waper.identification.utils import haversine_distance
        dt = float(self.meta.get("dt_hours", 6)) * 3600.0
        paths, coords = self._track_paths()
        rows = []
        for i, p in enumerate(paths):
            if len(p) < 2: continue
            lon0, lat0, t0 = coords[p[0]]; lon1, lat1, t1 = coords[p[-1]]
            km = haversine_distance(lat0, lon0, lat1, lon1)
            secs = (t1 - t0) * dt
            rows.append(dict(track_id=i, group_velocity_ms=(km*1000.0)/secs if secs else np.nan))
        return pd.DataFrame(rows)
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit.** `git commit -am "feat(query): track durations, propagation, group_velocity (phase_speed deferred)"`

## Task 10: Graph topology — merges, splits, tracks_through, provenance

**Files:** Modify `waper/io/catalogue.py`; Test `tests/io/test_query.py`

- [ ] **Step 1: Write the failing test:**

```python
def test_graph_topology(cat):
    m = cat.merges(); s = cat.splits()
    assert set(m.columns) >= {"key","time","feature","in_degree"}
    assert set(s.columns) >= {"key","time","feature","out_degree"}
    # may legitimately be empty on the 2-step fixture; just assert shape/columns
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** (add to `Catalogue`):

```python
    def _digraph(self):
        import networkx as nx
        te = self.table("track_edges")
        return nx.from_pandas_edgelist(te, "src", "dst",
                                       edge_attr=["weight","distance"], create_using=nx.DiGraph) \
               if not te.empty else nx.DiGraph()

    def _degree_table(self, g, which):
        deg = g.in_degree() if which == "in" else g.out_degree()
        col = "in_degree" if which == "in" else "out_degree"
        rows = []
        for k, d in deg:
            if d > 1:
                t, f = k.split(":")
                rows.append({ "key": k, "time": int(t), "feature": int(f), col: d })
        return pd.DataFrame(rows, columns=["key","time","feature",col])

    def merges(self): return self._degree_table(self._digraph(), "in")
    def splits(self): return self._degree_table(self._digraph(), "out")

    def tracks_through(self, box):
        """Track ids whose any centroid falls in box=(w,e,s,n)."""
        w, e, s, n = box
        tn = self.table("track_nodes")
        inbox = tn[(tn.lon>=w)&(tn.lon<=e)&(tn.lat>=s)&(tn.lat<=n)]
        keys = set(inbox["key"])
        paths, _ = self._track_paths()
        return [i for i, p in enumerate(paths) if any(k in keys for k in p)]

    def provenance(self, track_id):
        """Genesis (lon,lat,time) = first node of the track path."""
        paths, coords = self._track_paths()
        lon, lat, t = coords[paths[track_id][0]]
        return dict(track_id=track_id, genesis_lon=lon, genesis_lat=lat, genesis_time=t)
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit.** `git commit -am "feat(query): merges/splits/tracks_through/provenance"`

## Task 11: Climatology aggregations

**Files:** Modify `waper/io/catalogue.py`; Test `tests/io/test_query.py`

- [ ] **Step 1: Write the failing test:**

```python
def test_climatology_aggregations(cat):
    apdf = cat.amplitude_pdf(bins=5)
    assert {"bin_left","bin_right","density"}.issubset(apdf.columns)
    cc = cat.cross_stat_correlations()
    assert "amp_vs_extent_r" in cc
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** (add to `Catalogue`):

```python
    def amplitude_pdf(self, bins=20):
        a = self.amplitudes()["peak_amp"].to_numpy()
        dens, edges = np.histogram(a, bins=bins, density=True)
        return pd.DataFrame(dict(bin_left=edges[:-1], bin_right=edges[1:], density=dens))

    def duration_pdf(self, bins=20):
        d = self.track_durations()["duration_hours"].to_numpy()
        if d.size == 0:
            return pd.DataFrame(columns=["bin_left","bin_right","density"])
        dens, edges = np.histogram(d, bins=bins, density=True)
        return pd.DataFrame(dict(bin_left=edges[:-1], bin_right=edges[1:], density=dens))

    def seasonal_cycle(self, time_to_month=None):
        """Monthly RWP count. time_to_month maps a `time` index to month 1-12."""
        r = self.rwps().copy()
        if time_to_month is not None:
            r["month"] = r["time"].map(time_to_month)
            return r.groupby("month").size().rename("count").reset_index()
        return r.groupby("time").size().rename("count").reset_index()

    def spatial_frequency(self, dlon=10, dlat=10):
        r = self.rwps()
        lon_bin = (r["weighted_lon"] // dlon) * dlon
        lat_bin = (r["weighted_lat"] // dlat) * dlat
        out = r.assign(lon_bin=lon_bin, lat_bin=lat_bin)
        return out.groupby(["lon_bin","lat_bin"]).size().rename("count").reset_index()

    def cross_stat_correlations(self):
        r = self.rwps()
        amp_extent = r["peak_amp"].corr(r["zonal_extent_deg"]) if len(r) > 2 else np.nan
        return {"amp_vs_extent_r": float(amp_extent)}
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit.** `git commit -am "feat(query): amplitude/duration pdf, seasonal cycle, spatial frequency, correlations"`

## Task 12: Region / phase queries (WD + Shah & Monteiro)

**Files:** Modify `waper/io/catalogue.py`; Test `tests/io/test_query.py`

- [ ] **Step 1: Write the failing test:**

```python
def test_region_phase(cat):
    box = (0, 360, 20, 80)
    assert len(cat.rwps_in(box)) > 0
    res = cat.phase_at((202.5, 50.0), time=0)
    assert {"nearest_node_type","fractional_position","nearest_node_lon"}.issubset(res)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** (add to `Catalogue`):

```python
    def rwps_in(self, box):
        w, e, s, n = box
        r = self.rwps()
        return r[(r.weighted_lon>=w)&(r.weighted_lon<=e)&(r.weighted_lat>=s)&(r.weighted_lat<=n)]

    def packet_at(self, point, time):
        """rwp_id whose weighted centroid is nearest to point at this time (or None)."""
        from waper.identification.utils import haversine_distance
        lon0, lat0 = point
        r = self.filter(time=time).rwps()
        if r.empty: return None
        d = r.apply(lambda x: haversine_distance(lat0, lon0, x.weighted_lat, x.weighted_lon), axis=1)
        return int(r.iloc[int(d.values.argmin())]["rwp_id"])

    def phase_at(self, point, time):
        """Region's wave-phase: nearest node + fractional position between bracketing nodes.
        This is the explicit Shah & Monteiro 'head of warm/cold anomaly' variable."""
        from waper.identification.utils import _longitude_separation
        lon0, lat0 = point
        nd = self.filter(time=time).nodes()
        if nd.empty:
            return dict(nearest_node_type=None, fractional_position=np.nan, nearest_node_lon=np.nan)
        nd = nd.assign(dlon=nd["lon"].apply(lambda L: _longitude_separation(L, lon0)))
        nearest = nd.loc[nd["dlon"].idxmin()]
        west = nd[nd["lon"] <= lon0].sort_values("lon")
        east = nd[nd["lon"] > lon0].sort_values("lon")
        if len(west) and len(east):
            a = west.iloc[-1]["lon"]; b = east.iloc[0]["lon"]
            span = _longitude_separation(b, a)
            frac = _longitude_separation(lon0, a) / span if span > 0 else np.nan
        else:
            frac = np.nan
        return dict(nearest_node_type=nearest["node_type"],
                    fractional_position=float(frac),
                    nearest_node_lon=float(nearest["lon"]))
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit.** `git commit -am "feat(query): rwps_in, packet_at, phase_at (region-phase regime variable)"`

## Task 13: External point matching (Hunt catalogue, deferred-use but implement now)

**Files:** Modify `waper/io/catalogue.py`; Test `tests/io/test_query.py`

- [ ] **Step 1: Write the failing test:**

```python
import pandas as pd
def test_match_points(cat):
    r = cat.rwps()
    pts = pd.DataFrame({"lon":[r.weighted_lon.iloc[0]], "lat":[r.weighted_lat.iloc[0]], "time":[r.time.iloc[0]]})
    m = cat.match_points(pts, radius_km=2000)
    assert "matched" in m.columns and bool(m["matched"].iloc[0]) is True
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** (add to `Catalogue`):

```python
    def match_points(self, other_df, radius_km=850):
        """For each external point (lon,lat,time), True if within radius of any RWP centroid."""
        from waper.identification.utils import haversine_distance
        r = self.rwps()
        out = []
        for row in other_df.itertuples():
            same_t = r[r["time"] == row.time]
            hit = any(haversine_distance(row.lat, row.lon, x.weighted_lat, x.weighted_lon) <= radius_km
                      for x in same_t.itertuples())
            out.append(bool(hit))
        return other_df.assign(matched=out)
```

- [ ] **Step 4: Run, expect PASS.** Then run the whole query suite: `pytest tests/io/test_query.py -v`

- [ ] **Step 5: Commit.** `git commit -am "feat(query): match_points (POD/FAR co-location)"`

---

# PART 3 — Interactive visualization (`waper/interface/`)

## Task 14: Vendored NL diverging colormaps

**Files:** Create `waper/interface/colormaps.py`; Test `tests/interface/test_colormaps.py`

- [ ] **Step 1: Write the failing test:**

```python
from waper.interface.colormaps import joy_nl8, bokeh_palette

def test_nl_palette_white_plateau():
    pal = bokeh_palette(joy_nl8, n=256)
    assert len(pal) == 256
    white = [i for i,c in enumerate(pal) if c.lower() in ("#ffffff","#fefefe")]
    # NL8 white plateau sits just above centre (verified: data-fraction ~0.50–0.55)
    assert white, "expected a white plateau"
    assert 0.45 <= white[0]/255 <= 0.55
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** `waper/interface/colormaps.py` — copy the `cdictDivergeNL8` dict **verbatim** from `~/Dropbox/Scripts/myCmap.py` (self-contained, no file deps), plus a couple of siblings if desired:

```python
"""Vendored non-linear diverging colormaps (white plateau near zero).
Copied from the user's myCmap.py (the self-contained cdict variants only)."""
import matplotlib.colors as mc
from matplotlib.colors import LinearSegmentedColormap

cdictDivergeNL8 = {  # <-- paste the exact tuples from myCmap.py lines 451-499
    'red': ((0.0,0.192,0.192),(0.2,0.270,0.270),(0.3,0.455,0.455),(0.4,.67,.67),
            (0.45,.77,.77),(0.5,1.,1.),(0.525,1.,1.),(0.55,1.,1.),(0.6,.992,.992),
            (0.65,.95,.95),(0.7,0.9,0.9),(0.8,0.843,0.843),(1.0,0.647,0.647)),
    'green': ((0.0,0.211,0.211),(0.2,0.459,0.459),(0.3,0.678,0.678),(0.4,.751,.751),
              (0.45,.851,.851),(0.5,1.,1.),(0.525,1.,1.),(0.55,1.,1.),(0.6,.7,.7),
              (0.65,.682,.682),(0.7,0.427,0.427),(0.8,0.188,0.188),(1.0,0,0)),
    'blue': ((0.0,0.584,0.584),(0.2,0.706,0.706),(0.3,0.80,0.80),(0.4,.85,.85),
             (0.45,.914,.914),(0.5,1.,1.),(0.525,1.,1.),(0.55,1.,1.),(0.6,.480,.480),
             (0.65,.35,.35),(0.7,0.3,0.3),(0.8,0.253,0.253),(1.0,0.2,0.2)),
}
joy_nl8 = LinearSegmentedColormap("JoyNL8", cdictDivergeNL8)

def bokeh_palette(cmap, n=256):
    """Sample an mpl Colormap to an n-color hex list (preserves the NL white plateau)."""
    return [mc.rgb2hex(cmap(i/(n-1))) for i in range(n)]
```

- [ ] **Step 4: Run, expect PASS.** `pytest tests/interface/test_colormaps.py -v`

- [ ] **Step 5: Commit.** `git commit -am "feat(viz): vendor NL diverging colormaps + bokeh_palette"`

## Task 15: Layer builders (data-level, no render)

**Files:** Create `waper/interface/explorer.py`; Test `tests/interface/test_explorer.py`

> Add deps first: `pip install hvplot holoviews geoviews panel panel-material-ui datashader` and add them to `pyproject.toml`.

- [ ] **Step 1: Write the failing test** (build elements, assert types — no server):

```python
import holoviews as hv
from waper.interface import explorer

def test_layer_builders_return_elements(cat):
    hv.extension("bokeh")
    nodes = explorer.nodes_layer(cat, time=0)
    polys = explorer.polygons_layer(cat, time=0)
    assert isinstance(nodes, hv.Points)
    assert isinstance(polys, hv.Polygons)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** `waper/interface/explorer.py`:

```python
import geopandas as gpd
import holoviews as hv
import hvplot.pandas  # noqa
from shapely import wkb
from .colormaps import joy_nl8, bokeh_palette

NODE_CMAP = {"max": "#b2182b", "min": "#2166ac"}

def nodes_layer(cat, time):
    df = cat.filter(time=time).nodes()
    if df.empty:
        return hv.Points([], kdims=["lon","lat"])
    return df.hvplot.points(x="lon", y="lat", c="node_type", cmap=NODE_CMAP,
                            geo=True, hover_cols=["scalar","node_type"],
                            responsive=True, height=500)

def polygons_layer(cat, time):
    df = cat.filter(time=time).rwps()
    if df.empty:
        return hv.Polygons([])
    gdf = gpd.GeoDataFrame(df.assign(geometry=df["geometry_wkb"].apply(wkb.loads)),
                           geometry="geometry", crs="EPSG:4326")
    return gdf.hvplot.polygons(geo=True, alpha=0.25, c="rwp_id",
                               colorbar=False, responsive=True, height=500)

def edges_layer(cat, time):
    nd = cat.filter(time=time).nodes().set_index(["rwp_id","node_id"])
    ed = cat.filter(time=time).edges()
    segs = []
    for r in ed.itertuples():
        try:
            a = nd.loc[(r.rwp_id, r.src_node_id)]; b = nd.loc[(r.rwp_id, r.dst_node_id)]
        except KeyError:
            continue
        segs.append([(a.lon, a.lat), (b.lon, b.lat)])
    return hv.Path(segs).opts(color="black")

def field_layer(field_da, time_index):
    da = field_da.isel(time=time_index)
    vmax = float(abs(da).quantile(0.99))
    return da.hvplot.quadmesh(x="longitude", y="latitude", geo=True, project=True,
                              cmap=bokeh_palette(joy_nl8), clim=(-vmax, vmax),
                              rasterize=True, clabel="v (m s⁻¹)",
                              coastline=True, responsive=True, height=500)
```

- [ ] **Step 4: Run, expect PASS.** `pytest tests/interface/test_explorer.py -v`

- [ ] **Step 5: Commit.** `git commit -am "feat(viz): catalogue-backed layer builders (nodes/edges/polygons/field)"`

## Task 16: `RWPExplorer` Panel app + render smoke test

**Files:** Modify `waper/interface/explorer.py`; Test `tests/interface/test_explorer.py`

- [ ] **Step 1: Write the failing smoke test** (build app + force a Bokeh render; no browser):

```python
def test_explorer_renders(cat):
    import holoviews as hv
    app = explorer.RWPExplorer(cat, n_times=2)
    hv.render(app._map.object)   # forces the overlay to build without a server

def test_layer_toggle(cat):
    app = explorer.RWPExplorer(cat, n_times=2)
    app.layers = ["polygons"]    # toggling must not raise
    hv.render(app._map.object)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** (append to `waper/interface/explorer.py`):

```python
import panel as pn
import param

pn.extension(throttled=True)

class RWPExplorer(pn.viewable.Viewer):
    time = param.Integer(default=0, bounds=(0, 0))
    layers = param.ListSelector(default=["polygons","nodes"],
                                objects=["nodes","edges","polygons"])

    def __init__(self, cat, n_times, **params):
        self.cat = cat
        super().__init__(**params)
        self.param.time.bounds = (0, max(0, n_times - 1))
        nodes = hv.DynamicMap(pn.bind(self._nodes, self.param.time, self.param.layers))
        edges = hv.DynamicMap(pn.bind(self._edges, self.param.time, self.param.layers))
        polys = hv.DynamicMap(pn.bind(self._polys, self.param.time, self.param.layers))
        self._map = pn.pane.HoloViews(polys * edges * nodes,
                                      sizing_mode="stretch_width", theme="light_minimal")
        self._slider = pn.widgets.Player.from_param(self.param.time, name="time")
        self._toggles = pn.widgets.CheckButtonGroup.from_param(self.param.layers)

    def _nodes(self, time, layers):
        return nodes_layer(self.cat, time) if "nodes" in layers else hv.Points([], kdims=["lon","lat"])
    def _edges(self, time, layers):
        return edges_layer(self.cat, time) if "edges" in layers else hv.Path([])
    def _polys(self, time, layers):
        return polygons_layer(self.cat, time) if "polygons" in layers else hv.Polygons([])

    def __panel__(self):
        return pn.Column(self._slider, self._toggles, self._map, sizing_mode="stretch_width")
```
> Each callback returns the **same element type** whether on or off (skill rule). `field` is omitted from the toggle set because it needs the gridded `fields.zarr`; add it once Task 17 wires the field source.

- [ ] **Step 4: Run, expect PASS.** `pytest tests/interface/test_explorer.py -v`

- [ ] **Step 5: Manually serve once to eyeball** (optional but recommended): create `scripts/rwp_explorer.py` that loads a catalogue and calls `pn.serve(RWPExplorer(cat, n_times).servable())`; run `panel serve scripts/rwp_explorer.py --dev --show`. Confirm slider scrubs, toggles work, white-near-zero colorbar looks right.

- [ ] **Step 6: Commit.** `git commit -am "feat(viz): RWPExplorer Panel app + render smoke tests"`

## Task 17: Field layer + Hovmöller + track table (optional, high value)

**Files:** Modify `waper/interface/explorer.py`; Test `tests/interface/test_explorer.py`

- [ ] **Step 1:** Add a `field_da` arg to `RWPExplorer.__init__` (default `None`); when present, add `"field"` to `layers.objects`, add a `_field` DynamicMap built with `field_layer`, and prepend it to the overlay (`field * polys * edges * nodes`). Build the field source from `fields.zarr` if the catalogue has one, else pass the original `v` DataArray.

- [ ] **Step 2:** Add a tracks Tabulator: `pn.widgets.Tabulator(cat.track_durations())`; on row select (`Selection1D` + `pn.bind(watch=True)`) highlight that track's path on the map (a separate DynamicMap returning an `hv.Path`). **Do not** use `link_selections` (incompatible with DynamicMap).

- [ ] **Step 3:** Add a Hovmöller side panel: `field_da.mean("latitude").hvplot.image(x="longitude", y="time", cmap=bokeh_palette(joy_nl8), clim=(-vmax,vmax))`.

- [ ] **Step 4:** Render smoke test for each new piece (`hv.render(...)`), as in Task 16.

- [ ] **Step 5: Commit.** `git commit -am "feat(viz): field layer, Hovmöller, linked track table"`

---

## Self-Review checklist (run before handing off)
- **Spec coverage:** Part 1 tables (nodes/edges/rwps/samples/track_nodes/track_edges/meta/fields) → Tasks 1–6; Part 2 every API method in the design doc → Tasks 7–13 (`phase_speed`/`spatial_frequency` variants flagged: phase_speed deferred with reason); Part 3 colormaps/layers/app/linked-panels → Tasks 14–17.
- **Antimeridian:** add a test using the `date_line_wave_field` fixture asserting `extract_rwps` produces a valid (possibly `MultiPolygon`) geometry and `zonal_extent_deg` is sane — add to `tests/io/test_extract.py` before Part 3.
- **Type consistency:** `geometry_wkb` (bytes) everywhere; track keys `f"{time}:{feature}"` everywhere; `node_type` values `"max"`/`"min"`; `dt_hours` read from `meta`.
- **No placeholders:** the only deferred item is `phase_speed` (explicit scope note, needs node-level tracking) — not a stub.

## Dependencies summary
Part 1–2: `pyarrow`, `geopandas`, `zarr` (+ existing pandas/shapely/networkx/xarray).
Part 3: `hvplot`, `holoviews`, `geoviews`, `panel`, `panel-material-ui`, `datashader`, `matplotlib`. `duckdb` deferred.

## Relationship to other plans
- Implements `superpowers/plans/design/serialization_query_viz_plan.md`.
- Part 1 = Layer 2.4 streaming (`validation_strategy_plan.md`). Part 2 methods are what Layers 1/2/5 call. `phase_at`/`provenance`/`tracks_through` are the primitives for `western_disturbance_validation_plan.md` and `regime_rwp_structure_plan.md`.
- The Phase 0 `phase0_stats.py` extractors should be refactored to import from `waper/io/extract.py` once this lands (single source of truth).
```
