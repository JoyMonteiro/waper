# Three-Method RWP Identification Agreement Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `scripts/method_comparison/` analysis layer that measures, per timestep, where three RWP-identification methods (edge-pruning, node-amplitude, Zimin envelope) agree and disagree on the April 2011 forecast-bust event, and identifies the GT/ST thresholds that best agree with the fixed Zimin reference.

**Architecture:** All three methods are reduced to per-timestep boolean masks on the shared 512×512 north-polar stereographic grid (`waper.tracking.rwp_polygon`). Zimin is computed once and cached; the two WAPER variants are swept. IoU vs the Zimin reference is the agreement metric; the argmax over each sweep is the "best-agreement threshold." Outputs are a CSV and a plotting notebook. No `waper/` core code changes.

**Tech Stack:** Python 3.12, numpy, scipy, xarray, networkx, rasterio (via existing `rwp_polygon`), shapely, matplotlib, cartopy, pandas, pytest.

## Global Constraints

- Dataset: `datasets/forecast_bust_hourly.nc` — ERA5 300 hPa `v`, NH only (0–90°N), 0.25°, hourly, April 2011 (720 steps).
- Preprocessing (verbatim, matches `scripts/feature_tracks_gif.py`): `coarsen(latitude=4, longitude=4, boundary="trim").mean()` → 1°; `assign_coords(longitude=lambda d: d.longitude % 360).sortby("longitude")`; squeeze `pressure_level`; rename `valid_time` → `time`.
- Domain band: **20–80°N**, applied identically to every mask before any metric.
- Zimin reference: Hilbert envelope, zonal wavenumbers **3–11**, threshold **14 m/s** (fixed, never swept).
- GT sweep (`edge_pruning_threshold`): `[0.0, 0.01, 0.02, 0.04, 0.06, 0.08]` (m/s)/km, ST held at 20. **Never use `3e-5` or `0.3`.**
- ST sweep (`node_pruning_threshold` / node-amplitude cutoff): `[10, 15, 20, 25, 30, 35]` m/s.
- Grid constant: `WAPER_IMAGE_SIZE = 512`, hemisphere `"north"` throughout.
- Run tests with `pytest <path> -q` — environment is pre-activated; do NOT wrap in `conda run`.
- Pure analysis layer under `scripts/method_comparison/`; no edits to `waper/` core.

---

## File Structure

```
scripts/method_comparison/
├── __init__.py        # empty, makes the dir importable
├── metrics.py         # iou, disagreement_decomposition, detection_agreement
├── masks.py           # grid helpers, compute_rwp_envelope, the 3 mask builders
├── run_sweep.py       # driver: preprocess → cache Zimin → sweeps → CSV
└── method_comparison.ipynb   # plots + case studies
results/
└── method_comparison_sweep.csv
tests/
└── test_method_comparison.py
```

---

### Task 1: Agreement metrics

**Files:**
- Create: `scripts/method_comparison/__init__.py` (empty)
- Create: `scripts/method_comparison/metrics.py`
- Test: `tests/test_method_comparison.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `iou(a: np.ndarray, b: np.ndarray) -> float` — boolean masks; returns `|a∩b|/|a∪b|`; returns `1.0` if both empty.
  - `disagreement_decomposition(method: np.ndarray, ref: np.ndarray, band: np.ndarray) -> tuple[float, float]` — returns `(method_only_frac, ref_only_frac)` as fractions of `band.sum()`.
  - `detection_agreement(method: np.ndarray, ref: np.ndarray) -> bool` — True iff both have ≥1 True cell or both empty.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_method_comparison.py
import numpy as np
from scripts.method_comparison.metrics import (
    iou, disagreement_decomposition, detection_agreement,
)


def test_iou_disjoint_is_zero():
    a = np.zeros((4, 4), bool); a[0, 0] = True
    b = np.zeros((4, 4), bool); b[3, 3] = True
    assert iou(a, b) == 0.0


def test_iou_identical_is_one():
    a = np.zeros((4, 4), bool); a[1:3, 1:3] = True
    assert iou(a, a.copy()) == 1.0


def test_iou_both_empty_is_one():
    a = np.zeros((4, 4), bool)
    assert iou(a, a.copy()) == 1.0


def test_iou_contained():
    a = np.zeros((4, 4), bool); a[0:2, 0:2] = True   # 4 cells
    b = np.zeros((4, 4), bool); b[0, 0] = True        # 1 cell, subset of a
    # intersection 1, union 4
    assert iou(a, b) == 0.25


def test_disagreement_decomposition():
    band = np.ones((4, 4), bool)            # 16 cells
    method = np.zeros((4, 4), bool); method[0, :] = True   # 4 cells
    ref = np.zeros((4, 4), bool); ref[0, 0] = True; ref[3, 3] = True  # 2 cells
    # method_only = method & ~ref within band = 3 cells -> 3/16
    # ref_only = ref & ~method within band = 1 cell (3,3) -> 1/16
    m_only, r_only = disagreement_decomposition(method, ref, band)
    assert abs(m_only - 3 / 16) < 1e-9
    assert abs(r_only - 1 / 16) < 1e-9


def test_detection_agreement():
    empty = np.zeros((4, 4), bool)
    nonempty = np.zeros((4, 4), bool); nonempty[0, 0] = True
    assert detection_agreement(nonempty, nonempty.copy()) is True
    assert detection_agreement(empty, empty.copy()) is True
    assert detection_agreement(nonempty, empty) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_method_comparison.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.method_comparison.metrics'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/method_comparison/__init__.py
# (empty)
```

```python
# scripts/method_comparison/metrics.py
"""Set-algebra agreement metrics for boolean RWP masks on the shared grid."""
import numpy as np


def iou(a, b):
    """Intersection-over-union of two boolean masks. 1.0 if both are empty."""
    a = a.astype(bool); b = b.astype(bool)
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0
    return inter / union


def disagreement_decomposition(method, ref, band):
    """Return (method_only_frac, ref_only_frac) as fractions of the band area."""
    method = method.astype(bool) & band
    ref = ref.astype(bool) & band
    denom = np.count_nonzero(band)
    if denom == 0:
        return 0.0, 0.0
    method_only = np.count_nonzero(method & ~ref) / denom
    ref_only = np.count_nonzero(ref & ~method) / denom
    return method_only, ref_only


def detection_agreement(method, ref):
    """True iff both masks detect >=1 cell, or both are empty."""
    return bool(np.any(method)) == bool(np.any(ref))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_method_comparison.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/method_comparison/__init__.py scripts/method_comparison/metrics.py tests/test_method_comparison.py
git commit -m "feat(comparison): agreement metrics for RWP masks"
```

---

### Task 2: Grid helpers and Zimin envelope

**Files:**
- Create: `scripts/method_comparison/masks.py`
- Test: `tests/test_method_comparison.py` (append)

**Interfaces:**
- Consumes: `waper.tracking.rwp_polygon` (`WAPER_IMAGE_SIZE`, `_get_raster_transform`, `transform_to_stereographic`).
- Produces:
  - `pixel_lonlat_grid(hemisphere="north") -> tuple[np.ndarray, np.ndarray]` — `(lon, lat)` each `(512, 512)`, the lon/lat of every pixel centre (lon in 0–360).
  - `band_mask(lat_min=20.0, lat_max=80.0, hemisphere="north") -> np.ndarray` — `(512, 512)` bool, True inside the band.
  - `compute_rwp_envelope(v, wavenumber_range=(3, 11)) -> np.ndarray` — Hilbert envelope of a 2D `(lat, lon)` field, same shape.

- [ ] **Step 1: Write the failing tests (append to the test file)**

```python
# tests/test_method_comparison.py  (append)
from scripts.method_comparison.masks import (
    pixel_lonlat_grid, band_mask, compute_rwp_envelope,
)


def test_pixel_lonlat_grid_shapes_and_ranges():
    lon, lat = pixel_lonlat_grid("north")
    assert lon.shape == (512, 512)
    assert lat.shape == (512, 512)
    # NH stereographic grid: latitudes span roughly 0..90 in the disc, NaN/<0 in corners
    finite = np.isfinite(lat)
    assert lat[finite].max() > 85.0
    assert (lon[finite] >= 0).all() and (lon[finite] <= 360).all()


def test_band_mask_excludes_outside():
    bm = band_mask(20.0, 80.0, "north")
    lon, lat = pixel_lonlat_grid("north")
    inside = bm
    # every True pixel must have latitude in [20, 80]
    assert (lat[inside] >= 20.0).all()
    assert (lat[inside] <= 80.0).all()
    assert bm.sum() > 0


def test_compute_rwp_envelope_recovers_modulation():
    # v(x) = A(x) * cos(k x), A slowly varying, k inside the 3-11 band
    nlon = 360
    x = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
    A = 10.0 + 5.0 * np.cos(x)              # wavenumber-1 modulation (outside band)
    carrier = np.cos(7 * x)                 # wavenumber 7 (inside band)
    v = (A * carrier)[None, :]              # shape (1, nlon)
    env = compute_rwp_envelope(v, (3, 11))
    # envelope should track A(x) away from the wrap edges
    interior = slice(30, nlon - 30)
    rel_err = np.abs(env[0, interior] - A[interior]) / A[interior]
    assert rel_err.max() < 0.15
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_method_comparison.py -q`
Expected: FAIL — `ImportError: cannot import name 'pixel_lonlat_grid'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/method_comparison/masks.py
"""Mask builders that reduce each RWP-identification method to a (512,512) bool
mask on the shared north-polar stereographic grid."""
import numpy as np
from scipy.signal import hilbert

from waper.tracking.rwp_polygon import (
    WAPER_IMAGE_SIZE,
    _get_raster_transform,
    transform_to_stereographic,
)


def pixel_lonlat_grid(hemisphere="north"):
    """Longitude/latitude (degrees) of every pixel centre on the 512x512 grid."""
    n = WAPER_IMAGE_SIZE
    tf = _get_raster_transform(hemisphere)
    cols, rows = np.meshgrid(np.arange(n), np.arange(n))
    # Affine maps (col, row) -> (x, y) in stereographic metres; works elementwise.
    xs, ys = tf * (cols + 0.5, rows + 0.5)
    lon, lat = transform_to_stereographic(
        np.asarray(xs), np.asarray(ys), hemisphere=hemisphere, inverse=True
    )
    lon = np.asarray(lon).reshape(n, n) % 360.0
    lat = np.asarray(lat).reshape(n, n)
    return lon, lat


def band_mask(lat_min=20.0, lat_max=80.0, hemisphere="north"):
    """Boolean (512,512) mask, True where the pixel latitude is in [lat_min, lat_max]."""
    _, lat = pixel_lonlat_grid(hemisphere)
    with np.errstate(invalid="ignore"):
        return np.isfinite(lat) & (lat >= lat_min) & (lat <= lat_max)


def compute_rwp_envelope(v, wavenumber_range=(3, 11)):
    """Zimin et al. (2003/2006) Hilbert envelope of a 2D (lat, lon) field.

    FFT along longitude, zero wavenumbers outside the band, inverse FFT to a
    band-passed real field, Hilbert transform -> analytic signal, magnitude.
    """
    v = np.asarray(v, dtype=float)
    nlon = v.shape[-1]
    F = np.fft.fft(v, axis=-1)
    k = np.abs(np.fft.fftfreq(nlon, d=1.0 / nlon))  # integer zonal wavenumbers
    lo, hi = wavenumber_range
    keep = (k >= lo) & (k <= hi)
    F_filt = F * keep
    v_band = np.fft.ifft(F_filt, axis=-1).real
    return np.abs(hilbert(v_band, axis=-1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_method_comparison.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/method_comparison/masks.py tests/test_method_comparison.py
git commit -m "feat(comparison): grid helpers and Zimin Hilbert envelope"
```

---

### Task 3: The three mask builders

**Files:**
- Modify: `scripts/method_comparison/masks.py`
- Test: `tests/test_method_comparison.py` (append)

**Interfaces:**
- Consumes: `pixel_lonlat_grid`, `band_mask` (Task 2); `waper.tracking.rwp_polygon.rasterize_all_rwps`; `waper.tracking.feature_tracks._footprint_from_region`.
- Produces:
  - `zimin_mask(envelope, ds_lon, ds_lat, band, threshold=14.0, hemisphere="north") -> np.ndarray` — `(512,512)` bool. `envelope` is the `(lat, lon)` field; `ds_lon`/`ds_lat` are its 1-D coords (degrees, lon 0–360); `band` is the precomputed band mask.
  - `edge_pruning_mask(time_step_data, band) -> np.ndarray` — reads `time_step_data.raster_data`.
  - `node_amplitude_mask(association_graph, st, band, hemisphere="north") -> np.ndarray` — per-node cluster footprints where `|scalar| >= st`.

- [ ] **Step 1: Write the failing tests (append)**

```python
# tests/test_method_comparison.py  (append)
import networkx as nx
from scripts.method_comparison.masks import (
    zimin_mask, edge_pruning_mask, node_amplitude_mask,
)


class _FakeTSD:
    def __init__(self, raster):
        self.raster_data = raster


def test_edge_pruning_mask_none_raster_is_empty():
    bm = band_mask()
    m = edge_pruning_mask(_FakeTSD(None), bm)
    assert m.shape == (512, 512)
    assert m.sum() == 0


def test_edge_pruning_mask_thresholds_and_bands():
    bm = band_mask()
    raster = np.zeros((512, 512), dtype=np.int32)
    raster[bm] = 1            # label inside band
    raster[~bm] = 2           # label outside band (must be dropped)
    m = edge_pruning_mask(_FakeTSD(raster), bm)
    assert m.sum() == bm.sum()
    assert not m[~bm].any()


def test_zimin_mask_thresholds_within_band():
    bm = band_mask()
    ds_lon = np.arange(0, 360, 1.0)
    ds_lat = np.arange(0, 91, 1.0)          # NH 1-deg
    env = np.zeros((ds_lat.size, ds_lon.size))
    # strong envelope only at 50N, 100E -> should appear; a strong patch at 5N should not
    env[50, 100] = 30.0
    env[5, 100] = 30.0
    m = zimin_mask(env, ds_lon, ds_lat, bm, threshold=14.0)
    assert m.shape == (512, 512)
    assert m.sum() > 0
    _, plat = pixel_lonlat_grid("north")
    assert (plat[m] >= 20.0).all() and (plat[m] <= 80.0).all()


def test_node_amplitude_mask_keeps_only_strong_nodes():
    bm = band_mask()
    g = nx.Graph()
    # one strong cluster near 50N/100E, one weak cluster near 50N/200E
    g.add_node(("max", 0), scalar=30.0,
               cluster_extrema=[((99.0, 49.0), 0, 30.0), ((101.0, 51.0), 0, 28.0),
                                ((100.0, 50.0), 0, 29.0)])
    g.add_node(("max", 1), scalar=8.0,
               cluster_extrema=[((199.0, 49.0), 1, 8.0), ((201.0, 51.0), 1, 7.0),
                                ((200.0, 50.0), 1, 6.0)])
    m = node_amplitude_mask(g, st=20.0, band=bm)
    assert m.shape == (512, 512)
    assert m.sum() > 0
    # nothing should be burned near 200E (weak node dropped); check via lon grid
    plon, plat = pixel_lonlat_grid("north")
    near_weak = m & (np.abs(plon - 200.0) < 5.0) & (np.abs(plat - 50.0) < 5.0)
    assert near_weak.sum() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_method_comparison.py -q`
Expected: FAIL — `ImportError: cannot import name 'zimin_mask'`

- [ ] **Step 3: Write the implementation (append to `masks.py`)**

```python
# scripts/method_comparison/masks.py  (append)
from scipy.interpolate import RegularGridInterpolator

from waper.tracking.rwp_polygon import rasterize_all_rwps
from waper.tracking.feature_tracks import _footprint_from_region


def _empty_mask():
    return np.zeros((WAPER_IMAGE_SIZE, WAPER_IMAGE_SIZE), dtype=bool)


def zimin_mask(envelope, ds_lon, ds_lat, band, threshold=14.0, hemisphere="north"):
    """Threshold the Hilbert envelope at `threshold`, sampled onto the shared grid."""
    plon, plat = pixel_lonlat_grid(hemisphere)
    interp = RegularGridInterpolator(
        (ds_lat, ds_lon), envelope, method="nearest",
        bounds_error=False, fill_value=0.0,
    )
    pts = np.stack([plat.ravel(), plon.ravel()], axis=-1)
    E = interp(pts).reshape(plat.shape)
    return (E >= threshold) & band


def edge_pruning_mask(time_step_data, band):
    """Boolean mask from a WAPER timestep's rasterized RWP footprints."""
    raster = time_step_data.raster_data
    if raster is None:
        return _empty_mask()
    return (np.asarray(raster) > 0) & band


def node_amplitude_mask(association_graph, st, band, hemisphere="north"):
    """Per-cluster footprints for nodes whose |scalar| >= st (no edge connection)."""
    polys = []
    for _, attr in association_graph.nodes(data=True):
        if abs(attr["scalar"]) < st:
            continue
        coords = [pt[0] for pt in attr["cluster_extrema"]]  # pt = ((lon,lat), cid, scalar)
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        geom = _footprint_from_region(lons, lats, hemisphere)
        polys.append((geom, len(polys) + 1))
    raster = rasterize_all_rwps(polys, hemisphere=hemisphere)
    if raster is None:
        return _empty_mask()
    return (np.asarray(raster) > 0) & band
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_method_comparison.py -q`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/method_comparison/masks.py tests/test_method_comparison.py
git commit -m "feat(comparison): zimin, edge-pruning, and node-amplitude mask builders"
```

---

### Task 4: Dataset loader and base WAPER run

**Files:**
- Create: `scripts/method_comparison/run_sweep.py`
- Test: `tests/test_method_comparison.py` (append)

**Interfaces:**
- Consumes: `waper.Waper`; preprocessing constants (Global Constraints).
- Produces:
  - `load_dataset(path="datasets/forecast_bust_hourly.nc") -> xarray.DataArray` — preprocessed `v` DataArray (1°, lon 0–360, dims `time,latitude,longitude`).
  - `run_base_waper(v_da, node_pruning_threshold=5, edge_pruning_threshold=0.02) -> waper.Waper` — a `Waper` after `identify_rwps()`; used to harvest `association_graph` per timestep for node-amplitude.

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/test_method_comparison.py  (append)
import pytest
from scripts.method_comparison.run_sweep import load_dataset, run_base_waper


@pytest.mark.slow
def test_load_dataset_shape():
    v = load_dataset()
    assert set(v.dims) == {"time", "latitude", "longitude"}
    assert float(v.longitude.min()) >= 0.0 and float(v.longitude.max()) < 360.0
    # 1-degree coarsened NH
    assert v.latitude.size <= 91 + 1
    assert v.time.size == 720


@pytest.mark.slow
def test_run_base_waper_has_association_graphs():
    v = load_dataset().isel(time=slice(0, 2))
    w = run_base_waper(v)
    assert len(w._time_step_data) == 2
    assert w._time_step_data[0].association_graph is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_method_comparison.py -q -m slow`
Expected: FAIL — `ImportError: cannot import name 'load_dataset'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/method_comparison/run_sweep.py
"""Driver: preprocess the forecast-bust dataset, cache the Zimin reference masks,
sweep the two WAPER variants, and write the agreement CSV."""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr

from waper import Waper

from .masks import (
    band_mask, compute_rwp_envelope, zimin_mask,
    edge_pruning_mask, node_amplitude_mask,
)
from .metrics import iou, disagreement_decomposition, detection_agreement

DATA_PATH = "datasets/forecast_bust_hourly.nc"
RESULTS_CSV = "results/method_comparison_sweep.csv"
ZIMIN_CACHE = "results/zimin_masks.npy"

GT_GRID = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08]
ST_GRID = [10, 15, 20, 25, 30, 35]
BAND = (20.0, 80.0)
ZIMIN_THRESHOLD = 14.0


def load_dataset(path=DATA_PATH):
    """Load + preprocess to 1-degree, lon 0-360, dims (time, latitude, longitude)."""
    raw = xr.open_dataset(path)
    da = (
        raw["v"]
        .rename({"valid_time": "time"})
        .squeeze("pressure_level", drop=True)
        .coarsen(latitude=4, longitude=4, boundary="trim").mean()
        .assign_coords(longitude=lambda d: d.longitude % 360)
        .sortby("longitude")
    )
    return da


def run_base_waper(v_da, node_pruning_threshold=5, edge_pruning_threshold=0.02):
    """WAPER run used only to harvest per-timestep association graphs."""
    w = Waper(
        data_array=v_da.to_dataset(name="v"),
        scalar_name="v",
        latitude_label="latitude",
        longitude_label="longitude",
        time_label="time",
        clip_value=2,
        extrema_threshold=10,
        min_latitude=20,
        max_latitude=80,
        node_pruning_threshold=node_pruning_threshold,
        edge_pruning_threshold=edge_pruning_threshold,
    )
    w.identify_rwps()
    return w
```

Create the `results/` directory so writes succeed:

```bash
mkdir -p results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_method_comparison.py -q -m slow`
Expected: PASS (2 passed) — takes ~30–60 s for the 2-step base run.

If pytest warns about the unknown `slow` marker, add to `pyproject.toml` under `[tool.pytest.ini_options]`: `markers = ["slow: integration runs that load data"]`.

- [ ] **Step 5: Commit**

```bash
git add scripts/method_comparison/run_sweep.py tests/test_method_comparison.py
git commit -m "feat(comparison): dataset loader and base WAPER run"
```

---

### Task 5: Sweep orchestration and CSV output

**Files:**
- Modify: `scripts/method_comparison/run_sweep.py`
- Test: `tests/test_method_comparison.py` (append)

**Interfaces:**
- Consumes: everything above; `Waper` per GT value.
- Produces:
  - `compute_zimin_masks(v_da, band, threshold=14.0) -> np.ndarray` — stacked `(ntime, 512, 512)` bool reference masks.
  - `sweep(v_da, gt_grid=GT_GRID, st_grid=ST_GRID) -> pandas.DataFrame` — columns `method, threshold, mean_iou, detection_agreement, mean_method_only_frac, mean_zimin_only_frac, n_timesteps`.
  - `main()` — runs the full sweep, writes `RESULTS_CSV`.

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/test_method_comparison.py  (append)
from scripts.method_comparison.run_sweep import compute_zimin_masks, sweep


@pytest.mark.slow
def test_sweep_smoke_two_timesteps():
    v = load_dataset().isel(time=slice(0, 2))
    df = sweep(v, gt_grid=[0.02], st_grid=[20])
    assert set(df.columns) == {
        "method", "threshold", "mean_iou", "detection_agreement",
        "mean_method_only_frac", "mean_zimin_only_frac", "n_timesteps",
    }
    assert set(df["method"]) == {"edge_pruning", "node_amplitude"}
    assert (df["n_timesteps"] == 2).all()
    assert (df["mean_iou"] >= 0).all() and (df["mean_iou"] <= 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_method_comparison.py -q -m slow`
Expected: FAIL — `ImportError: cannot import name 'compute_zimin_masks'`

- [ ] **Step 3: Write the implementation (append to `run_sweep.py`)**

```python
# scripts/method_comparison/run_sweep.py  (append)


def compute_zimin_masks(v_da, band, threshold=ZIMIN_THRESHOLD):
    """Stacked (ntime,512,512) bool Zimin reference masks."""
    lon = v_da.longitude.values
    lat = v_da.latitude.values
    masks = np.empty((v_da.time.size, 512, 512), dtype=bool)
    for t in range(v_da.time.size):
        env = compute_rwp_envelope(v_da.isel(time=t).values, (3, 11))
        masks[t] = zimin_mask(env, lon, lat, band, threshold=threshold)
    return masks


def _aggregate(method_masks, zimin_masks, band, method_name, threshold):
    ious, dets, m_onlys, z_onlys = [], [], [], []
    for mm, zm in zip(method_masks, zimin_masks):
        ious.append(iou(mm, zm))
        dets.append(detection_agreement(mm, zm))
        m_only, z_only = disagreement_decomposition(mm, zm, band)
        m_onlys.append(m_only); z_onlys.append(z_only)
    return {
        "method": method_name,
        "threshold": threshold,
        "mean_iou": float(np.mean(ious)),
        "detection_agreement": float(np.mean(dets)),
        "mean_method_only_frac": float(np.mean(m_onlys)),
        "mean_zimin_only_frac": float(np.mean(z_onlys)),
        "n_timesteps": len(ious),
    }


def sweep(v_da, gt_grid=GT_GRID, st_grid=ST_GRID):
    """Full agreement sweep -> tidy DataFrame."""
    band = band_mask(*BAND)
    zimin_masks = compute_zimin_masks(v_da, band)

    rows = []

    # Node-amplitude: one base run supplies association graphs; re-threshold per ST.
    base = run_base_waper(v_da)
    assoc = [tsd.association_graph for tsd in base._time_step_data]
    for st in st_grid:
        method_masks = [node_amplitude_mask(g, st, band) for g in assoc]
        rows.append(_aggregate(method_masks, zimin_masks, band, "node_amplitude", st))

    # Edge-pruning: a full WAPER run per GT; read raster_data.
    for gt in gt_grid:
        w = run_base_waper(v_da, node_pruning_threshold=20, edge_pruning_threshold=gt)
        method_masks = [edge_pruning_mask(tsd, band) for tsd in w._time_step_data]
        rows.append(_aggregate(method_masks, zimin_masks, band, "edge_pruning", gt))

    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)
    v = load_dataset()
    df = sweep(v)
    df.to_csv(RESULTS_CSV, index=False)
    print(df.to_string(index=False))
    # report best-agreement thresholds
    for method in ("edge_pruning", "node_amplitude"):
        sub = df[df["method"] == method]
        best = sub.loc[sub["mean_iou"].idxmax()]
        print(f"best {method}: threshold={best['threshold']} mean_iou={best['mean_iou']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_method_comparison.py -q -m slow`
Expected: PASS — the smoke sweep runs 2 timesteps for 1 GT + 1 ST (~1–2 min).

- [ ] **Step 5: Commit**

```bash
git add scripts/method_comparison/run_sweep.py tests/test_method_comparison.py
git commit -m "feat(comparison): sweep orchestration and CSV output"
```

---

### Task 6: Plotting notebook (IoU curves, disagreement maps, case studies)

**Files:**
- Create: `scripts/method_comparison/method_comparison.ipynb`

**Interfaces:**
- Consumes: `RESULTS_CSV`; `load_dataset`, `compute_zimin_masks`, `run_base_waper`, mask builders; `band_mask`, `pixel_lonlat_grid`.
- Produces: figures only (no functions other tasks depend on).

This task has no unit test (a notebook of figures). Build it cell-by-cell, executing each cell and confirming a figure renders before moving on.

- [ ] **Step 1: Cell 1 — load the sweep results and plot IoU-vs-threshold**

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("results/method_comparison_sweep.csv")
fig, ax = plt.subplots(figsize=(7, 4))
for method, g in df.groupby("method"):
    g = g.sort_values("threshold")
    ax.plot(g["threshold"], g["mean_iou"], marker="o", label=method)
    best = g.loc[g["mean_iou"].idxmax()]
    ax.scatter([best["threshold"]], [best["mean_iou"]], s=120,
               facecolors="none", edgecolors="k", zorder=5)
ax.set_xlabel("threshold (GT in (m/s)/km, or ST in m/s)")
ax.set_ylabel("mean IoU vs Zimin (14 m/s)")
ax.set_title("Agreement with Zimin envelope — April 2011")
ax.legend(); fig.tight_layout()
```

Run the cell. Expected: two curves with the argmax point circled on each.

- [ ] **Step 2: Cell 2 — climatological disagreement map at best thresholds**

```python
import cartopy.crs as ccrs
from scripts.method_comparison.run_sweep import load_dataset, compute_zimin_masks, run_base_waper
from scripts.method_comparison.masks import band_mask, pixel_lonlat_grid, edge_pruning_mask, node_amplitude_mask

v = load_dataset()
band = band_mask(20.0, 80.0)
zimin = compute_zimin_masks(v, band)
plon, plat = pixel_lonlat_grid("north")

# best thresholds from the CSV
best_gt = df[df.method == "edge_pruning"].sort_values("mean_iou").iloc[-1]["threshold"]
w = run_base_waper(v, node_pruning_threshold=20, edge_pruning_threshold=float(best_gt))
method_masks = np.stack([edge_pruning_mask(tsd, band) for tsd in w._time_step_data])

agree = (method_masks & zimin).mean(axis=0)
method_only = (method_masks & ~zimin).mean(axis=0)
zimin_only = (~method_masks & zimin).mean(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                         subplot_kw={"projection": ccrs.NorthPolarStereo()})
for ax, fld, title in zip(axes, [agree, method_only, zimin_only],
                          ["agree", "edge-only", "zimin-only"]):
    ax.coastlines(); ax.set_extent([-180, 180, 20, 80], ccrs.PlateCarree())
    ax.pcolormesh(plon, plat, np.where(band, fld, np.nan),
                  transform=ccrs.PlateCarree(), vmin=0, vmax=fld.max())
    ax.set_title(title)
fig.tight_layout()
```

Run the cell. Expected: three NH polar maps showing where the methods agree / edge-pruning over-detects / Zimin over-detects.

- [ ] **Step 3: Cell 3 — case-study overlay at the forecast-bust peak**

```python
t = 24 * 14  # 2011-04-15 00Z (index into hourly series)
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": ccrs.NorthPolarStereo()})
ax.coastlines(); ax.set_extent([-180, 180, 20, 80], ccrs.PlateCarree())
ax.contourf(v.longitude, v.latitude, v.isel(time=t),
            levels=np.linspace(-40, 40, 17), cmap="RdBu_r",
            transform=ccrs.PlateCarree())
for mask, color in [(zimin[t], "k"), (method_masks[t], "lime")]:
    ax.contour(plon, plat, mask.astype(float), levels=[0.5],
               colors=color, transform=ccrs.PlateCarree())
ax.set_title("v field + Zimin (black) vs edge-pruning (green), 2011-04-15 00Z")
fig.tight_layout()
```

Run the cell. Expected: the v field with both method footprints overlaid — the visual "why" for divergence.

- [ ] **Step 4: Commit**

```bash
git add scripts/method_comparison/method_comparison.ipynb
git commit -m "feat(comparison): plotting notebook for IoU curves, disagreement maps, case studies"
```

---

## Final run (after all tasks pass)

```bash
mkdir -p results
~/miniconda3/envs/waper/bin/python -m scripts.method_comparison.run_sweep
```

Expected: ~2–3 h; prints the sweep table and the best-agreement GT and ST. Then open the notebook to produce the figures.

---

## Self-Review notes (addressed)

- **Spec coverage:** Zimin reference (Task 2–3, 5) · edge-pruning sweep (Task 3, 5) · node-amplitude sweep (Task 3, 5) · IoU + decomposition + detection (Task 1) · best-threshold identification (Task 5 `main`) · IoU curves + disagreement maps + case studies (Task 6) · 20–80°N band applied everywhere (`band_mask`, every builder) · 1° / hourly / 720-step dataset (Task 4 `load_dataset`) · no `waper/` core edits (all under `scripts/`).
- **Placeholder scan:** none — every code step is complete.
- **Type consistency:** mask builders all return `(512,512)` bool; `band` is passed as a precomputed bool array to every builder and metric; `sweep` columns match the test and the spec CSV schema; `association_graph` node attrs (`scalar`, `cluster_extrema` as `((lon,lat), cid, scalar)`) match `compute_association_graph`.
