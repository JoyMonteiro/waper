# Serialization, Query Layer, and Interactive Visualization — Design & Implementation Plan

> **Context:** Infrastructure that underpins the whole validation program (`conductor/validation_strategy_plan.md`). It is the concrete realization of **Layer 2.4** ("stream per-timestep RWP objects/footprints to disk") and turns the Phase 0 `per_rwp.csv` / `per_track.csv` MVP into a durable, queryable catalogue. Three layers, built in order:
> 1. **Serialization** — a portable "RWP catalogue" on disk (no pickle).
> 2. **Query / science layer** — a small API that returns the *scientifically meaningful* quantities from the references (Souders, Chang & Yu, Hunt, Shah & Monteiro) with no boilerplate.
> 3. **Interactive visualization** — a HoloViz (HoloViews/GeoViews/Panel) explorer reading directly from the catalogue: diverging white-at-zero colorbar, time slider/animation, toggleable edges/nodes/polygons/tracks.

---

## Part 1 — Serialization: the RWP Catalogue format

### 1.1 Principle
WAPER's `WaperSingleTimestepData` mixes analysis data (graphs, polygons, samples) with regenerable heavyweights (pyvista `PolyData`, the quadtree). **Do not pickle it** — pickle is fragile across `networkx`/`shapely`/`pyvista`/Python versions, unsafe to share, and not queryable. Instead decompose by data type into columnar files that are portable, typed, compressed, and partial-read friendly. The graphs are fundamentally relational, so an **edge-list + node-list** representation *is* their natural form and round-trips to `networkx` in two lines.

### 1.2 On-disk layout (one directory per run, partitioned by time)
```
<catalogue>/
  meta.json                      # config, provenance, units, resolution/cadence, waper SHA
  nodes/        year=YYYY/month=MM/*.parquet
  edges/        year=YYYY/month=MM/*.parquet
  rwps/         year=YYYY/month=MM/*.parquet      # GeoParquet (polygon geometry)
  samples/      year=YYYY/month=MM/*.parquet
  track_nodes/  year=YYYY/*.parquet
  track_edges/  year=YYYY/*.parquet
  fields.zarr/                                    # optional gridded fields (v, raster, envelope)
```

### 1.3 Schemas (verified against the WAPER data model)
- **`nodes`** — one crest/trough per RWP per timestep: `time, rwp_id, node_id, node_type{max|min}, lon, lat, scalar, cluster_id, region_id`. (From `tsd.pruned_graph.nodes[n]`: `coords→(lon,lat)`, `scalar`, `node_type`, `cluster_id`, `region_id`. Drop `spherical_coords`/`cluster_extrema` — derivable, keeps it lean.)
- **`edges`** — RWP-graph topology: `time, rwp_id, src_node_id, dst_node_id`. (From `tsd.pruned_graph.edges`, grouped by the path each edge belongs to.)
- **`rwps`** (GeoParquet) — one RWP per timestep: `time, rwp_id, weighted_lon, weighted_lat, peak_amp, n_nodes, zonal_extent_deg, geometry`. (`weighted_*` from `rwp_info`; `geometry` = the shapely `polygon` stored as a GeoPandas geometry column; `peak_amp = max(|scalar|)` over the path.)
- **`samples`** — point samples per RWP/node: `time, rwp_id, node_id, pt_idx, lon, lat`. (From `rwp_info[...]["sample_points"]`, the Nx2 `(lon,lat)` arrays.)
- **`track_nodes`** — `time, feature_id, lon, lat`. (From `waper._tracking_graph.nodes`, key `(time, feature)`, attr `coords`.)
- **`track_edges`** — `time_from, feat_from, time_to, feat_to, weight, distance`. (From `_tracking_graph.edges`, attrs `weight`, `distance`. **Use the raw graph** so merge/split structure — in-degree>1 / out-degree>1 — is preserved.)
- **`fields.zarr`** — optional `(time, lat, lon)` stacks: the input `v`, the `raster_data` RWP-id map, and (later) the Zimin amplitude envelope from Layer 4. Zarr because it is chunked and parallel-write-friendly — exactly the Layer 2.4 streaming need.
- **`meta.json`** — the `WaperConfig` (incl. GT/ST and the resolution/cadence chosen in Phase 0), dataset provenance, **units**, `waper` git SHA, antimeridian convention. Without this the catalogue is not reproducible.

### 1.4 Module: `waper/io/catalogue.py`
```python
def save_catalogue(waper, path, *, meta=None, partition="month"): ...
def load_catalogue(path) -> "Catalogue": ...   # see Part 2
```
- `save_catalogue` extracts the six tables (functions mirror the Phase 0 `phase0_stats.py` extractors — share that code), writes Parquet via **pyarrow** and GeoParquet via **geopandas**, appends partitions (`pyarrow.dataset` write with `partitioning=["year","month"]`), and stacks fields into Zarr. Multi-year runs call it per time-chunk (the Layer 2.4 streaming loop), never holding everything in memory.
- Stable keys: store tracking-graph nodes as two integer columns `(time, feature)`; reconstruct the tuple/string key on load. RWP-graph `node_id` cast to a stable int.

### 1.5 Tests
- **Round-trip:** save → load → reconstruct `pruned_graph` and `_tracking_graph` with `networkx.from_pandas_edgelist`; assert node/edge counts, merge/split degree sets, and polygon equality (shapely `.equals`) match the in-memory objects.
- **Schema:** assert dtypes/columns; assert units present in `meta.json`.

### 1.6 Why these formats
- **Parquet (pyarrow)** — columnar, typed, compressed, predicate pushdown, language-agnostic.
- **GeoParquet (geopandas)** — standard, keeps shapely geometry (WKB under the hood), spatially queryable.
- **Zarr** — chunked, xarray-native, parallel writes.
- **Rejected:** GraphML (chokes on tuple node ids / array attrs), `nx.node_link_data` JSON (slow, bulky, not queryable at climatology scale — fine only for a single graph), pickle (fragile/unsafe).

---

## Part 2 — Query / science layer

### 2.1 Problem
Right now, extracting the quantities the reference papers report (amplitude PDFs, durations, group velocity, seasonal cycle, wavenumber, merges/splits, region-phase) requires hand-written boilerplate every time. The catalogue should expose those quantities **directly**, each method named after — and documented with — the paper result it produces.

### 2.2 Design
A `Catalogue` object wrapping the on-disk tables, with a **lazy, composable, tidy-output** API:
- **Lazy + composable:** `cat.filter(time=..., region=..., min_amp=...)` returns a narrowed `Catalogue` (predicates pushed down to the Parquet read). Chain freely.
- **Tidy output:** every accessor returns a pandas/GeoPandas DataFrame ready for stats or for hvPlot (Part 3).
- **Backend:** **pandas / geopandas first** — load filtered partitions into memory and aggregate there; this is simplest and fine until the catalogue is genuinely large. Keep the API backend-agnostic so a **DuckDB-over-Parquet** path (out-of-core, predicate pushdown) can be dropped in *later* for multi-year scale without changing call sites. Units travel in `meta` and are attached to outputs.

### 2.3 API surface (each method ↔ a reference result)
```python
cat = load_catalogue(path)

# --- raw accessors (filtered, tidy) ---
cat.rwps(); cat.nodes(); cat.edges(); cat.samples(); cat.tracks()

# --- structural metrics (Chang & Yu; Souders) ---
cat.amplitudes()            # per-RWP peak/mean |v|            (Souders §3a, Fig.1)
cat.zonal_extent()          # deg lon, wraparound-aware
cat.implied_wavenumber()    # 180 / mean adjacent-node spacing (Chang & Yu: 5–8)

# --- track metrics (Souders Table 1) ---
cat.track_durations()       # hours/days; %<8d
cat.track_propagation()     # deg eastward; %<180°
cat.group_velocity()        # m/s  (Souders Fig.7; Chang & Yu)
cat.phase_speed()           # m/s  → downstream-development test c_g>c_p

# --- graph topology (tracking quality; WD/Hunt provenance) ---
cat.merges(); cat.splits()  # in-degree>1 / out-degree>1 on the raw graph
cat.tracks_through(box)      # tracks whose footprint enters a region
cat.provenance(track_id)     # trace parent packet back to genesis lon/lat

# --- climatology aggregations (Souders) ---
cat.seasonal_cycle()         # monthly activity         (Souders Fig.4; Hunt Fig.3)
cat.spatial_frequency(grid)  # gridded RWP frequency     (Souders Figs.2–3)
cat.amplitude_pdf(); cat.duration_pdf()
cat.cross_stat_correlations()# amp~size, dur~prop, amp~c_g (Souders Fig.8)

# --- region/phase queries (WD + Shah & Monteiro) ---
cat.rwps_in(box)
cat.packet_at(point, time)   # the RWP over a point
cat.phase_at(point, time)    # nearest node type + fractional position between nodes
                             # → the explicit "head of warm/cold anomaly" regime variable

# --- external matching (Hunt catalogue, later) ---
cat.match_points(other_df, radius_km)   # POD/FAR-style co-location
```
Most methods are a `groupby`/`agg` or a join across the six tables; keep each small. The Phase 0 `phase0_stats.py` extractors become the first implementations of `amplitudes`/`zonal_extent`/`implied_wavenumber`/`track_durations`/`merges`/`splits`.

### 2.4 Payoff
A Souders Fig. 4 reproduction becomes one line: `cat.filter(region=NH).seasonal_cycle().hvplot.line()`. The Shah & Monteiro regime variable becomes `cat.phase_at(hotspot, day)`. This layer is what the validation **layers consume** — Layer 1 invariants assert on `cat.*` outputs, Layer 2 climatology calls the aggregations, Layer 5 calls `provenance`/`phase_at`.

### 2.5 Tests
Unit-test each method against the Phase 0 hand-computed numbers on `validation.nc`, and against the published target bands (a thin wrapper over the Layer 2 checks).

---

## Part 3 — Interactive visualization upgrade (HoloViz)

Replace the static Matplotlib `_plot_*` in `waper/interface/visualization.py` with a **HoloViews/GeoViews layer set + a Panel explorer** that reads straight from the catalogue (Parquet → DataFrame/GeoDataFrame → hvPlot; `fields.zarr` → xarray). Keep the Matplotlib `_plot_*` for static publication figures; the new module is the *interactive* path.

### 3.1 Stack & dependencies
`hvplot`, `holoviews`, `geoviews`, `panel`, `panel-material-ui`, `datashader`, `geopandas`, `pyarrow`, `zarr` (+ `matplotlib`, already a dep, for the vendored NL colormaps and the retained publication figures). `duckdb` is **deferred** — add it only when the catalogue outgrows in-memory pandas. GeoViews gives cartographic projection + coastlines; hvPlot's `geo=True` routes through it.

### 3.2 The Pandey-style diverging colorbar (white near zero) — use the user's NL maps
Don't reinvent this with `RdBu_r`. The user already has the exact aesthetic in `~/Dropbox/Scripts/myCmap.py`: the `cdictDivergeNL`…`cdictDivergeNL8` family are RdBu-style **non-linear** diverging maps — same colors as the base `cdictDiverge`, but the stops are repositioned to put a **pure-white plateau straddling the center** with saturated colors pushed to the ends. That is precisely the "close-to-zero → white" look. Variants differ in plateau width and ramp speed (NL8 = widest, smoothest white plateau; NL3/NL4 narrower; NL6/NL7/NLtest **truncate the dark ends** — use those when the `|v|` range is small so the saturated extremes aren't wasted).

**Vendor them into the package** as `waper/interface/colormaps.py` — copy the self-contained `cdictDivergeNL*` dicts (they need no external files), build `LinearSegmentedColormap`s, and expose a helper that returns either the mpl `Colormap` (for the kept publication figures, §3) or a pre-sampled Bokeh palette (for HoloViews). The file-based maps in that script (`joyDivCmapX/RdBl/YlGr/PuGr/PuOr/PiYG`) depend on `PYTHONPATH` + `.txt` data files — only vendor those if needed, bundling the `.txt` alongside.

```python
# waper/interface/colormaps.py
import matplotlib.colors as mc
from matplotlib.colors import LinearSegmentedColormap

cdictDivergeNL8 = {...}                    # copied verbatim from myCmap.py (self-contained)
joy_nl8 = LinearSegmentedColormap("JoyNL8", cdictDivergeNL8)

def bokeh_palette(cmap, n=256):
    """Sample an mpl Colormap to a hex palette — preserves the NL white plateau."""
    return [mc.rgb2hex(cmap(i / (n - 1))) for i in range(n)]
```

Use it with a **symmetric, *linear* `clim`** — the non-linearity lives in the colormap, so do **not** add a non-linear norm on top:
```python
import hvplot.xarray  # noqa
from waper.interface.colormaps import joy_nl8, bokeh_palette
vmax = float(abs(v_da).quantile(0.99))
field = v_da.hvplot.quadmesh(
    x="longitude", y="latitude", geo=True, project=True,
    cmap=bokeh_palette(joy_nl8),           # NL palette → white plateau near 0
    clim=(-vmax, vmax),                    # symmetric + linear → plateau centered on v=0
    rasterize=True,                        # 0.25° global ≈ 1M cells: aggregate server-side
    clabel="v (m s⁻¹)", coastline=True, responsive=True, height=500,
)
```

**Verified facts (don't re-derive):**
- Pre-sampling `joy_nl8` to 256 hex preserves the white plateau (it lands at data-fraction 0.502–0.549). So the NL effect survives the Bokeh palette conversion — `bokeh_palette()` is the portable, backend-agnostic path. (Passing the `Colormap` object directly also works via HoloViews' `process_cmap`; confirm once `holoviews` is installed.)
- The plateau is **baked into each variant** and several (incl. NL8: 0.5–0.55) extend slightly to the **positive** side — so with symmetric `clim`, white covers `v=0` up to a small positive value (the Pandey "near-zero and weakly-positive = white" look). Pick a more symmetric variant (e.g. NL) if you want white dead-centered.
- The **same mpl `Colormap` object** drives the retained Matplotlib `_plot_*` publication figures (§3 intro) — so interactive and publication output share one consistent colorbar for free.
- `rasterize=True` (not `datashade`) keeps the colorbar and hover. For crisp Pandey-style discrete bands, add `.opts("Image", color_levels=N)`.

### 3.3 Layers (each toggleable)
Build each as its own element/DynamicMap so it can be shown/hidden independently and combined with `*`:
- **v field** — `quadmesh` as above (the white-at-zero base).
- **nodes** — `nodes_df.hvplot.points(x="lon", y="lat", c="node_type", geo=True, cmap={"max": "red", "min": "blue"})`, hover showing `scalar`.
- **edges** — build a `Path` from the `edges` table joined to node coords (one segment per edge): `gv.Path(segments).opts(color="k")` / `hvplot.paths`.
- **polygons** — GeoParquet straight in: `rwps_gdf.hvplot.polygons(geo=True, alpha=0.25, c="rwp_id", colorbar=False)`.
- **tracks** — centroid paths over time from `track_edges`+`track_nodes`: `hvplot.paths(geo=True)`, optionally colored by track id.

### 3.4 Time slider / animation (preserve zoom/pan)
Use the **DynamicMap trigger pattern** (per the HoloViz skill) so panning/zooming survives frame changes: a Panel time slider / `Player` widget drives a `_time` param; each layer's DynamicMap reads the current time from `self` and returns the same element type every call. `Player` gives play/pause animation; the slider scrubs.

### 3.5 App structure (Panel `Viewer` + Material UI)
```python
import panel as pn, panel_material_ui as pmui, param
pn.extension(throttled=True)

class RWPExplorer(pn.viewable.Viewer):
    time   = param.Integer(default=0)              # driven by slider/Player
    layers = param.ListSelector(                    # toggle overlays
        default=["field", "polygons", "nodes"],
        objects=["field", "nodes", "edges", "polygons", "tracks"])
    min_amp = param.Number(default=14)              # calls cat.filter(min_amp=...)

    def __init__(self, cat, **params):
        self.cat = cat
        # one DynamicMap per layer; each returns an empty element when toggled off
        field = hv.DynamicMap(pn.bind(self._field, self.param.time, self.param.layers))
        nodes = hv.DynamicMap(pn.bind(self._nodes, self.param.time, self.param.layers, self.param.min_amp))
        # ... edges, polygons, tracks ...
        super().__init__(**params)
        self._map = pn.pane.HoloViews(field * nodes * ...,   # overlay
                                      sizing_mode="stretch_width", theme="light_minimal")
        self._player = pmui.Player.from_param(self.param.time)   # or pn.widgets.Player
        self._toggles = pmui.CheckButtonGroup.from_param(self.param.layers)

    def __panel__(self):
        return pmui.Page(title="WAPER RWP Explorer",
                         sidebar=[self._player, self._toggles, ...filters...],
                         main=[self._map])
```
- Each `_layer` callback returns its element when its name is in `self.layers`, else an empty element of the **same type** (skill rule: one element type per DynamicMap).
- Linked side panels (optional, high value): an amplitude **histogram** and a **Hovmöller** (lon–time of the v field) that update with filters; a **Tabulator** of tracks where tapping a row highlights that track on the map (use `Selection1D` + `pn.bind(watch=True)` — `link_selections` does **not** work with DynamicMap).
- Serve with `panel serve app.py --dev --show`.

### 3.6 Performance & correctness notes (from the HoloViz patterns)
- **Big field:** `rasterize=True` for the v quadmesh; consider `resample_when=` and decimating nodes/edges by viewport at global scale.
- **Responsive sizing:** pass `responsive=True, height=N` as **hvPlot arguments**, never via `.opts()` (hvPlot's internal `width=700` otherwise wins); set `sizing_mode="stretch_width"` on the pane; set Bokeh `theme=` on the pane, not globally.
- **Tiles/basemap:** use valid hvPlot tile strings (`"CartoLight"`, `"EsriTerrain"`, …) — not `"CartoDB.DarkMatter"`.
- **Antimeridian:** RWPs cross 180°; GeoViews/cartopy can tear polygons at the dateline. Reuse WAPER's existing wraparound handling when building `Path`/`Polygons`, and test a dateline-crossing case (the Phase 0 / Layer 1 dateline event).
- **Parquet integration:** hvPlot consumes the query-layer DataFrames directly. Filter via the Part 2 API before plotting so only the needed rows reach the browser; the DuckDB push-down path (deferred) makes that filtering out-of-core later.

---

## Implementation order
1. **Part 1 — serialization** (`waper/io/catalogue.py` + round-trip tests). MVP: promote the Phase 0 CSVs to the Parquet/GeoParquet schema; share the Phase 0 extractors.
2. **Part 2 — query layer** (`Catalogue` class). Start with the methods Phase 0/Layer 2 already need (`amplitudes`, `track_durations`, `merges`/`splits`, `seasonal_cycle`); unit-test against Phase 0 numbers.
3. **Part 3 — visualization.** (a) static hvPlot layers reading the catalogue (validate the white-at-zero colorbar against a known Pandey figure); (b) the Panel explorer (slider/Player + layer toggles + amplitude filter); (c) linked panels (histogram, Hovmöller, track Tabulator).

## Relationship to other plans
- **`validation_strategy_plan.md`:** Part 1 implements Layer 2.4 streaming; Part 2 is what every layer's checks call; the Phase 0 `phase0_stats.py` extractors are the seed of Part 2.
- **`western_disturbance_validation_plan.md` / `regime_rwp_structure_plan.md`:** `cat.provenance`, `cat.tracks_through`, and `cat.phase_at` are the exact primitives those plans need (WD provenance, the Shah & Monteiro region-phase regime variable); the explorer's region/phase overlays make those analyses visual.
```
