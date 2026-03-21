# WAPER Algorithm Documentation

> **Phase 8.3 of the [refactoring spec](../conductor/waper_refactoring_spec.md).**
> This document describes the algorithm as currently implemented. Sections may
> be updated as further refactoring proceeds.

## References

- Pandey, K., Monteiro, J. M., & Natarajan, V. (2020). An Integrated Geometric
  and Topological Approach for the Identification and Visual Analysis of Rossby
  Wave Packets. *Monthly Weather Review*, 148(8), 3139–3157.
- Biju, M. (2022). Identification and Tracking of Rossby Wave Packets.
  BS-MS Thesis, IISER Pune.

---

## 1. Overview

WAPER identifies and tracks Rossby Wave Packets (RWPs) in the meridional wind
field using a purely spatial, graph-based approach. Unlike envelope-based
methods that rely on spectral transforms, WAPER operates directly on the
geometric and topological properties of the scalar field.

The pipeline has two main stages:

1. **Identification** — For each time step, detect alternating maxima and
   minima, cluster them, build an association graph, prune it, and extract
   RWP paths.
2. **Tracking** — Connect RWP features across consecutive time steps using
   spatial overlap of rasterised polygon footprints.

### Pipeline diagram

```
Input scalar field (e.g. meridional wind at 300 hPa)
  │
  ├─ Extrema Detection (§2)
  │    ├─ Local maxima (maximum_filter, size=3)
  │    └─ Local minima (minimum_filter, size=3)
  │
  ├─ Connected Region Labelling (§3)
  │    ├─ Clip field at ±clip_value → positive/negative regions
  │    └─ VTK connectivity filter → RegionId per point
  │
  ├─ Extrema Clustering (§4)
  │    ├─ Geodesic distances via Dijkstra on the mesh
  │    ├─ Hill-climbing penalty for cross-ridge paths
  │    └─ DBSCAN clustering within each connected region
  │
  ├─ Association Graph Construction (§5)
  │    └─ K-D tree proximity on the zero-crossing isocontour
  │
  ├─ Graph Pruning (§6)
  │    ├─ Node pruning (amplitude threshold)
  │    └─ Edge pruning (weight, aspect ratio, longitude separation)
  │
  ├─ Path Extraction & Ranking (§7)
  │    ├─ Monotonic-eastward simple paths
  │    ├─ Region-wrap detection and splitting
  │    └─ Greedy independent-set selection
  │
  ├─ Polygon Footprints (§8)
  │    ├─ Per-node convex hulls in stereographic projection
  │    └─ Weighted centroid computation
  │
  ├─ Rasterisation & Quadtree (§9)
  │    └─ 512×512 stereographic raster → quadtree decomposition
  │
  └─ Temporal Tracking (§10)
       ├─ Quadtree merge → overlap weights
       ├─ Distance pruning
       └─ Topological-sort DP → track extraction
```

---

## 2. Extrema Detection

**File:** `waper/identification/max_min.py`

Local maxima and minima of the scalar field are detected using SciPy's
`maximum_filter` and `minimum_filter` with a 3×3 kernel. The longitude
dimension uses `wrap` mode to handle the periodic boundary; the latitude
dimension uses `constant` mode.

A point is flagged as a maximum if its value equals the local maximum in the
3×3 neighbourhood. Minima are identified analogously.

The detected extrema are then filtered by an amplitude threshold
(`extrema_threshold`): only maxima exceeding this value (and minima below its
negation) are retained. This is done via VTK's `clip_scalar`, which also
provides the connectivity information needed in the next step.

### Key parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `clip_value` | 2 | Threshold for clipping the scalar field into positive/negative regions |
| `extrema_threshold` | 10 | Minimum absolute value for an extremum to be retained |

---

## 3. Connected Region Labelling

**File:** `waper/identification/topology.py` → `identify_connected_regions()`,
`add_connectivity_data_min()`

After clipping the scalar field at `±clip_value`, each contiguous region of
the same sign is assigned a unique `RegionId` using VTK's
`vtkConnectivityFilter` (extraction mode: all regions, with colour-by-region
enabled). This produces a labelling where every grid point in a positive
(negative) region shares a `RegionId` with its spatially connected neighbours.

The `RegionId` serves two purposes:

1. **Clustering scope** — Extrema are only clustered with other extrema in the
   same connected region (§4).
2. **Wrap detection** — During path extraction (§7), if two same-type nodes
   in a candidate path share a `RegionId`, the path has "wrapped around" to
   revisit the same physical feature and is split.

---

## 4. Extrema Clustering

**File:** `waper/identification/topology.py` → `cluster_extrema()`

Within each connected region, nearby extrema of the same sign are grouped into
clusters. Each cluster represents a single wave component (a ridge or trough).
The clustering proceeds as follows:

### 4.1 Distance matrix computation

For every pair of extrema within the same `RegionId`, the geodesic distance
along the mesh is computed using VTK's `vtkDijkstraGraphGeodesicPath`. This
gives the distance *along the scalar field surface*, not the great-circle
distance — an important distinction because it respects the field's geometry.

Pairs in different connected regions are never compared (their distance is set
to `CLUSTER_MAX_DISTANCE = 15000 km`).

### 4.2 Hill-climbing penalty

The raw geodesic distance is augmented with a penalty that discourages merging
extrema separated by a region where the scalar field crosses back through zero
or changes sign. This prevents two peaks on opposite sides of a ridge from
being clustered together.

For maxima (`sign > 0`):

- The **reference value** is the weaker (smaller) of the two peak values.
- The **path minimum** is the smallest scalar value along the Dijkstra path.
- The **fractional descent** is `f = max(0, (reference − path_minimum) / |reference|)`.

For minima (`sign < 0`):

- The **reference value** is the weaker (least negative) trough.
- The **path maximum** is the largest scalar value along the Dijkstra path.
- The **fractional descent** is `f = max(0, (path_maximum − reference) / |reference|)`.

The penalty added to the geodesic distance is:

```
penalty = f × penalty_length_scale_km
```

**Example:** Two maxima at 30 and 25 m/s, with the Dijkstra path dipping to
10 m/s. Reference = 25, descent = 15, f = 0.6, penalty = 0.6 × 2000 = 1200 km
added to the geodesic distance. This makes it harder for DBSCAN to place them
in the same cluster.

### 4.3 DBSCAN clustering

The penalised distance matrix is passed to `sklearn.cluster.DBSCAN` with
`metric="precomputed"` and `min_samples=1`. This means every extremum is
assigned to a cluster (no noise label), but extrema that are too far apart
(after penalty) end up in separate clusters.

Any extrema left unassigned after the per-region DBSCAN passes are given
singleton cluster IDs.

### Key parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `cluster_max_eps_km` | 3000 | DBSCAN `eps` — maximum neighbourhood radius in km |
| `cluster_min_samples` | 2 | DBSCAN `min_samples` |
| `cluster_xi` | 0.15 | (Reserved for OPTICS; not currently used with DBSCAN) |
| `penalty_length_scale_km` | 2000 | Scaling factor for the hill-climbing penalty |

---

## 5. Association Graph Construction

**File:** `waper/identification/rwp_graph.py` → `compute_association_graph()`

The association graph is a bipartite graph connecting maxima clusters to minima
clusters. It encodes which ridges and troughs are adjacent and potentially part
of the same wave packet.

### 5.1 Node representation

Each node in the graph is identified by a tuple `("max", cluster_id)` or
`("min", cluster_id)`. Node attributes include:

| Attribute | Description |
|-----------|-------------|
| `coords` | `(longitude, latitude)` of the cluster's peak/trough point |
| `spherical_coords` | 3D Cartesian coordinates on the VTK sphere |
| `cluster_id` | Integer cluster label |
| `scalar` | Peak scalar value in the cluster |
| `node_type` | `"max"` or `"min"` |
| `cluster_extrema` | List of all extrema points in the cluster |
| `region_id` | `RegionId` of the connected component this cluster belongs to |

### 5.2 Edge construction via the zero-crossing isocontour

The zero-value isocontour of the scalar field is extracted using VTK's
`vtkContourFilter`. For each point on this isocontour, the nearest maximum
cluster and nearest minimum cluster are found using two separate K-D trees
(built from the 3D spherical coordinates of the extrema).

If a `(max_id, min_id)` pair is linked by at least one isocontour point, an
edge is added to the association graph. This ensures that only clusters
separated by a zero-crossing — i.e., genuinely alternating ridges and
troughs — are connected.

---

## 6. Graph Pruning

**File:** `waper/identification/rwp_graph.py` →
`prune_association_graph_nodes()`, `prune_association_graph_edges()`

### 6.1 Node pruning

Edges whose weaker endpoint has `|scalar| < node_pruning_threshold` are
removed, along with any nodes that become disconnected. A maximum scalar
cap (`WAPER_MAX_SCALAR_VALUE = 100`) also filters out unrealistically large
values.

The "weaker endpoint" is determined as the minimum of `max_scalar` and
`|min_scalar|` for each edge, ensuring that both the ridge and trough are
sufficiently strong.

### 6.2 Edge pruning

Each surviving edge is evaluated on three criteria. Edges failing any
criterion are discarded.

#### 6.2.1 Longitude separation

The shortest angular separation in longitude between the two endpoints must
exceed `min_longitude_separation` (default: 6°). This prevents spurious
connections between very close clusters that happen to be separated by a
zero-crossing.

```python
dlon = _longitude_separation(lon_0, lon_1)  # handles wraparound
if dlon <= min_longitude_separation:
    discard
```

#### 6.2.2 Aspect ratio

The ratio `|Δlat| / |Δlon|` must not exceed `max_aspect_ratio` (default: 1.5).
This rejects nearly-meridional connections that do not represent the zonal
propagation characteristic of RWPs.

#### 6.2.3 Edge weight

The edge weight is computed as:

```
weight = (max_scalar − min_scalar) / haversine_distance × zonal_fraction
```

where:

- `max_scalar − min_scalar` is the amplitude contrast across the edge
  (always positive since max > 0 and min < 0).
- `haversine_distance` is the great-circle distance between the cluster
  centres in km.
- `zonal_fraction = Δlon / √(Δlon² + Δlat²)` is an orientation penalty
  that ranges from 1.0 (perfectly zonal edge) to 0.0 (perfectly meridional
  edge). This ensures that tilted edges receive lower weight, making the
  algorithm prefer zonally-oriented wave packet structures.

Edges with weight outside the range `[edge_pruning_threshold, max_edge_weight]`
are discarded.

### Key parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `node_pruning_threshold` | 20 | Minimum `|scalar|` for both endpoints |
| `edge_pruning_threshold` | 3×10⁻⁵ | Minimum edge weight |
| `max_edge_weight` | 1.0 | Maximum edge weight (rejects unrealistic gradients) |
| `min_longitude_separation` | 6.0° | Minimum Δlon between connected clusters |
| `max_aspect_ratio` | 1.5 | Maximum `|Δlat|/|Δlon|` for an edge |

---

## 7. Path Extraction and Ranking

**File:** `waper/identification/rwp_graph.py` → `get_ranked_paths()`

After pruning, the association graph contains only physically plausible
connections. RWP paths are extracted from this graph as the highest-weight
non-overlapping simple paths.

### 7.1 Candidate enumeration

All simple paths between all pairs of nodes are enumerated using
`nx.all_simple_paths()`. Each candidate path must satisfy:

- **Monotonically eastward:** Every successive node must be east of the
  previous one. This enforces the physical expectation that RWP components
  are arranged west-to-east.

The `is_to_the_east()` utility handles longitude wraparound (e.g., 350° is
west of 10°).

### 7.2 Region-wrap detection and splitting

A candidate path may span a very large fraction of the hemisphere, wrapping
around to revisit the same connected component of the scalar field. This is
non-physical — an RWP should not include the same ridge or trough twice.

**Detection:** For each candidate path, the algorithm checks whether any two
same-type nodes (both `"max"` or both `"min"`) share the same `region_id`.
Since `region_id` identifies connected components of the positive (for maxima)
or negative (for minima) scalar field, a duplicate means the path has looped
back to the same physical feature.

**Splitting:** When a wrap is detected, the path is split at its
**weakest edge** (lowest weight). The weakest edge is the natural cut point
because:

- It represents the least confident connection in the path.
- Thanks to the `zonal_fraction` penalty in the edge weight, tilted or
  long-distance edges tend to have lower weight, so they are preferentially
  cut.

Splitting is applied **recursively**: each sub-path is checked for wrapping
and split again if needed. Sub-paths with fewer than 2 nodes are discarded.

### 7.3 Path scoring and greedy selection

Each candidate path (after any splitting) is scored by the sum of its edge
weights. Paths are sorted by score in descending order and selected greedily:

1. Select the highest-weight path.
2. Mark all its nodes as "used".
3. Select the next highest-weight path whose nodes are all disjoint from
   already-used nodes.
4. Repeat until no more disjoint paths remain.

This produces a set of non-overlapping RWP paths that maximise total weight.

---

## 8. Polygon Footprints

**File:** `waper/tracking/rwp_polygon.py` → `get_polygon_for_rwp_path()`

Each identified RWP path is converted into a 2D polygon footprint for
visualisation and for the overlap-based tracking in §10.

### 8.1 Region point extraction

For each node in the path, the scalar field is clipped at `±(path_max / 3)`,
where `path_max` is the strongest absolute scalar value in the path. Points in
the connected region closest to the node's spherical coordinates are collected.

### 8.2 Hull construction

Points are projected from `(lon, lat)` to a polar stereographic coordinate
system (north or south pole, via `pyproj`). Three hull methods are available:

| Method | Description |
|--------|-------------|
| `"per_node"` (default) | Convex hull of each node's region points, then union of all per-node hulls. Produces a tighter fit that follows the wave structure. |
| `"convex"` | Single convex hull of all points across all nodes. Simpler but may include large empty areas between nodes. |
| `"concave"` | Concave hull (Shapely `concave_hull`, ratio=0.3) of all points. Tighter than convex but more computationally expensive. |

### 8.3 Weighted centroid

The representative location of the RWP is computed as the scalar-weighted
average of all region points (in stereographic coordinates), then
inverse-projected back to `(lon, lat)`:

```
weighted_x = Σ(xᵢ × |vᵢ|) / Σ|vᵢ|
weighted_y = Σ(yᵢ × |vᵢ|) / Σ|vᵢ|
```

This centroid is stored as `weighted_longitude` and `weighted_latitude` and
used as the node coordinate in the tracking graph.

---

## 9. Rasterisation and Quadtree

**Files:** `waper/tracking/rwp_polygon.py` → `rasterize_all_rwps()`;
`waper/tracking/quadtree.py`

### 9.1 Rasterisation

All RWP polygons for a given time step are rasterised onto a 512×512
stereographic grid. Each RWP is assigned a unique integer ID; pixels inside
the polygon receive that ID, others remain 0. Overlapping polygons are
resolved by the rasterisation order.

### 9.2 Quadtree construction

The raster image is recursively split into quadrants (4-ary tree). Each
quadtree node stores:

- `mean`: average pixel value
- `features`: set of RWP IDs present in that quadrant
- `level`: depth in the tree
- `start_pixel`: top-left corner of the quadrant

The quadtree enables efficient spatial overlap queries between time steps
without comparing every pixel pair.

### 9.3 Quadtree merge

To compute overlap between RWPs at time `t-1` and time `t`, the two quadtrees
are merged. The merge walks both trees simultaneously; at leaf nodes, the
features from both trees are combined to detect which RWP IDs co-occur in the
same spatial region. The overlap size (number of co-occurring pixels) is
recorded per `(prev_feature, curr_feature)` pair.

---

## 10. Temporal Tracking

**File:** `waper/tracking/tracking_graph.py`

### 10.1 Tracking graph construction

A directed graph is built with nodes `(time_index, rwp_id)`. For consecutive
time steps, edges are added between RWP features that overlap spatially. The
edge weight is:

```
weight = overlap_pixels / max(size_prev, size_curr)
```

where `size_prev` and `size_curr` are the total pixel counts of the respective
RWP features. This normalisation ensures that the overlap is measured relative
to the larger of the two features.

The haversine distance between the weighted centroids is also stored on each
edge.

### 10.2 Tracking graph pruning

Edges whose centroid distance exceeds `track_pruning_threshold` are removed.
This prevents tracking connections between features that are spatially far
apart even if they happen to overlap in the raster (e.g., due to large
polygons).

### 10.3 Track extraction

Tracks are extracted as heaviest-weight paths through the tracking DAG using
a topological-sort dynamic programming algorithm:

1. Topologically sort the DAG.
2. For each node in order, propagate the best cumulative weight to its
   successors.
3. Starting from each sink node (out-degree 0), trace back through
   predecessors to reconstruct the heaviest path ending at that sink.
4. Apply greedy independent-set selection (same algorithm as §7.3) to
   produce non-overlapping tracks.

This runs in O(V + E) time, avoiding the factorial cost of enumerating all
simple paths.

---

## Appendix A: Complete Parameter Reference

| Parameter | Default | Stage | Description |
|-----------|---------|-------|-------------|
| `scalar_name` | (required) | All | Name of the scalar variable in the input dataset |
| `latitude_label` | (required) | All | Name of the latitude coordinate |
| `longitude_label` | (required) | All | Name of the longitude coordinate |
| `time_label` | (required) | All | Name of the time coordinate |
| `clip_value` | 2 | §2–3 | Threshold for clipping into positive/negative regions |
| `extrema_threshold` | 10 | §2 | Minimum absolute value for retained extrema |
| `max_latitude` | None | §2 | Northern latitude bound (degrees) |
| `min_latitude` | None | §2 | Southern latitude bound (degrees) |
| `cluster_max_eps_km` | 3000 | §4 | DBSCAN neighbourhood radius (km) |
| `cluster_min_samples` | 2 | §4 | DBSCAN minimum cluster size |
| `cluster_xi` | 0.15 | §4 | Reserved for OPTICS |
| `penalty_length_scale_km` | 2000 | §4 | Hill-climbing penalty scale factor |
| `node_pruning_threshold` | 20 | §6.1 | Minimum scalar magnitude for node retention |
| `edge_pruning_threshold` | 3×10⁻⁵ | §6.2 | Minimum edge weight |
| `max_edge_weight` | 1.0 | §6.2 | Maximum edge weight |
| `min_longitude_separation` | 6.0° | §6.2 | Minimum Δlon between connected clusters |
| `max_aspect_ratio` | 1.5 | §6.2 | Maximum `|Δlat|/|Δlon|` per edge |
| `hull_method` | `"per_node"` | §8 | Polygon construction method |
| `hemisphere` | `"north"` | §8–9 | Stereographic projection pole |
| `track_pruning_threshold` | 0.3 | §10.2 | Maximum centroid distance for tracking edges |
| `debug` | False | All | Enable debug logging |

## Appendix B: Constants

| Constant | Value | File | Description |
|----------|-------|------|-------------|
| `WAPER_MAX_SCALAR_VALUE` | 100 | `rwp_graph.py` | Upper cap on node scalar values |
| `WAPER_MAX_NODE_DISTANCE` | 1000 | `rwp_graph.py` | (Defined but not actively used) |
| `CLUSTER_MAX_DISTANCE` | 15000 km | `topology.py` | Default distance for cross-region pairs |
| `RADIUS_EARTH_KM` | 6371 | `utils.py` | Earth radius for haversine |
| `RADIUS_SPHERE` | 63.71 | `utils.py` | VTK sphere radius (scaled) |
| `WAPER_IMAGE_SIZE` | 512 | `rwp_polygon.py` | Raster grid resolution |
| `WAPER_SUBSAMPLE` | 5 | `rwp_polygon.py` | Polygon boundary subsampling factor |
| `WAPER_CLUSTER_WIDTH` | 60° | `rwp_polygon.py` | Longitude-wraparound detection threshold |
