# WAPER: Comprehensive Architecture and Algorithm Documentation

## 1. Introduction

The **WAPER** (Rossby **Wa**ve **P**acket **E**xtraction and **R**epresentation) package is an automated software tool designed to identify, extract, and track Rossby Wave Packets (RWPs) from gridded atmospheric datasets, primarily focusing on upper-tropospheric meridional winds ($v$). 

Rossby Wave Packets act as precursors to extreme weather events. Traditionally, algorithms used to identify RWPs have relied on transform-based methods (like the Hilbert transform or complex demodulation) to construct an "envelope" representing the RWP. These methods inherently discard the phase information of the individual wave components, which is critical for understanding local synoptic characteristics. 

WAPER implements an integrated **geometric and topological approach** introduced by *Pandey et al. (2020)* and further extended with a feature-based tracking mechanism from *Malavika Biju's Thesis (2022)*. Operating entirely in the spatial domain, WAPER builds a graph representation of RWPs that encapsulates both the amplitude and the phase (crests and troughs) of the wave packet. Additionally, the software has been upgraded to compute spatial relations on a spherical geometry, completely circumventing the boundary artifacts and distortion issues present in 2D planar tracking.

---

## 2. Mathematical and Algorithmic Framework

The system executes in two primary phases: **Identification** (building the spatial topology within a single timestep) and **Tracking** (connecting topologies across successive timesteps).

### 2.1. Phase 1: Spatial Identification (Per Timestep)

The objective of this phase is to convert a continuous scalar field (the meridional wind, $v$) into a discrete, pruned bipartite graph representing the alternating components of a wave packet.

#### 2.1.1. Critical Point Extraction
The fundamental units of the RWP are the local maxima (southerly wind peaks) and local minima (northerly wind peaks). 
* A grid-based algorithm checks each spatial point against its immediate spatial neighbors.
* To filter out insignificant atmospheric noise, points are discarded if their absolute scalar value falls below an `extrema_threshold` (typically set around $5 \text{ m/s}$).

#### 2.1.2. Level Sets and Sublevel/Superlevel Boundaries
The scalar field is divided into distinct regions using a thresholding operation (`clip_value`):
* **Superlevel Set:** Regions where $v \geq 0$ (or a specified positive threshold). Used to bound maxima.
* **Sublevel Set:** Regions where $v \leq 0$ (or a specified negative threshold). Used to bound minima.

#### 2.1.3. Component Clustering via Affinity Propagation
A wave packet often features a cluster of proximal local extrema rather than a single point. 
* Within each connected component of the superlevel/sublevel sets, the algorithm calculates the **geodesic distance** between every pair of critical points. 
* A similarity matrix is constructed using the negative of the shortest geodesic path distance, penalized by the minimum/maximum scalar value found along the straight-line connecting the points.
* **Affinity Propagation** (a message-passing clustering algorithm) is applied using this similarity matrix. This autonomously determines the optimal number of wave components without requiring a predefined cluster count. We term the resulting groups "$v$-max clusters" and "$v$-min clusters".

#### 2.1.4. Association Graph Construction
We must determine which $v$-max clusters and $v$-min clusters are spatially adjacent to form a continuous RWP. 
* The **zero-isocontour** of the wind field acts as a natural separator between $v$-max and $v$-min clusters.
* For every point on the zero-isocontour, the algorithm finds the nearest local maximum and nearest local minimum. 
* If a $v$-max cluster and a $v$-min cluster share evidence of a boundary on this isocontour, an edge is created between them, forming a bipartite **Association Graph**.

#### 2.1.5. Graph Pruning and Edge Weighting
The raw Association Graph contains many weak or irrelevant connections that do not represent a true physical wave packet. Edges are scored and pruned based on two criteria:
1. **Scalar Weight:** The minimum of the maximum absolute values of the two connected clusters. If this falls below the `node_pruning_threshold`, the node/edge is discarded.
2. **Gradient Weight (Curvature Vorticity):** An estimate of curvature vorticity is computed between the two clusters:
   $$ \text{Estimated Gradient}(e_{i,j}) = \max_{m_i \in M_i, m_j \in M_j} \frac{|v_{m_i}| + |v_{m_j}|}{\text{dist}(m_i, m_j)} $$
   Edges falling below the `edge_pruning_threshold` are pruned.

#### 2.1.6. Representative Path Optimization
Finally, within each connected component of the pruned graph, the algorithm identifies the "representative RWP path". It extracts all simple paths with strictly increasing longitudes (avoiding backward connections) and selects the path that maximizes the sum of the gradient weights.

---

### 2.2. Phase 2: Temporal Tracking (Across Timesteps)

The objective of this phase is to associate the spatial RWP graphs identified at time $t_i$ with those at $t_{i+1}$, capturing their movement, growth, splits (bifurcation), and merges (amalgamation).

#### 2.2.1. Footprint Extraction and Polygon Generation
To track a wave packet, the algorithm must define its spatial boundary. The algorithm draws closed contours enclosing the representative clusters of the RWP and creates a combined spatial footprint for the entire sequence.

#### 2.2.2. Rasterization and Quadtree Representation
Tracking complex geometric polygons using continuous math is computationally intensive. WAPER rasterizes these footprints into a discrete boolean grid (scaled to a power of two, $2^n \times 2^n$).
* This 2D array is compressed into a **Quadtree**. A quadtree recursively subdivides the grid into four quadrants until it reaches a node that is entirely homogeneous (all background or all feature). 
* This hierarchical structure massively accelerates spatial queries and union/intersection operations.

#### 2.2.3. Spatial Overlap Metric
To determine if Feature $A$ at timestep $t_i$ corresponds to Feature $B$ at $t_{i+1}$, the intersection of their Quadtrees is computed. The significance of the overlap is measured by the weight formula:
$$ W(A^i, B^{i+1}) = \frac{Q(A^i, B^{i+1})}{\max(Q^i_A, Q^{i+1}_B)} $$
Where $Q(X)$ denotes the pixel area (derived instantly from the quadtree node levels).

#### 2.2.4. Tracking Graph Construction
A directed **Tracking Graph** is built where nodes represent features at specific timesteps, and edges represent temporal overlaps.
* Edges with weights below `track_pruning_threshold` are removed.
* RWP lifecycles are extracted by finding the maximum-weight paths through this directed acyclic graph (DAG) across time.

---

## 3. Software Architecture Breakdown

The codebase is logically separated into three modules: `identification`, `tracking`, and `interface`.

### 3.1. `waper.identification`
Responsible for per-timestep topology extraction.
* **`max_min.py`:** Provides the `add_maxima_data` and `add_minima_data` functions. It uses explicit neighborhood comparisons on the 2D grid to identify topological extrema.
* **`topology.py`:** The bridge between NumPy grids and VTK structures. It leverages `vtkDijkstraGraphGeodesicPath` to compute shortest paths within the superlevel/sublevel sets and utilizes `sklearn.cluster.AffinityPropagation` for grouping the critical points.
* **`rwp_graph.py`:** Orchestrates the bipartite graph using `networkx.Graph`. Handles the evaluation of edge weights, pruning algorithms, and the extraction of the maximal simple paths.
* **`utils.py`:** Standardizes geometry by wrapping `xarray.DataArray` into spherical grids using `geovista`. Contains spherical coordinate tools like `haversine_distance`.

### 3.2. `waper.tracking`
Responsible for temporal continuity and quadtree logic.
* **`rwp_polygon.py`:** Translates cluster coordinates into `shapely` polygons, bounding them, and rasterizing them onto a boolean grid mapped to global extents.
* **`quadtree.py`:** Custom implementation of a quadtree using `networkx.DiGraph`. It contains recursive logic (`split4`, `insert_node`) to build the tree and a `merge` function to execute rapid spatial intersections between consecutive timesteps.
* **`tracking_graph.py`:** Constructs the inter-timestep tracking graph, iterates through the permutations of features, applies the overlap weight metric, and filters out the final temporal paths.

### 3.3. `waper.interface`
Provides the overarching abstraction and user-facing APIs.
* **`api.py`:** Contains the primary orchestration loop.
  * `WaperConfig`: A dataclass storing all hyperparameters (`extrema_threshold`, `node_pruning_threshold`, variable names).
  * `WaperSingleTimestepData`: A state container holding all intermediate artifacts for a specific time step (raw data, VTK polydata, graphs, quadtrees).
  * `Waper`: The main class exposing `.identify_rwps()` and `.track_rwps()`.
* **`visualization.py`:** Integrates `matplotlib`, `cartopy`, and `pyvista` to plot maps. It renders intermediate and final steps: extrema point clouds, clustered convex hulls, pruned association graphs, raster footprints, and temporal tracking diagrams over geographical projections.

---

## 4. Assessment and Architectural Bottlenecks

While the underlying algorithm faithfully implements the scientific models on a robust spherical projection, specific implementation details present severe performance and scaling bottlenecks. 

1. **Discrete Extrema Searching ($O(N^2)$ Loops):**
   - The current `max_min.py` implementation relies on nested Python `for` loops across the entire latitude/longitude grid to find critical points. This is drastically slow compared to vectorized C-level operations. Replacing this with `scipy.ndimage.maximum_filter` and `minimum_filter` would yield orders of magnitude in speedup.

2. **VTK Overhead for Geodesic Calculations:**
   - `topology.py` converts scalar fields into heavy `vtkUnstructuredGrid` objects and uses ray-casting (`vtkCellLocator`) and Dijkstra's algorithm via `vtkDijkstraGraphGeodesicPath`. This bridging of modern PyVista/NumPy and archaic VTK operations incurs a massive memory and serialization cost. Using `scipy.sparse.csgraph.dijkstra` on a masked grid adjacency matrix is highly recommended.

3. **Quadtree via NetworkX DiGraph:**
   - `quadtree.py` builds the quadtree hierarchy as a `networkx.DiGraph`. NetworkX objects carry substantial dictionary overhead for every node and edge. For recursive spatial intersection, an off-the-shelf spatial index like an R-tree (`geopandas.sindex` / `pygeos.STRtree`) acting directly on the polygons would entirely remove the need for rasterization and custom graph-based quadtrees, making tracking instantaneous.

4. **Exponential Path Search:**
   - In `tracking_graph.py`, extracting final tracks relies on `nx.all_simple_paths`. In dense networks (many feature splits and merges), this scales factorially. Because tracking moves strictly forward in time, the tracking graph is a Directed Acyclic Graph (DAG). Calculating tracks should be replaced with a dynamic programming longest-path algorithm to guarantee linear time complexity $O(V+E)$.