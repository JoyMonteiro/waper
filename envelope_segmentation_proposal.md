# Proposal: Zimin Envelope for RWP Zonal Segmentation

## The Problem
Currently, `waper` uses a topological approach (Morse-Smale complexes and connectivity graphs) to identify Rossby Wave Packets (RWPs). This is incredibly robust to meridional tilt, effectively capturing the full structure of the packet. However, the edge addition process can be too greedy zonally, causing distinct packets along the same waveguide to be chained together into one massive, over-extended RWP.

While the Zimin-style Hilbert transform envelope ($E$) perfectly isolates the zonal energy of distinct packets, thresholding $E$ in 2D space (e.g., $E > 15 \text{ m/s}$) is flawed because the 1D transform breaks apart meridionally-tilted RWPs.

## The Solution: Scaffolded Envelopes
To fix the greedy zonal merging without re-introducing the meridional tilt issue, **we decouple structure from amplitude**. We use the `waper` topology as a fixed structural scaffold, and project the Zimin envelope onto it as an amplitude weight. 

Because the scaffold already follows the tilt of the waveguide, evaluating the envelope *along* the scaffold makes the envelope analysis completely tilt-aware.

Here are two proposed implementation strategies:

---

### Approach 1: Graph Watershed (Recommended)
This approach integrates perfectly with the existing `waper` graph infrastructure.

1. **Calculate Global Envelope:** Compute the Zimin envelope $E(\lambda, \phi)$ for the given timestep.
2. **Assign Node Energies:** For every node (extremum) in the `waper` connectivity graph, sample $E$ at that node's coordinate to assign it an energy value.
3. **Identify Local Maxima:** Within an identified RWP graph component, find nodes that are local maxima of $E$ relative to their topological neighbors. A single, distinct RWP should have exactly one prominent energy maximum.
4. **Prune Valleys:** If a single graph component contains multiple prominent energy maxima, it means greedy edge addition has merged distinct packets. Find the "valley" of nodes with low $E$ connecting these peaks.
5. **Segment:** Remove the graph edges in the valley (e.g., where $E$ drops below 30% of the adjacent peaks). The graph naturally separates into the distinct physical packets.

*Why tilt doesn't matter here:* We are evaluating $E$ by walking along the graph edges. Since the edges follow the meridionally tilted wave train, the evaluation inherently tracks the tilt.

---

### Approach 2: Tilt-Aware Zonal Profile
This approach operates on the final polygon footprints rather than the graph.

1. **Calculate Global Envelope:** Compute the Zimin envelope $E(\lambda, \phi)$.
2. **Mask by Footprint:** Use the 2D polygon footprint of a `waper`-identified RWP as a strict spatial mask.
3. **Collapse Meridionally:** For each longitude bin, calculate the maximum (or integral) of $E$ **only for the latitudes inside the polygon mask**. 
4. **Analyze 1D Profile:** This creates a 1D zonal energy profile that naturally "snakes" along the tilted waveguide.
5. **Segment:** Look for deep local minima (valleys) in this 1D profile. Split the RWP polygon at the longitude of the deep minimum.

*Why tilt doesn't matter here:* Because the mask itself contains the tilt, collapsing the bounded area into a 1D array perfectly aggregates the tilted signal without it fracturing across rigid latitude bands.

---

## Conclusion
By limiting the Zimin envelope to a supplementary filtering role—evaluated strictly within the bounds of the topological features `waper` has already identified—we can leverage its powerful zonal isolation physics while completely bypassing its topological flaws.

---

## Revised Recommendation: Envelope-Weighted Edge Pruning (Simplified Approach 1)

After reviewing both approaches against the actual `waper` codebase — specifically how
`compute_association_graph`, `prune_association_graph_edges`, `get_ranked_paths`, and the
downstream `feature_tracks` layer interact — we recommend a **simplified variant of
Approach 1** implemented as an additional criterion inside the existing edge-pruning step.

### Why not Approach 2 (Polygon Splitting)?

Approach 2 is architecturally backward. In `waper`, the polygon is a *derived output*:
`get_ranked_paths` extracts paths from the pruned graph, then
`get_polygon_for_rwp_path` builds a convex hull around each path's nodes. Splitting a
polygon post-hoc means reverse-engineering which graph nodes belong to which fragment
and manually severing edges — undoing work already done. The segmentation decision
should be made *before* paths and polygons are built, at the graph level.

### Why not the full Graph Watershed (original Approach 1)?

The original formulation — find local $E$-maxima within a connected component, locate
valleys, segment — requires a post-hoc walk over assembled paths. This is unnecessarily
complex. The key insight is that greedy merging happens at the level of *individual
edges*, and `waper` already has an edge-by-edge pruning step
(`prune_association_graph_edges` in `rwp_graph.py`). We can add the envelope criterion
directly there.

### The mechanism: node-level $E$, edge-level decision

Each edge in the association graph connects a max cluster centroid to a min cluster
centroid — roughly half a wavelength apart. The Zimin envelope $E$ varies on the scale
of the *wave packet* (thousands of km), not the *wavelength* (hundreds of km). So $E$ is
nearly constant across a single edge. There is no need to sample $E$ at intermediate
points along the edge. We simply evaluate $E$ at the two endpoint nodes:

1. **Compute** the Zimin envelope $E(\lambda, \phi)$ once per timestep.
2. **For each candidate edge** in `prune_association_graph_edges`, look up
   $E(\text{node}_a)$ and $E(\text{node}_b)$.
3. **Discard the edge** if $\min(E_a, E_b)$ falls below a threshold (e.g., 10–15 m/s).

This is a ~15-line addition to the existing edge-pruning function.

### Why the "bridging edge" has low $E$

The greedy edge that chains two distinct packets is born in
`compute_association_graph`: a zero-isocontour point in the low-amplitude gap between
the packets gets its nearest max cluster (trailing crest of packet A) and nearest min
cluster (leading trough of packet B) registered as an edge. Both endpoints can survive
the existing `node_pruning_threshold` because their *instantaneous scalar values*
$|v|$ are still high. But the Zimin envelope at those locations is low: $E$ reflects
the local *wave-packet amplitude modulation*, not the individual extremum's strength.
An isolated 25 m/s trough at the fringe of a decaying packet has high $|v|$ but low $E$.

### Why meridional tilt is a non-issue here

The Zimin envelope's tilt problem arises when thresholding $E$ in 2D to define spatial
boundaries — a tilted packet fractures across rigid latitude bands. But we are
*point-sampling* $E$ at specific node locations. A meridionally tilted crest at
(30°W, 55°N) still has high $E$ at that point because the Hilbert transform captures
the zonal wavenumber amplitude *at that latitude*. The tilt doesn't reduce $E$ at the
extremum's own latitude circle. The problem only manifests when merging across
latitudes via a 2D contour of $E$, which we explicitly avoid.

### Downstream benefit for feature tracking

The greedy merging problem is the root cause of the split-tracking failure documented
in `handoff_feature_tracks.md`: the massive parent footprint produces IoU values too
low for child fragments to inherit continuity (see the purple track at ~60°W). By
severing the bridging edge upstream, `get_ranked_paths` yields two separate, shorter
RWP paths → two smaller polygons → `extract_features` produces tighter convex hulls →
the IoU between parent and child at split time stays high. This would make the
`min_split_iou = 0.05` band-aid unnecessary and allow restoring a robust threshold
(e.g., 0.2).

### Open questions

- **Threshold calibration:** The envelope threshold for edge rejection needs to be
  tuned. A fixed absolute threshold (e.g., 15 m/s) is simple but may not generalise
  across seasons or hemispheres. A relative threshold (e.g., reject if
  $\min(E_a, E_b) < 0.3 \cdot \text{median}(E)$ over all nodes) is more adaptive but
  adds complexity.
- **Interaction with existing edge weight:** The current `edge_weight` function already
  encodes amplitude via $(v_\text{max} - v_\text{min}) / d$. Does the envelope add
  genuinely new information, or does it merely duplicate what the scalar-based weight
  already captures? The hypothesis is that $E$ adds *packet-membership* information
  that individual scalar values cannot provide, but this needs empirical verification.
- **Computational cost:** Computing the Hilbert transform along every latitude circle
  is $O(N_\text{lat} \cdot N_\text{lon} \log N_\text{lon})$ — negligible compared to
  the VTK operations already in the pipeline.
