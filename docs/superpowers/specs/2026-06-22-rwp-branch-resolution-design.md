# Design: Latitude-gated branch resolution for RWP path selection

**Date:** 2026-06-22
**Status:** Approved design — pending implementation plan
**Topic:** Eliminate spurious spatially-overlapping / length-1 RWPs in
`get_ranked_paths` by enforcing in-band zonal exclusivity and resolving graph
branches in favour of the stronger branch.

---

## 1. Problem

WAPER's RWP identification (`waper/identification/rwp_graph.py`) produces three
related artifacts, visible on the April 2011 forecast-bust case:

1. **Length-1 "RWPs"** — a single extremum emitted as an RWP.
2. **Spatially-overlapping RWPs** — two RWPs whose features occupy the same
   longitude band at nearly the same latitude (e.g. a strong wave train plus a
   second RWP made of the weak features it didn't claim), so their footprints
   overlap the same physical feature.
3. **Crossing spines** — two RWPs whose wave-train spines cross.

## 2. Root cause

`get_ranked_paths(assoc_graph, max_weight)` builds RWPs by **greedy
node-disjoint path selection**: it enumerates every eastward simple path, sorts
by summed edge weight, and keeps any path whose *nodes* are not already used. It
enforces only node-disjointness — never spatial coherence. Consequences:

- The strong train claims the strong features; leftover features form a
  spurious weaker RWP that interleaves it (artifact 2 and 3).
- A leftover feature with no remaining partners becomes a length-1 path
  (artifact 1), made possible by a specific bug: the `source`/`sink` double loop
  does not skip `source == sink`, and `is_to_the_east(lon, lon)` returns
  `0 > 0 == False`, so `nx.all_simple_paths(g, n, n)` yields `[[n]]`.

"Stronger RWP wins a contested feature" is *already* handled by weight-ordered
node-disjoint selection — the fix only concerns what happens to the loser's
leftover features.

## 3. Definitions

- **Longitude span of a path** — paths are monotonic-east (`_is_monotonic_east`),
  so the westmost node is `path[0]` and the eastmost is `path[-1]`. The span is
  the eastward arc `(start = lon(path[0]), length = Σ _longitude_separation(consecutive))`,
  evaluated mod 360.
- **Arcs overlap** — two eastward longitude arcs `[s_a, s_a+L_a]` and
  `[s_b, s_b+L_b]` (mod 360) share any longitude.
- **Latitude ranges of two paths are within the gate** — the gap between the two
  `[min_lat, max_lat]` ranges is `≤ lat_gate` (overlapping ranges → gap 0).
- **In-band interleave** — longitude arcs overlap **and** latitude ranges are
  within `lat_gate`. This is the test for "these two paths are the same
  waveguide and must not coexist." Different-waveguide packets (e.g. subtropical
  vs polar jet) have latitude ranges further apart than `lat_gate` and are
  therefore *not* interleaving — they may share longitudes.
- **Branch strength** — at a junction the competing arms are sub-paths; an arm's
  strength is the **summed edge weight** of that sub-path. (A lone orphan's arm
  is the single connecting edge.) This metric is consistent with how
  `get_ranked_paths` already ranks paths, and is the parameter the acceptance
  test validates; if the test fails it is the first thing to reconsider.

## 4. Algorithm

An RWP is and remains a **simple, monotonic-east, sign-alternating path**. No
node belongs to two RWPs. No RWP has length 1.

### Pass 1 — select with in-band zonal exclusivity (modify `get_ranked_paths`)

1. **Skip `source == sink`** in the path-enumeration double loop (removes length-1
   paths at the source).
2. Build candidate paths and sort by summed weight as today.
3. Greedy selection, with one added rejection test:
   ```
   for path in sorted_paths:
       if not set(path).isdisjoint(used_nodes):      # node-disjoint (existing)
           continue
       if any(_paths_interleave_in_band(g, path, ap, lat_gate) for ap in top_paths):
           continue                                   # NEW: in-band exclusivity
       top_paths.append(path); used_nodes.update(path)
   ```
   A path rejected by the interleave test is **not** emitted as its own RWP; its
   nodes become orphans for pass 2. A subtropical path (latitude > `lat_gate`
   from every accepted path) is *not* rejected and survives as its own RWP — this
   is why a global rule is wrong and the gate is required.

### Pass 2 — reassign orphans by branch resolution (new `reassign_orphans`)

Orphans = nodes of the pruned graph in no accepted path. Iterate to a fixpoint
(bounded by a max-iteration guard):

For each orphan `o`:
- **Candidate junctions** — pruned-graph neighbours `nb` of `o` that are in some
  accepted RWP and whose latitude is within `lat_gate` of `o`. (Edges only
  connect a max cluster to a min cluster, so `o` and `nb` are opposite types and
  alternation is preserved.) If none → leave `o` orphaned (dropped at the end).
- Pick the strongest candidate edge `(o, nb)`.
- `o` lies on one side of `nb` (west or east, by `is_to_the_east`); it competes
  with **`nb`'s existing arm on that same side**. Compare branch strengths:
  - **`o`'s arm weaker** → drop `o` (mark permanently dropped; do not reprocess).
  - **existing arm weaker** → drop that arm: remove its nodes from the RWP (they
    re-orphan and may re-attach on a later iteration — cascade), and splice `o`
    in as `nb`'s neighbour on that side. The RWP stays a simple monotonic-east
    path.

After the fixpoint: any node still orphaned (no in-band in-RWP neighbour, or it
lost its contest) is dropped — it is not emitted as an RWP. This yields no
length-1 RWPs and no in-band spatial overlap.

### Wiring

- `lat_gate` is a new `WaperConfig` field, default **15.0** (degrees), exposed as
  a `Waper.__init__` kwarg `lat_gate=15.0`.
- `get_ranked_paths(assoc_graph, max_weight, lat_gate=15.0)` does pass 1 then calls
  `reassign_orphans(assoc_graph, top_paths, lat_gate)` and returns the refined
  paths. `_identify_rwps` ([api.py:213](waper/interface/api.py:213)) passes
  `config.lat_gate`. The return type (list of node-list paths) is unchanged, so
  all downstream consumers (`get_polygon_for_rwp_path`, `io/extract.py`) are
  unaffected.

## 5. Acceptance test (the gate)

On `datasets/forecast_bust_hourly.nc`, preprocessed to 1° (as in
`scripts/feature_tracks_gif.py`), timestep **2011-04-04 23Z (`t = 95`)**, run with
`node_pruning_threshold=20, edge_pruning_threshold=0.02, min_latitude=20,
max_latitude=80`:

- **No length-1 RWPs** — every returned path has ≥ 2 nodes.
- **No in-band spatial overlap** — for every pair of returned RWPs whose latitude
  ranges are within 15°, their longitude arcs do not overlap.
- **The weak interleaving RWP is gone** — specifically the 2-node path
  `min@~144°→max@~212°` and the length-1 `min@~1°` present in the current output
  are absent.
- **The dominant western train is preserved** — the strongest path (the ~10-node
  train spanning ~225°→44°) is still returned, with its strong nodes intact.

Plus invariant checks runnable on synthetic graphs (fast unit tests):
- A hand-built graph with a known branch resolves to the stronger arm.
- A subtropical path (latitude > 15° from a midlatitude path, overlapping
  longitudes) is **kept** as its own RWP.
- `source == sink` never yields a length-1 path.

## 6. Out of scope (YAGNI)

- Variance-defined waveguide bands — the relative 15° latitude gate is the first
  cut; band detection is a later refinement.
- Seeding *new* RWPs from orphan clusters — unnecessary, because genuine separate
  trains (subtropical) are kept by pass 1; pass-2 orphans are only ever the
  losers of in-band contests, which should be absorbed or dropped, not re-seeded.
- The antimeridian Hilbert seam in `compute_rwp_envelope` (a separate
  diagnostic-side issue the user has deprioritised).
