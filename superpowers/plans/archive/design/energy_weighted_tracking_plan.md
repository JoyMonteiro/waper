# Energy-Weighted RWP Tracking — Design

> Status: design (awaiting review). Author handoff date: 2026-06-15.
> Companion implementation plan to follow via `writing-plans`.

## Problem

RWP tracking associates features across timesteps by **binary overlap** of the
rasterized RWP polygon, with `weight = overlap_pixels / max(area_prev, area_curr)`
([`waper/tracking/tracking_graph.py`](../../../waper/tracking/tracking_graph.py)),
and uses the amplitude-weighted centroid (`weighted_longitude/latitude`) as the
track position.

The RWP polygon is the **whole packet envelope** (~30–60° wide). The packet's
group propagation is only a few degrees per 6 h step — far smaller than the
envelope. Consequences:

- **Overlap saturates** (~1.0 every step), so the association can't resolve
  motion and a large envelope overlaps several neighbours → ambiguous links.
- **The centroid washes out**: averaging over the entire envelope (and over many
  crests/troughs that grow upstream and decay downstream) keeps the position
  near the envelope centre. The track *looks frozen even though energy
  propagates through the packet.*
- **Every crest/trough contributes equally**, regardless of amplitude, so weak
  peripheral features drive the association as much as the energetic core.

## Goal

Make tracking follow the **energetic core** of the RWP — the high-amplitude
crests/troughs where the packet's energy is concentrated — so that motion is
resolvable and weak features stop driving the association. Keep the existing
merge/split tracking machinery.

Non-goal: the polygon for visualization (explicitly out of scope per user).

## Approach: energy-density tracking

Energy of the meridional-wind packet is `∝ v²` (both crests `v>0` and troughs
`v<0` are energetic). Instead of a binary mask, drive tracking with an **energy
density** that is peaked at the strong cores. Three coordinated changes, all
reusing the current rasterize → quadtree → `merge` pipeline:

### 1. Energy field per RWP
Produce a per-pixel **energy raster** co-registered with the existing feature-id
raster (same stereographic grid from `rasterize_all_rwps`). Each pixel owned by
an `rwp_id` carries an energy value (default `v²`), so the strong crests/troughs
dominate and the periphery is ~0.

- **Default source:** sample `v²` on the raster grid (inverse-project each pixel
  → interpolate the lon/lat `v` field), masked to the RWP footprint.
- **Fallback (decide in implementation):** synthesize the field from node
  amplitudes — sum of `amplitude²`-weighted kernels centred on each extremum —
  if grid sampling proves awkward/costly.

### 2. Energy-weighted association
In `compute_size_features` and `merge`, replace pixel **counts** with energy
**sums**:
- `feature_size  = Σ energy over the feature's pixels`
- `overlap_size  = Σ energy over the overlapping pixels`
- `weight = overlap_size / max(feature_size_prev, feature_size_curr)`

Because the periphery contributes ~0 energy, the overlap is dominated by the
core; when the core moves, the energy-overlap drops appreciably even though the
broad footprints still overlap. Merge (in-degree>1) / split (out-degree>1)
structure is preserved — only the per-feature/per-overlap accounting changes.

### 3. Energy-weighted centroid (track position)
Change the weighted centroid in `get_polygon_for_rwp_path`
([`waper/tracking/rwp_polygon.py`](../../../waper/tracking/rwp_polygon.py)) from
`|v|` weights to `v²` (energy) weights, so `weighted_longitude/latitude` follow
the dominant core rather than the envelope mean.

## Components & interfaces

| Unit | Change | Depends on |
|---|---|---|
| `rwp_polygon.get_polygon_for_rwp_path` | energy (`v²`)-weighted centroid | node `scalar` values |
| `rwp_polygon` (new) `rasterize_energy` | co-registered energy raster per `rwp_id` | footprint polygons + `v` field (or node amplitudes) |
| `quadtree.compute_size_features` / `merge` / `construct` | energy **sums** instead of pixel counts | energy raster |
| `tracking_graph.build_tracking_graph` | edge weight from energy overlap; node coords = energy centroid | above |
| `api._identify_rwps` | pass the `v` field (or amplitudes) needed to build the energy raster | existing per-timestep data |

Distance-based pruning (`prune_tracking_graph`) is unchanged.

## Data flow

identification (per timestep) → footprint polygons + energy-weighted centroid →
`rasterize_all_rwps` (feature-id raster) **+** new energy raster → quadtree
(carrying energy) → `merge` across consecutive timesteps → energy-overlap weight
→ tracking DiGraph → existing prune/track extraction.

## Testing / validation

- **Unit — motion becomes visible:** synthetic two-timestep packet shifted east;
  assert the `v²`-weighted centroid displacement is larger / non-trivial vs the
  current `|v|`-weighted centroid (which barely moves).
- **Unit — energy overlap discriminates motion:** a moved packet yields a
  clearly lower association weight than a stationary copy (binary overlap would
  not).
- **Unit — periphery ignored:** adding a low-amplitude peripheral crest does not
  materially change the association weight or the centroid.
- **Regression:** existing tracking/merge/split tests still pass; merge/split
  topology on the `two_timestep_field` fixture is unchanged in structure.
- **Empirical:** on `datasets/forecast_bust.nc`, a known packet's track shows
  monotonic eastward centroid motion (group propagation) instead of a
  near-stationary track.

## Defaults & open questions (resolve during implementation)

- Energy definition: `v²` (default) vs node `amplitude²`.
- Energy raster source: grid-sampled `v²` (default) vs synthesized kernels.
- Whether energy-overlap alone suffices or a light eastward-displacement prior is
  also needed (deferred; revisit after measuring tracks).

## Out of scope

- Visualization polygon shape (corridor/concave/convex) — dropped per user.
- Group-velocity / phase-vs-group motion model — possible follow-up.
- The identification-side `5–7` spurious-edge / V-topology grouping issue —
  tracked separately.

## Relationship to other work

- Builds on the current branch, which already contains the identification
  `_path_circles_globe` region-wrap fix and the explorer polar-stereo fix.
- Tracking quality feeds the validation program
  ([`validation_strategy_plan.md`](validation_strategy_plan.md)) and the
  merge/split facts noted there.
