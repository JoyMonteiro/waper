# Feature-Track Layer (SP1) — Design

> Status: design (awaiting review). Date: 2026-06-18.
> Part of the RWP-tracking redesign. This is **Layer 1**; Layer 3 (RWP identity
> over time / group velocity) is deliberately deferred — see "Deferred".

## Problem

RWP tracking currently associates whole **RWP groups** across timesteps (by
polygon/energy overlap). But which crests/troughs form "one RWP" is decided
independently each timestep by the pruning/association logic, so the grouping
**flips between steps**: when two groups merge (e.g. ~150°E in `forecast_bust`)
the group centroid teleports; when a connecting edge is lost the group splits and
the centroid teleports again. Tracking an object whose very identity flips each
step is the root failure — no amplitude threshold fixes it.

The crests and troughs themselves, however, move continuously (phase
propagation). **They are the stable physical objects; the RWP grouping is a
derived, unstable abstraction.** So continuity must live at the feature level.

## Goal & scope

Build a **feature-track layer**: track each individual crest/trough across time as
a continuous trajectory, robust to (a) a feature briefly weakening below
threshold and (b) the RWP grouping merging/splitting around it.

- **Identification is unchanged.** The layer is a read-only post-processing step
  over a completed `identify_rwps()` run.
- **Phase velocity** falls directly out of each feature track.
- **Group velocity** and any persistent *RWP* identity are **out of scope here**
  (Layer 3), to be designed from real feature-track behaviour rather than guessed.

## Key insight enabling this

`tsd.association_graph` is built *before* node pruning
([`api.py:200`](../../../waper/interface/api.py)) and already holds **every**
extremum as a node with `coords=(lon,lat)`, `scalar`, `node_type`
(`"max"`/`"min"`), `cluster_id`, and `cluster_extrema`. So the full unpruned
feature pool already exists — no identification change is needed; the existing
pruned graph/paths/polygons are untouched.

## Architecture & data flow

```
identify_rwps()                          (unchanged)
  → per timestep t: extract_features(tsd)         # all extrema + footprints + strong/weak label
  → track_features(features over all t)           # same-type max-overlap matching + recovery
  → list[FeatureTrack]                            # continuous per-feature trajectories
  → visualise (GIF) / serialise (table)
```

New module: `waper/tracking/feature_tracks.py`. No changes to identification,
the association/pruning code, or the existing tracking graph.

## Components

### 1. Feature extraction — `extract_features(tsd, clip_value, amplitude_threshold)`
For each extremum in `tsd.association_graph`, emit a `Feature`:
`(time, cluster_id, node_type, lon, lat, scalar, footprint, strength)` where
- **footprint** = convex hull (in stereographic metres) of the extremum's
  **region sampled points**, obtained from `get_region_points_and_values` against
  the per-timestep connected regions of the field clipped at a **single global
  `clip_value`** (≈2–3 m/s). A low global clip means a feature's region extends
  out to where |v| falls below `clip_value`, so **strong features get larger
  footprints than weak ones**, and every feature is measured identically. This
  reuses the exact region-extraction path `get_polygon_for_rwp_path` already runs
  internally; the only new work is running it for *all* extrema, not just
  pruned-path nodes.
- **strength** = `"strong"` if `|scalar| ≥ amplitude_threshold` else `"weak"`.
  `amplitude_threshold` is configurable (e.g. an absolute m/s or a |v| percentile)
  and is a **label only** — weak features are kept, not discarded.

The per-timestep clipped regions (positive and negative) are computed once per
step and reused for all that step's extrema.

### 2. Matching across steps — `track_features(features_by_time, ...)`
For each consecutive pair `(t, t+1)`:
- Consider **strong** features only as match anchors; match **same `node_type`**
  (max→max, min→min — a crest never matches a trough).
- Score candidate pairs by **footprint overlap area** (stereographic
  intersection); assign **one-to-one**, greedily by descending overlap (a feature
  does not split at this layer — splitting is a grouping concept, Layer 3).
- A strong `t+1` feature left unmatched is a **birth** (new track). A strong `t`
  feature left unmatched goes to recovery (below).

### 3. Recovery — within `track_features`
If a strong `t` feature has no strong match at `t+1`:
- attempt to match it against the **weak** pool at `t+1` (same type, max overlap);
  if found, the track continues, that step flagged `recovered`;
- a track is **terminated** after `max_recover_steps` (default **2**) consecutive
  recovered/weak steps with no strong re-match, or when the feature's centroid
  leaves `[min_latitude, max_latitude]`.

### 4. Output — `FeatureTrack`
An ordered sequence of `(time, lon, lat, scalar, node_type, recovered)` plus a
stable `track_id`. A list of these is the layer's product; it serialises to a flat
table (one row per feature-step) for analysis, and **phase velocity** is the
centroid displacement along a track over time. (Catalogue integration can follow;
not required for v1.)

### 5. Visualisation — `scripts/feature_tracks_gif.py`
A full-hemisphere PlateCarrée GIF over the v field, each `FeatureTrack` drawn as a
distinct coloured trajectory (with its footprint and a recovered/strong marker),
so real behaviour can be observed — in particular **whether neighbouring features
move together**, which is the evidence Layer 3 needs.

## Key decisions / defaults

| Decision | Choice |
|---|---|
| Feature footprint | convex hull of the extremum's **region sampled points** |
| Region clip | **single global `clip_value`** (footprint scales with amplitude) |
| Match metric | footprint **overlap area**, gated to same `node_type` |
| Assignment | one-to-one, greedy by descending overlap |
| Recovery | match against weak pool; terminate after `max_recover_steps`=2 weak steps |
| Track end | recovery exhausted, or feature leaves the latitude band |
| Threshold role | label only (`strong`/`weak`); weak features retained for recovery |

## Testing / validation

- **Unit (synthetic features, no full pipeline):**
  - two same-type footprints shifted between two steps → matched into one track;
  - a feature present at t, absent (no strong) at t+1 but overlapping a weak
    feature, present again at t+2 → one continuous `recovered` track;
  - a feature whose centroid crosses `max_latitude` → track ends there;
  - a max footprint overlapping a min footprint → **not** matched.
- **Empirical (`forecast_bust`):** a strong crest produces a continuous track over
  many steps; across the 150°E merge, the individual feature tracks **do not**
  break or teleport (contrast the current RWP-group tracking).

## Deferred (Layer 3 — explicitly not in this spec)

- Persistent **RWP identity** over time and **merge/split events** derived by
  comparing per-step groupings *through* the feature tracks.
- **Group velocity** (needs the grouping layer; per-feature tracks give phase
  velocity only).
- The amplitude `threshold` value itself — to be swept/chosen from real tracks.
- Tie-breaking when several same-type extrema share one connected region at the
  global clip (identical footprints): start with amplitude/proximity tie-break;
  revisit once observed on real data.

## Out of scope

- Any change to identification, pruning, the association/pruned graphs, RWP
  polygons, or the existing energy-overlap tracking graph.
- Catalogue/serialisation schema changes (optional follow-up).

## Relationship to prior work

Builds on the `energy-weighted-tracking` branch (energy-weighted centroid,
energy overlap, spherical centroid). This feature-track layer is the intended
*replacement substrate* for RWP-group overlap association; the energy work
remains valid as the per-step "which feature is the energetic core" weighting.
