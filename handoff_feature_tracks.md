# Feature-Track Layer — Session Handoff

> **PARTIALLY SUPERSEDED (2026-06-19).** The "Current goal / open problem" and
> "What to try next" sections below are obsolete: the purple-track goal was shown by
> diagnostic to be **unreachable by tuning** — a feature track follows phase, while the
> eastward march is the envelope moving at group velocity. Do **not** retry
> `footprint_fraction` / `min_split_iou` / `max_recover_steps` for it. See
> `assessments/2026-08-07.md` and the do-not-retry register in `assessments/README.md`.
> The algorithm and parameter documentation in this file remains accurate.

## What this is

`waper/tracking/feature_tracks.py` is a post-processing layer that tracks individual RWP
extrema (crests / troughs) as continuous trajectories across time.  It runs **after**
`w.identify_rwps()` and is read-only with respect to the WAPER state.

The primary use case: understand how a Rossby-wave-packet crest propagates (phase velocity),
survives weakening periods, and splits into fragments.

---

## Key files

| File | Role |
|---|---|
| `waper/tracking/feature_tracks.py` | All tracking logic — Feature, TrackStep, FeatureTrack dataclasses; `extract_features`, `match_features`, `track_features`, `feature_tracks_to_dataframe`, `phase_velocity` |
| `tests/test_feature_tracks.py` | 15 unit + integration tests |
| `scripts/feature_tracks_gif.py` | Standalone GIF renderer — run to produce `/tmp/feature_tracks.gif` |
| `datasets/forecast_bust_hourly.nc` | 1-hourly ERA5 v-wind 300 hPa, April 2011, 0.25° NH-only |

The PR for the original implementation is at https://github.com/JoyMonteiro/waper/pull/1.

---

## Algorithm overview

### `extract_features(tsd, time, scalar_name, clip_value, amplitude_threshold, footprint_fraction=None)`

1. Clips the VTK mesh to `|v| >= clip_value` (separately for max and min).
2. Labels connected regions (`topology.identify_connected_regions`).
3. Maps each graph node (extremum) to its VTK region.
4. For regions with **multiple extrema**, runs a **multi-source BFS** flood fill on the mesh
   adjacency: each mesh point is assigned to whichever extremum's wavefront reaches it first.
   Boundaries fall at natural saddle points.
5. **`footprint_fraction`** (new, last added): after BFS assignment, trims the point set to
   only those where `|scalar| >= peak / footprint_fraction`.  This prevents large diffuse
   tails from inflating convex hulls.  GIF currently uses `footprint_fraction=3`.
6. Builds a convex hull in north-polar stereographic metres as the `Feature.footprint`.

### `match_features(prev, curr, max_displacement_deg=None)`

- Returns `{prev_idx: [curr_idx, ...]}` — **one-to-many** (splits allowed).
- Each curr feature claimed by **at most one** prev (no merges).
- Greedy by descending footprint **IoU** (intersection / union in stereographic metres).
- Optional `max_displacement_deg` pre-filters by centroid distance.

### `track_features(features_by_time, max_recover_steps=2, lat_bounds=None, max_displacement_deg=None, min_split_iou=0.2)`

- Seeds one `FeatureTrack` per strong feature at `t=0`.
- Each step: match current heads against `curr_strong`; unmatched heads try `curr_weak`
  (recovery, up to `max_recover_steps` consecutive steps).
- **Splits**: when a head matches multiple curr features, the primary child (highest IoU)
  continues the existing track; each additional child spawns a new `FeatureTrack` with
  `parent_id` set and the parent's history copied in.  Extra children are only kept if
  `IoU >= min_split_iou` — guards against phantom splits from large convex hulls.
- `FeatureTrack.parent_id` is `None` for seeds, set to the parent's `track_id` for splits.

---

## GIF script: current parameters

```python
# Dataset
ds = (raw.rename({"valid_time": "time"})
         .squeeze("pressure_level", drop=True)
         .sel(time=slice("2011-04-12", "2011-04-18"))
         .coarsen(latitude=4, longitude=4, boundary="trim").mean()  # 0.25° → 1°
         .assign_coords(longitude=lambda d: d.longitude % 360)      # -180..180 → 0..360
         .sortby("longitude"))

# Waper
w = Waper(..., clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80, ...)

# Feature extraction
fb = [extract_features(w._time_step_data[t], t, "v", 10, thr, footprint_fraction=3)
      for t in range(nt)]

# Tracking
tracks = track_features(fb, max_recover_steps=4, lat_bounds=(20.0, 80.0),
                        max_displacement_deg=None,   # no cap — relying on IoU alone
                        min_split_iou=0.2)
```

**Why `clip_value=2` for Waper but `10` for `extract_features`?**
Waper needs `clip_value=2` to build the full connectivity graph (including weak edges).
`extract_features` uses `clip_value=10` to define the region for footprint computation —
a tighter threshold so footprints reflect strong signal only.

**Why coarsen to 1°?**
`forecast_bust_hourly.nc` is 0.25° → 361×1440 points/step.  `cluster_extrema` runs
Dijkstra between all extrema pairs in each VTK region: O(n) per pair.  At 0.25° this
took ~70 s/step.  After 4× coarsening (≈ 90×360 = same as the original 6-hourly file)
it takes ~1 s/step and the full 168-step run finishes in ~10 min.

**Why no displacement cap?**
Diagnostic (see session transcript) showed the 8° cap was blocking legitimate 12°/step
jumps when the connected-region topology reorganised between hourly steps.  With `footprint_fraction=3`
and `min_split_iou=0.2` providing enough filtering, relying on IoU alone gives better
continuity without spurious long-range matches.

---

## Current goal / open problem

**Purple track (3rd-strongest feature at t=0) should extend to ~40°E by April 17.**

What's happening: when the polygon supporting the purple feature breaks into three
fragments, the tracker loses continuity.  The `footprint_fraction=3` change was the
most recent attempt — it was still running when this handoff was written.

Known issues that may remain:
1. **Fragment coverage after split**: the child fragment that carries the energy eastward
   may still not have enough IoU with the parent footprint at the split step, depending
   on how tightly the footprints are trimmed.
2. **Recovery through weak steps**: if a fragment briefly falls below `amplitude_threshold`
   (90th percentile), the weak-pool recovery logic must bridge it.  `max_recover_steps=4`
   is currently set.
3. **footprint_fraction tuning**: fraction=3 was suggested but not yet evaluated; fraction=4
   was also proposed.  Tighter fractions give smaller footprints (better spatial precision)
   but lower IoU between consecutive steps (harder matching).

### What to try next

1. **Evaluate the current GIF** (`footprint_fraction=3`, no displacement cap).  Look at
   the purple track specifically — does it now follow the eastward-propagating fragment?
2. **Try `footprint_fraction=4`** if polygons are still too large.
3. **If tracking still breaks at the split**: add a diagnostic that prints the IoU between
   the purple head and all curr features at the split timestep.  Check whether the
   forward fragment has IoU >= 0.2 with the parent — if not, lower `min_split_iou` or
   change the footprint trimming strategy.
4. **Longer-term**: consider energy-weighted centroid matching as a complement to IoU —
   the "energy" that propagates is the group-velocity signal, not the phase-velocity
   centroid.

---

## How to re-run

```bash
# Full GIF (10 min)
~/miniconda3/envs/waper/bin/python scripts/feature_tracks_gif.py

# Tests only (30 s)
~/miniconda3/envs/waper/bin/pytest tests/test_feature_tracks.py -q
```

Output: `/tmp/feature_tracks.gif` — 168 frames, 150 ms/frame, 20–80°N view.

---

## Coordinate-system gotcha

`forecast_bust_hourly.nc` stores longitude as −180…180.  Waper internally converts to
0–360.  The GIF script explicitly applies `.assign_coords(longitude=lambda d: d.longitude % 360)`
before passing to Waper so that `f.lon` values (0–360) align with the contourf background
(also 0–360 after the conversion).  If you use a different dataset, check for this.

---

## Footprint geometry

Footprints are **convex hulls** of the trimmed point set in **north-polar stereographic
metres** (`pyproj CRS: +proj=stere +lat_0=90 +lon_0=0`).  Cartopy reprojection is done
via `ax.add_geometries([footprint], crs=ccrs.Stereographic(central_latitude=90, central_longitude=0))`
— do **not** manually extract vertices and inverse-transform them, that produces straight-line
artifacts near the pole.
