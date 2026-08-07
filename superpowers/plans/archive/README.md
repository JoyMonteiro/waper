# Archived plans

Plans in this directory are **retired**: the work they specify has landed in the
codebase, or they have been superseded by a later document. They are kept for
provenance — to answer "why is the code like this?" — not as work queues.

> **Do not judge a plan by its checkboxes.** Several of these were executed via
> subagent-driven development, where the implementer commits code without ticking
> the source plan. `2026-06-15-energy-weighted-tracking.md` reads 0/34 and is
> fully shipped. The "Evidence" column below is the real status.

Everything here was moved with `git mv` on 2026-08-07; history is intact and any
file can be restored by moving it back.

---

## Retired because the work shipped

| Plan | Delivered | Evidence in tree |
|---|---|---|
| `implementation/phase_0_2_execution_plan.md` | Refactoring spec Phases 0 & 2 — dead-code removal, logging, `is_to_the_east`, bare-`except` fixes | 5/5 ticked |
| `implementation/2026-03-17-phase-3.md` | Identification improvements: clustering replacement, weighted centroid, path ranking, wraparound pruning | `topology.py` uses **OPTICS** (`cluster_extrema`, `max_eps_km`/`min_samples`/`xi`). Note the plan specifies DBSCAN — that was itself superseded by spec Task 3.7 (OPTICS) before it ever became the final state |
| `implementation/2026-03-20-phase-4.md` | Tracking improvements: per-node hulls, Southern Hemisphere, DAG DP path extraction, quadtree merge decoupling | `api.py:48-49` `hull_method="per_node"`, `hemisphere`; `tracking_graph.py:136` topological-sort DP replacing `all_simple_paths` |
| `implementation/2026-03-21-hill-climbing-penalty.md` | Fractional-descent penalty replacing the dead zero-crossing check | `topology.py:142-171` fractional descent → `penalty_km = f * penalty_length_scale_km`; `api.py:50` `penalty_length_scale_km=2000.0` |
| `implementation/2026-06-06-serialization-query-viz.md` + `design/serialization_query_viz_plan.md` | Serialization, query, and visualization layer | 88/88 ticked |
| `implementation/2026-06-15-energy-weighted-tracking.md` + `design/energy_weighted_tracking_plan.md` | Energy-weighted RWP tracking — the work this branch is named for | `waper/tracking/energy_overlap.py`; `rwp_polygon.py` `energy_disks()` / `rasterize_energy()`; amplitude²-weighted centroid. Commits `d75c04a`..`5b89416` |
| `implementation/2026-06-18-feature-track-layer.md` + `design/feature_track_layer_plan.md` | Feature-track layer (SP1) — per-extremum trajectory tracking | `waper/tracking/feature_tracks.py`, `tests/test_feature_tracks.py`. Merged as PR #1. Commits `3567f67`..`13e3a5f` |
| `implementation/2026-06-20-rwp-method-comparison.md` + `specs/2026-06-20-…-design.md` | Three-method agreement study (WAPER vs Zimin envelope vs node-amplitude vs edge-pruning) | `scripts/method_comparison/`, `tests/test_method_comparison.py`, `results/method_comparison_sweep.csv` (720 timesteps). Commits `b86fb34`..`98074b4` |
| `implementation/2026-06-22-rwp-branch-resolution.md` + `specs/2026-06-22-…-design.md` | Latitude-gated RWP branch resolution — in-band zonal exclusivity + orphan reassignment | `rwp_graph.py` helpers, `api.py:52` `lat_gate=15.0`. Commits `f7849bd`..`fea7757`. All 5 SDD tasks + final whole-branch review complete (`.superpowers/sdd/progress.md`) |

### Caveat on branch resolution

That last one is **implemented and reviewed but not merged or pushed** — the code
lives only on the `energy-weighted-tracking` branch, which has no remote copy. The
plan is retired because its tasks are executed; the merge is outstanding git work,
not outstanding plan work. Four minor findings were consciously deferred rather
than fixed (`_arc_bins` off-by-one, `_lat_ranges_within` naming, a missing
early-return test, test-file import style) — see `progress.md` for the
adjudication.

---

## Retired as superseded

| File | Superseded by |
|---|---|
| `transcripts/2026-architecture-assessment-session.md` | A 7,568-line raw chat transcript of the architecture-assessment session. Its conclusions were written up as `implementation/waper_refactoring_spec.md`, which is still active. Kept only as provenance for that spec |

---

## What deliberately stayed active

- **`implementation/2026-08-07-housekeeping-backlog.md`** — open housekeeping and
  engineering items that need no scientific input, plus an explicit list of the ones
  that *do* and must not be actioned as housekeeping.

- **`implementation/waper_refactoring_spec.md`** — Phases 0–4 are done (see the
  table above), but **Phase 5 (VTK → PyVista/SciPy) has not started**:
  `waper/identification/{utils,max_min,topology}.py` still import VTK and there is
  no `scipy.sparse` shortest-path in `topology.py`. 32/126 boxes understates it,
  but the live remainder is real.
- **`design/validation_strategy_plan.md`** and **`implementation/phase0_implementation_plan.md`**
  — the Phase 0 gate has never been run; every downstream layer depends on it.
- **`design/western_disturbance_validation_plan.md`**, **`design/regime_rwp_structure_plan.md`**
  — Layer 5 applications, not started, both blocked on Phase 0.
- **`design/clustering_investigation_plan.md`** — OPTICS parameter tuning. Its
  header says "pick up after Phases 4+ are complete"; that condition is now
  **satisfied**, so this is unblocked rather than stale.
- **`design/architecture_and_algorithm.md`** — reference documentation, not a plan.
  Arguably belongs under `docs/`.
