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
| `implementation/waper_refactoring_spec.md` | Phases 0–9 — the whole spec. Phases 6–9, its last live remainder, were closed by `implementation/2026-08-12-engineering-backlog.md` | Ruff `D` gate in `pyproject.toml` (`[tool.ruff.lint.pydocstyle]`, `D` in `select`) with `waper/interface/api.py`, `explorer.py`, `waper/io/*` documented (8.2); `waper/interface/projections.py` + `tests/interface/test_projections.py` (7.1/7.2/7.4); `results/benchmarks.md` + `tests/test_benchmark.py` (6.3); `Waper.identify_rwps(n_jobs=…)` + `tests/test_parallel_identify.py` (9.2); `WaperConfig.to_yaml`/`from_yaml` + `Waper.from_config` + `tests/test_config_yaml.py` (9.4). 7.3 (Hovmöller) deferred by Joy 2026-08-12; 9.3 retired as superseded by `waper/io/` |
| `implementation/2026-08-07-housekeeping-backlog.md` | All three tiers, closed 2026-08-07 | Tier work is in the tree; its "needs Joy's input" register was carried forward verbatim rather than lost — see the note below |

### Caveat on branch resolution

That last one was implemented and reviewed on `energy-weighted-tracking` before it
had a remote copy. **Merged and pushed as of 2026-08-07** — it is on `main`
(`31f6c72`, `30badc1`, `f80f94d`). The four minor findings deferred during review
were all closed in Tier 1 of the housekeeping backlog (`_arc_bins` off-by-one fixed
with a regression test, `_lat_ranges_within` renamed, the early-return test added,
the test-file import hoisted) — see `progress.md` for the original adjudication.

---

## Retired as superseded

| File | Superseded by |
|---|---|
| `transcripts/2026-architecture-assessment-session.md` | A 7,568-line raw chat transcript of the architecture-assessment session. Its conclusions were written up as `waper_refactoring_spec.md`, itself now archived here. Kept only as provenance for that spec |

---

## What deliberately stayed active

Both plans archived above on 2026-08-13 carried a register of items that are **not** closed:
the housekeeping backlog's "Explicitly NOT here — these need Joy's input" list (operating-point
promotion, the group-velocity question, envelope-weighted edge pruning, the ERA5 re-download,
and the two uncommitted files). That register lives on, reproduced verbatim, in
`superpowers/plans/implementation/2026-08-12-engineering-backlog.md`. Read it there before
proposing engineering cleanup — archiving these two files retired the *work*, not the register.

- **`design/validation_strategy_plan.md`** and **`implementation/phase0_implementation_plan.md`**
  — the Phase 0 gate has never been run; every downstream layer depends on it.
- **`design/western_disturbance_validation_plan.md`**, **`design/regime_rwp_structure_plan.md`**
  — Layer 5 applications, not started, both blocked on Phase 0.
- **`design/clustering_investigation_plan.md`** — OPTICS parameter tuning. Its
  header says "pick up after Phases 4+ are complete"; that condition is now
  **satisfied**, so this is unblocked rather than stale.
- **`design/architecture_and_algorithm.md`** — reference documentation, not a plan.
  Arguably belongs under `docs/`.
