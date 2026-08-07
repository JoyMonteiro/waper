# WAPER experiment log

Chronological record of tuning runs, what they showed, and what to try next.
The figures and stdout logs under `datasets/figures/` are the raw artifacts;
this directory is the narrative around them.

## Format

One file per experiment session: `YYYY-MM-DD-<slug>.md`. Each entry contains:

- **Question** — what we set out to learn
- **Setup** — dataset, base config, sweep variable, command run
- **Artifacts** — paths to figures / stdout logs produced
- **Findings** — what the numbers/figures actually show (with values)
- **Decisions** — config changes adopted, defaults nudged
- **Next** — concrete follow-up experiments

Keep entries terse. Numbers > prose. Link to figures; don't re-describe them.

Qualitative judgement on identification/tracking quality goes in **`assessments/`**
at the repo root (one file per working day), not here — including its do-not-retry
register of rejected approaches. Cross-link when a sweep backs an assessment.

## Conventions

- `forecast_bust.nc` (28 timesteps, April 2011) — fast iteration sweep dataset.
- `event_winds_abs_2.nc` (81 timesteps, June–July 1981) — second sweep dataset for cross-validation. Use alongside `forecast_bust.nc` for any new sweep so we don't overfit tuning to one event.
- `event_winds_abs_1.nc` (81 timesteps, June 1980) — qualitative visual-inspection dataset; keep it independent of tuning sweeps.
- `souders_v_{1,2}.nc` — only 3 timesteps each; too short for sweeps, useful for single-frame regression checks.
- `v_winds_300mb_nh_2022_2023.nc` — **empty stub file (239 B)**. Needs re-download via `download_era5.ipynb` before it can be used as a long climatology sweep.
- Parameters of interest (current defaults in `WaperConfig`, `waper/interface/api.py`):
  - `edge_pruning_threshold` (GT) = `3e-5` (effectively off)
  - `node_pruning_threshold` (ST) = caller-specified
  - `penalty_length_scale_km` = `2000`
  - `cluster_max_eps_km` = `3000`, `cluster_xi` = `0.15` (OPTICS)
  - `extrema_threshold` = caller-specified
  - `lat_gate` = `15.0` — latitude gate (deg) for branch resolution: two candidate
    paths that overlap in longitude and lie within this gap are treated as branches
    of the same wave train. **Never swept.**
  - `hull_method` = `per_node` (`per_node` | `convex` | `concave`) — RWP polygon
    construction.
  - `hemisphere` = `north` (`north` | `south`) — sets the stereographic projection
    and pole; all sweep datasets so far are NH.
- Sweep drivers live in `datasets/`: `sensitivity.py`, `gt_sensitivity.py`, `penalty_sensitivity.py`.

## Index

- [2026-03-21 — Audit of pre-existing sweeps + next-step plan](2026-03-21-initial-audit.md)
