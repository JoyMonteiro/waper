# 2026-03-21 — Audit of pre-existing sweeps + next-step plan

Retroactive entry. Captures what the Mar-21 sweeps (run before this log existed)
actually showed and what we should do next. Numbers extracted from
`datasets/figures/sensitivity/{output,gt_output}.log`.

## Question

After the algorithmic refactor (Phase 3, Phase 4, hill-climbing penalty),
which combination of `(edge_pruning_threshold, node_pruning_threshold,
penalty_length_scale_km)` gives physically reasonable RWPs on
`forecast_bust.nc`?

Reference point from Pandey et al. (2020, Fig. 4): a healthy operating
point produces ~3–4 RWPs/timestep, mean edge length ~2000 km, mean RWP
extent in the few-thousand-km range.

## Setup

- **Dataset:** `datasets/forecast_bust.nc` (28 timesteps, 300 mb v-wind)
- **Latitude band:** 20–80 N
- **Base config** (shared across all three sweeps):
  - `clip_value=2`, `extrema_threshold=11`
  - `node_pruning_threshold=20`, `edge_pruning_threshold=3e-5` (when not the swept variable)
  - `max_edge_weight=1`, `track_pruning_threshold=0.3`
- **Sweeps:** one variable at a time, others fixed at the base config above.

## Artifacts

| Sweep | Figure | Histograms | Raw log |
|---|---|---|---|
| GT (3 endpoints, sensitivity.py) | `figures/sensitivity/sweep_gt.png` | `sweep_gt_histograms.png` | `output.log` (Mar 21 15:06) |
| ST (3 endpoints) | `figures/sensitivity/sweep_st.png` | `sweep_st_histograms.png` | same log |
| Penalty (3 endpoints) | `figures/sensitivity/sweep_penalty.png` | `sweep_penalty_histograms.png` | same log |
| GT extended (6 endpoints, gt_sensitivity.py) | `figures/sensitivity/gt_sensitivity.png` | — | `gt_output.log` (Mar 21 19:51) |
| Penalty extended (6 endpoints, penalty_sensitivity.py) | `figures/sensitivity/penalty_sensitivity.png` | — | (no captured stdout) |
| Per-timestep figures, event_winds_abs_1 with `penalty=4000` | `figures/event_winds_abs_1/*` (Mar 21 21:20) | — | — |

## Findings

### GT sweep (edge_pruning_threshold)

ST fixed at 20, penalty at default 2000.

| GT | RWPs/ts | mean_edge km | max_edge km | nodes/RWP | EW extent km |
|---:|--------:|-------------:|------------:|----------:|-------------:|
| 0.005 | 2.4 | 2847 | 9080 | 6.3 | 3899 |
| 0.010 | 2.4 | 2783 | 8474 | 6.4 | 3850 |
| 0.015 | 2.9 | 2538 | 6782 | 5.3 | 3962 |
| **0.020** | **3.6** | **2233** | 5098 | 4.2 | 4083 |
| 0.025 | 3.9 | 1955 | 4118 | 3.6 | 3375 |
| 0.030 | 4.2 | 1751 | 3242 | 3.1 | 2933 |
| 0.040 | 4.1 | 1550 | 2643 | 2.6 | 2165 |
| 0.050 | 3.3 | 1355 | 2471 | 2.3 | 1511 |
| 0.060 | 2.0 | 1036 | 1780 | 2.0 | 952 |
| 0.080 | 1.0 | 865 | 1285 | 2.0 | 785 |

- Below 0.015: GT is effectively off, max_edge floats up to the geometry cap (~9000 km).
- 0.020–0.030: knee region. RWPs/ts rises into the Pandey 3–4 band, max_edge drops to physically plausible 3000–5000 km.
- Above 0.040: over-pruning — RWPs collapse to 2-node fragments (nodes/RWP=2.0), then ts_with_rwp drops.
- Current `WaperConfig` default (`3e-5`) is essentially "no pruning" — the sensible operating range is **0.02–0.03**.

### ST sweep (node_pruning_threshold)

GT fixed at 3e-5, penalty at 2000.

| ST | RWPs/ts | mean_edge km | extent km | ts_with_rwp |
|---:|--------:|-------------:|----------:|-----------:|
| 10 | 3.5 | 2759 | 13059 | 28/28 |
| 15 | 3.1 | 2803 | 12335 | 28/28 |
| **20** | **2.4** | **2847** | 13160 | 28/28 |
| 25 | 2.5 | 2905 | 10452 | 28/28 |
| 30 | 2.4 | 2992 | 8200 | 28/28 |
| 35 | 2.2 | 2993 | 6040 | 28/28 |
| 40 | 1.8 | 2818 | 4597 | 28/28 |
| 45 | 1.4 | 2860 | 3937 | 27/28 |
| 50 | 0.9 | 2424 | 3107 | 23/28 |

- Smooth monotonic decrease in RWPs/ts and extent as ST climbs. No obvious knee.
- `ST=20` (current) sits low; `ST=10–15` is closer to Pandey-band but combined with `GT=3e-5` the edges are too long (mean ~2800 km).
- ST and GT are coupled — ST sweep alone is not informative without revisiting it under `GT≈0.02`.

### Penalty sweep (penalty_length_scale_km)

GT=3e-5 (full sweep) and separately GT=0.03 (penalty_sensitivity.py).

At GT=3e-5, ST=20:

| Lchar | RWPs/ts | mean_edge km | max_edge km | extent km |
|------:|--------:|-------------:|------------:|----------:|
| 500 | 2.1 | 3498 | 9101 | 10948 |
| 1000 | 2.0 | 3276 | 9080 | 13214 |
| 1500 | 2.3 | 3089 | 9080 | 13190 |
| **2000** | **2.4** | **2847** | 9080 | 13160 |
| 3000 | 3.0 | 2672 | 9080 | 11234 |
| 5000 | 3.4 | 2499 | 7301 | 10079 |

At GT=0.03 (from `penalty_sensitivity.png`):
- RWPs/ts climbs from ~2.9 at L=250 to ~5.5 by L=10000 — keeps rising, plateaus around L≈5000–7500.
- Max edge length drops sharply between L=250 → 2000 (from 3500 km plateau down to ~1700 km), then flat.
- Mean E–W extent peaks at L≈3000 km, declines after.
- The vertical line marking "current default 2000 km" is roughly at the elbow.

**Penalty interpretation:** higher L_char → more aggressive splitting → more, smaller RWPs. The hill-climbing penalty is doing its job: without it (low L), big multi-crest clumps merge into a few long RWPs.

### Visual inspection (event_winds_abs_1, penalty=4000)

Re-rendered Mar 21 21:20 with `penalty_length_scale_km=4000`,
`edge_pruning_threshold=0.02`, `extrema_threshold=10`, `ST=20`. Figures live in
`datasets/figures/event_winds_abs_1/` — 81 timesteps × 4 plots each plus
1 track. Not analyzed yet beyond eyeballing the latest few frames.

## Decisions (implicit, not yet codified)

- `visualize.py` was updated to use **GT=0.02, penalty=4000, ST=20** as the
  candidate operating point. These have **not** been promoted to
  `WaperConfig` defaults — defaults still say `GT=3e-5`, `penalty=2000`.
- No experiment yet validates whether the visualize.py settings are good on
  any dataset other than `forecast_bust.nc` (and informally event_winds_abs_1).
- Tried to inspect `v_winds_300mb_nh_2022_2023.nc` as a longer climatology
  sweep target — turns out it's a 239 B empty stub, probably from an
  aborted ERA5 download. Switched the cross-validation plan to
  `event_winds_abs_2.nc` (81 timesteps, summer 1981).

## Next experiments

In rough priority order:

1. **Joint (GT, penalty) grid on both `forecast_bust.nc` and
   `event_winds_abs_2.nc`.** Single-variable sweeps are misleading because GT
   and penalty both reduce RWP size. Run a 2D sweep over
   `GT ∈ {0.015, 0.02, 0.025, 0.03}` × `penalty ∈ {1500, 2000, 3000, 4000,
   5000}` at ST=20, on **both** datasets. Plot RWPs/ts and mean_edge as
   heatmaps per dataset. Pick the operating point that sits inside the
   "3–4 RWPs/ts, 1800–2400 km mean edge" target box on **both** —
   `forecast_bust` alone is only 28 timesteps and overfitting is a real
   risk. `event_winds_abs_2` (81 timesteps, summer 1981) is the cross-check.
   - Note: this will require generalizing the sweep scripts to accept a
     dataset list, or duplicating them.

2. **Re-sweep ST at the chosen (GT, penalty), again on both datasets.** ST
   sweep with GT=3e-5 was uninformative; redo it at the new operating point
   to check that ST=20 is still right (or shift to ST=15 / 25). Same
   per-dataset comparison.

3. **Hold-out qualitative check on `event_winds_abs_1.nc`.** Re-render with
   the chosen (GT, ST, penalty) and visually compare against the current
   Mar-21 21:20 figures (which use GT=0.02, penalty=4000). This is the
   hold-out — it should NOT be used to pick the operating point, only to
   confirm nothing visually regresses.

4. **Restore the long climatology.** Re-download
   `v_winds_300mb_nh_2022_2023.nc` via `download_era5.ipynb` (it's an empty
   239 B stub right now). Once available, run the chosen config on it as a
   final scale check — multi-month behavior may surface issues that don't
   appear in 28- or 81-step files.

5. **Promote validated values into `WaperConfig` defaults.** Once (1)–(3)
   converge, change the defaults in `waper/interface/api.py` and update
   `visualize.py` / sensitivity scripts to use the dataclass defaults
   instead of hard-coding. Add a follow-up entry here.

6. **Visual sanity check on a known event.** Pick one timestep from
   `event_winds_abs_1.nc` with a textbook RWP and compare the polygon
   against the Hovmöller / synoptic interpretation in
   `references/pandey.md`. Currently we have no ground-truth-anchored
   inspection.

7. **Capture stdout reliably.** `penalty_sensitivity.py` has no saved log.
   Wrap the three sweep drivers so each run writes
   `figures/sensitivity/<script>-YYYYMMDDTHHMM.log` automatically.

## Open questions

- The penalty sweep at GT=0.03 doesn't plateau in RWPs/ts even at L=10000 —
  is that physical (we keep splitting), or is the penalty asymptote wrong?
- ST sweep at GT=3e-5 gave nearly flat mean_edge (2800–3000 km across the
  whole sweep). That's suspicious — likely the few surviving long edges
  dominate the mean. Histograms (`sweep_st_histograms.png`) should clarify;
  haven't been read yet.
- Tracking pruning (`track_pruning_threshold=0.3`) and
  `TRACK_DISPLAY_THRESHOLD_KM=8000` in `visualize.py` have never been
  swept. Probably fine, but worth a quick sanity check once identification
  is locked.
