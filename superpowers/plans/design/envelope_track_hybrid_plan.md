# Envelope-Track Hybrid: Group-Velocity Tracking with Phase Attachment

> **Status:** design, awaiting implementation plan. Decided 2026-08-13; resolves the
> "group-velocity question" that has blocked the feature-track line since 2026-06-19
> (see `assessments/README.md` do-not-retry register and `assessments/2026-08-07.md`).

## 1. The problem this solves

WAPER has two validated but mismatched halves:

- **Identification** (the Pandey graph) is precise and differentiated: against the Zimin
  envelope it shows detection agreement 1.0 through the usable GT band and WAPER-only
  excess area of 0.04 (`results/method_comparison_sweep.csv`) — and it is the only
  method that preserves phase. This is the scientific core and it stays untouched.
- **Tracking** answers the wrong question. The feature-track layer
  (`waper/tracking/feature_tracks.py`) is *correct* — matching held IoU 0.84–0.94 in the
  purple-track diagnostic — but a feature track follows **phase**. The eastward march we
  care about (e.g. "the April-2011 packet reaches ~40°E by Apr 17") is the envelope
  moving at **group** velocity, carried by a succession of distinct troughs with no
  footprint lineage between them. No graph-tracking refinement can express that answer;
  this is confirmed by diagnostic and registered as do-not-retry.

Reading Souders et al. (2014b) closely shows how envelope-only trackers "work": WPA is
smoothed hard (T21 spatial + 24-h running mean) before tracking, they track prominent
**envelope local maxima** (not footprint centroids — this is why their centroids don't
zigzag), the search box is eastward-biased (30°W/90°E), "significant" is defined to
exclude stationary/decaying features, and even then hybrid methods run ~90% detection
with ~20% false alarms concentrated at merges/splits. Their machinery is legitimate for
packet *identity*; it discards exactly the phase structure our science needs.

**Design decision:** phase and group are two different physical objects. Track both,
each with the tool suited to it, and join them:

> The **smoothed envelope track** is the spine (packet identity, genesis→lysis, group
> velocity). The **WAPER graph components** are the anatomy attached to that spine per
> timestep (which trough, where, how strong). Graph-to-graph tracking is demoted to
> what it is good at: short-horizon phase continuity.

This is option (a) of the open decision ("build an envelope/group-velocity primitive"),
implemented almost entirely by re-aiming components that already exist and are tested.

## 2. Architecture

```
v(λ,φ,t) ──► identify_rwps() ──► graph components + RWP polygons     (unchanged)
    │
    └─► WPA(λ,φ,t) = Zimin envelope → T21 truncate → 24-h running mean
              │                                       (exists: masks.py, commit 98074b4)
              ▼
        envelope features per timestep  (threshold + connected components → Feature)
              │
              ▼
        envelope tracks  (existing match_features / track_features, unchanged)
              │
              ▼
        attachment: (time, rwp_id) → envelope_track_id   (max footprint overlap)
```

### 2.1 Component 1 — envelope field (promote, don't rewrite)

Promote from `scripts/method_comparison/masks.py` into `waper/identification/envelope.py`:

- `compute_rwp_envelope(v, wavenumber_range=(3, 11))` — Zimin 2003 zonal Hilbert
  envelope (already written).
- `t21_truncate(field, ntrunc=21)` — spherical-harmonic truncation (already written,
  incl. the equator-reflection path for hemisphere-only data).
- `temporal_running_mean(stack, hours=24, ...)` (already written, step inferred).

Promotion work is packaging + generalization, not algorithm work: the script versions
assume `hemisphere="north"` defaults; the promoted versions must take the same
hemisphere-aware treatment the viz layer got in `a7e01d2`. `masks.py` then imports from
`waper` so the method-comparison sweep and the tracker share one implementation.

*Deliberately deferred:* the Zimin **2006 streamline** variant (what Souders actually
used). Start zonal; T21 smoothing already blunts the tilt-fracturing problem, and the
tracker's existing gap-bridging (`max_recover_steps`) covers short dropouts. Add the
streamline transform only if fragmentation is observed in practice (§6 risk 1).

### 2.2 Component 2 — envelope features and tracks (reuse the proven tracker)

New module `waper/tracking/envelope_tracks.py`:

- `extract_envelope_features(wpa_2d, threshold, min_area_km2, hemisphere) -> list[Feature]`
  — threshold the smoothed WPA, label connected regions (`scipy.ndimage.label` with
  longitude wraparound), drop regions below `min_area_km2`, and build each region's
  stereographic footprint with the existing `_footprint_from_region`. Record per
  feature: centroid, footprint, peak WPA, area.
- Tracking: **feed these features to the existing `match_features` / `track_features`
  unchanged.** That layer is proven sound; it was tracking the wrong field. On the
  T21+24h WPA field, downstream development is a smooth eastward drift of one blob —
  the regime the IoU matcher is best at.
- `group_velocity(track, dt_hours) -> float` — centroid great-circle zonal speed over
  the track, the sibling of the existing `phase_velocity`. This makes the
  `c_g > c_p` invariant of the validation plan (Layer 1) computable for the first time.

Defaults (all in `WaperConfig`, all Souders-anchored, all sweepable):

| Parameter | Default | Source / note |
|---|---|---|
| `envelope_wavenumbers` | (3, 11) | Souders §2; already the masks.py default |
| `envelope_truncation` | T21 | Souders (tested T15–T42) |
| `envelope_smoothing_hours` | 24 | Souders (tested 0–48 h) |
| `envelope_threshold` | 14.0 m/s | Souders min WPA. **Must be swept down (~8–14) for the subtropical/WD regime** — same reasoning as WD plan §4 |
| `envelope_min_area_km2` | ≈ Souders' 40 × 2.5° pixels | convert, don't copy the pixel count |
| significant-track filter | ≥ 2 days, ≥ 40° east | *reporting* filter only — never applied during tracking, so stationary/decaying packets remain inspectable (unlike Souders) |

### 2.3 Component 3 — attachment (the ~100 lines of genuinely new code)

`attach_rwps_to_envelope_tracks(rwp_polygons_by_time, envelope_tracks) -> dict[(time, rwp_id), track_id]`

Per timestep: intersect each WAPER RWP footprint with each envelope feature footprint
(both already live in the same stereographic CRS); assign the RWP to the envelope track
whose feature covers the largest fraction of the RWP's area, provided that fraction
exceeds `attach_min_fraction` (default 0.3); break exact ties by higher feature peak
WPA. Otherwise the RWP is **unattached** — a
legitimate state (sub-threshold packet, envelope dropout) that must be reported, not
forced.

Surface the result through `waper/io/`: `extract_rwps()` and the track extractors gain
an `envelope_track_id` column (nullable). Trough succession — the thing single-extremum
tracking could not express — now falls out for free: it is consecutive timesteps of one
envelope track containing different attached graph nodes.

### 2.4 What is *not* changing

- Identification: no changes to extrema, clustering, association graph, or pruning.
- The RWP-level tracking graph (`tracking_graph.py`) and the feature-track layer keep
  their current roles for short-horizon phase continuity; nothing is deleted.
- No eastward search-box prior, no prominence-based subobject machinery (Souders'
  TRACK/H1/H2 conflict-resolution rules): the IoU matcher on a heavily smoothed field
  should not need them. Add only if the acceptance test says otherwise (YAGNI).

## 3. Acceptance criteria

1. **Synthetic (unit, fast):** extend the tracking benchmark's synthetic field
   (amplitude 30 — see `results/benchmarks.md` for why not 20) to a modulated packet
   whose carrier moves at c_p and envelope at c_g ≠ c_p. Assert `group_velocity` of the
   envelope track ≈ c_g and `phase_velocity` of a feature track ≈ c_p, and that every
   in-packet extremum attaches to the single envelope track.
2. **The purple-track goal (integration, the reason this design exists):** on
   `forecast_bust_hourly.nc` (coarsened 4×, lon %360-sorted — see handoff perf note),
   an envelope track continuous from the Atlantic sector (~300–320°E) around Apr 11–12
   reaches ≥ 40°E by Apr 17, and ≥ 3 distinct graph troughs attach to it over its life.
3. **No regression in expressiveness:** phase tracks on the same case still terminate
   at the purple seed's physical decay (t=11) — that behavior was *correct* and must
   not be "fixed" by this work.
4. Existing suite stays green (`pytest -m "not slow"`).

## 4. Testing strategy

- Unit: envelope helpers already have tests (`tests/test_method_comparison.py`) — they
  move/extend with the promotion. New tests for `extract_envelope_features` (wraparound
  region, min-area rejection, hemisphere handling), `group_velocity`, and attachment
  (clean case, tie, unattached case).
- Integration: criteria 1–2 above as marked tests (`slow` for the forecast-bust run).
- Science-facing: record the first forecast-bust run as a `datasets/experiments/`
  session entry and an `assessments/` entry (per existing conventions).

## 5. Open questions folded into the design (no longer blocking)

- **Envelope-weighted edge pruning** (`envelope_segmentation_proposal.md`): the WPA
  field is now computed per timestep anyway, so the proposal's simplified variant
  becomes a cheap optional refinement of *identification* — but it targets over-merging,
  which `lat_gate` branch resolution (shipped, unmeasured) may already fix. **Hold until
  the `lat_gate` sweep runs;** retire the proposal if the sweep shows over-merging
  controlled, revisit (trivially, envelope in hand) if not.
- **Operating point:** envelope-threshold tuning must happen on top of *one* base
  configuration. Resolving the GT=0.02/penalty=4000/ST=20 (visualize.py) vs
  GT=3e-5/penalty=2000 (WaperConfig) divergence is a prerequisite for the tuning step,
  though not for building the machinery.

## 6. Risks

1. **The forecast-bust case is adversarial for Hilbert envelopes** — Ghinassi et al.
   (2018) showed the transform struggles with exactly this non-wavelike Atlantic RWP.
   T21+24h smoothing and `max_recover_steps` gap-bridging are the mitigations; if the
   envelope still fragments, the fallback is the Zimin-2006 streamline transform
   (§2.1), not tracker heroics. If the acceptance test fails even then, that is a
   scientifically honest result: record it in `assessments/` before adding machinery.
2. **Inherited envelope arbitrariness.** Threshold, min-area, and merge/split ambiguity
   at packet collisions are where Souders' ~20% false alarms live. We accept this for
   the identity layer because every ambiguous case remains inspectable through the
   attached phase structure — an audit capability envelope-only methods lack.
3. **Hemisphere generality.** The promoted envelope code must not fossilize the NH-only
   assumptions of `masks.py`; SH support is required by the Layer-2 validation targets.

## 7. Disposition of existing plans

Requested explicitly: what runs independently of this design, and what retires.

### Independent — can run now, in parallel, unaffected

| Item | Why independent |
|---|---|
| **Merge `engineering-backlog` → `main`** (9 commits, 0 behind, suite green) | Pure housekeeping; this design builds on that branch's code |
| **`lat_gate` on/off sweep** (assessment Next #2) | Identification-side; also *gates* the envelope-edge-pruning decision (§5) — run it early |
| **T21-filtering result write-up** (assessment Next #1) | The arrays exist (`results/per_timestep_iou_filtered.npy`, Δ≈0.001 vs unfiltered); needs recording in `datasets/experiments/`, nothing more |
| **Operating-point promotion** (assessment Next #4) | Joy's sign-off + a YAML/defaults change; prerequisite only for this design's *tuning* step (§5) |
| **Phase 0 gate** (`phase0_implementation_plan.md`) | Identification statistics + resolution study; still the gate for all science layers. Step 0.4 (tracking-graph split/merge probe) still applies to the retained tracking graph |
| **`clustering_investigation_plan.md`** | Entirely identification-side (OPTICS parameters) |
| **ERA5 re-download, `example.png` / notebook disposition** | Credential/housekeeping items from the needs-Joy register |

### Affected — keep, with amendments once this ships

| Item | Amendment |
|---|---|
| **`validation_strategy_plan.md`** | Layer 2 duration/propagation statistics and the Layer 1 temporal invariants should be computed from **envelope tracks** (they are Souders-comparable by construction — same field, same smoothing, same thresholds). The `c_g > c_p` invariant becomes directly computable. Identification-side layers unchanged |
| **`western_disturbance_validation_plan.md`** | §5.2 upstream provenance and §5.4 precursor lead time should trace **envelope-track lineage** instead of walking the raw tracking graph through merges — strictly more robust. Parts A/B and the §4 subtropical tuning are unaffected (tuning gains one knob: subtropical `envelope_threshold`) |
| **`regime_rwp_structure_plan.md`** | Same substitution wherever it reuses WD §5.1–5.2 embeddedness/provenance; the per-timestep phase analysis (its core) is untouched |
| **`architecture_and_algorithm.md`** | Documentation resync after implementation (add the envelope tier to §2.2) |

### Retire / supersede

| Item | Disposition |
|---|---|
| **The "group-velocity question"** (needs-Joy register; assessment Next #3) | **Resolved by this design** — option (a), envelope primitive. Strike from the register on approval |
| **`handoff_feature_tracks.md`** "open goal / what to try next" sections | Already marked obsolete; this design formally supersedes the goal. The algorithm/parameter documentation stays accurate (the layer is reused verbatim) and the file remains as component documentation |
| **`envelope_segmentation_proposal.md`** | Approaches 1 & 2 already in the do-not-retry register. The remaining simplified variant is conditionally retired: drop it if the `lat_gate` sweep shows over-merging controlled (§5); otherwise fold it into this design's envelope machinery as an identification refinement. Either way it stops being an independent plan |
| **Spec tasks 7.3 (Hovmöller) / 9.3 (`to_dataset`)** | Unchanged from the engineering-backlog disposition (deferred / superseded); listed only for completeness |

## 8. Implementation order (sketch — full plan via writing-plans)

1. Promote envelope helpers to `waper/identification/envelope.py`; re-point `masks.py`.
2. `extract_envelope_features` + tests.
3. Envelope tracks via existing tracker + `group_velocity` + synthetic acceptance test 1.
4. Attachment + `waper/io/` column + tests.
5. Forecast-bust integration run → acceptance test 2; record experiment + assessment
   entries.
6. Amendments of §7 (validation-plan tweaks, register strikes, handoff supersession
   note).
