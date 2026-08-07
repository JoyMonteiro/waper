# Housekeeping & engineering backlog

> Opened 2026-08-07. Items that need **no scientific input** — housekeeping and software
> engineering only. Each carries the evidence needed to act on it cold, so nothing has to
> be re-derived.
>
> **Verify status against the tree, not these checkboxes** — plans in this repo have a
> track record of shipping unticked. See `superpowers/plans/archive/README.md`.

## Done 2026-08-07

- [x] **Ignore large data artifacts.** `datasets/` held 4.6 GB of untracked NetCDF with no
      `*.nc` rule. Fixed in `92b4fa2`.
- [x] **Track the logs.** `assessments/`, `datasets/experiments/`, `results/` were untracked.
      Fixed in `0de90db`.
- [x] **Commit and push the backlog.** 26 unpushed commits + the uncommitted feature-tracks
      rework. Pushed to `origin/energy-weighted-tracking` (`13e3a5f..0de90db`).
- [x] **Fix the failing test.** numpy capped at `< 2.5`; datashader `importorskip` in the one
      test needing it. Fixed in `b9f4d0b`. Suite: 129 passed, 1 skipped.

---

## Tier 1 — small and safe (~1 hour)

- [ ] **Three deferred branch-resolution review findings.** Adjudicated non-correctness-
      affecting during the SDD review; see `.superpowers/sdd/progress.md`.
  - `_lat_ranges_within` (`waper/identification/rwp_graph.py:403-409`) — `lo`/`hi` are
    semantically inverted (they are overlap bounds; `lo > hi` means disjoint). Readability
    only, zero behaviour change.
  - Missing test: "no longitude overlap but latitude within gate → False". Obvious by
    early-return, still untested.
  - Mid-file `import pytest` in `tests/test_rwp_branch_resolution.py` — style.

- [ ] **`_arc_bins` off-by-one** (`waper/identification/rwp_graph.py:389-390`).
      `n = int(length // step) + 1` followed by `range(n + 1)` yields two bins past the arc
      end; should be `range(n)`. **Unlike the items above this perturbs identification
      output** (slightly wider overlap detection, conservative direction), so re-run the
      branch-resolution acceptance test alongside. The intended semantics are unambiguous,
      so no judgement call is involved.

- [ ] **Doc hygiene.**
  - `datasets/experiments/2026-03-21-initial-audit.md` — "Next experiments" has two items
    numbered 4, and items 4 and 5 are verbatim duplicates of each other.
  - `datasets/experiments/README.md` parameter list predates `lat_gate` (default 15.0),
    `hull_method`, and `hemisphere`. `lat_gate` in particular has never been swept.

## Tier 2 — half a day

- [ ] **Resurrect CI.** `.github/workflows/test.yaml` has never run on any of this work.
      Three independent breakages:
  - Triggers on `master` / `dev` / `ci`; this repo's default branch is `main` and work
    happens on feature branches.
  - Matrix requests Python 3.12, but `tox.ini`'s envlist stops at py311 (and still lists
    EOL py36/37/38).
  - `upload-artifact@v2`, `setup-python@v2`, `checkout@v3` are deprecated; the v2 artifact
    actions are retired and hard-fail.
  - Do this *after* the numpy pin (already done) or CI just goes red on the numba conflict.

- [ ] **Strip notebook outputs.** `scripts/method_comparison/method_comparison.ipynb`
      already has **five blobs of 1.5–6.2 MB** in history. Its current working-tree diff is
      5,057 lines of pure output — code cells are byte-identical to HEAD. Add `nbstripout`
      (or an equivalent filter) before committing notebooks again.

- [ ] **Consolidate changelogs.** `CHANGELOG.rst` still carries cookiecutter boilerplate —
      it references "my_new_project" and "john-doe-gh-account-name" — last real entry 2022.
      `HISTORY.md` is the live one but stops at 2026-03-12. Pick one, backfill from git log.

- [ ] **Reclaim ~4 GB of local git bloat.** `.git` is 4.2 GB; reachable objects are under
      100 MB. A single unreachable 4,015 MB blob, hash-verified as `datasets/validation.nc`,
      is dangling from an old commit (a 242 MB pack is probably `forecast_bust_hourly.nc`).
      It is unreachable from any branch, so it never reached the remote. `git gc --prune=now`
      with reflog expiry reclaims it. The data is safe in the working tree — but expiring
      the reflog permanently discards recovery of those old commits, **so confirm first**.

## Tier 3 — large, still no scientific input

- [ ] **Refactoring spec Phase 5: VTK → PyVista/SciPy.** The only live remainder of
      `waper_refactoring_spec.md`. `waper/identification/{utils,max_min,topology}.py` still
      import VTK and there is no `scipy.sparse` shortest-path. Multi-day; not a wrap-up item.

---

## Explicitly NOT here — these need Joy's input

Do not action these as housekeeping; each changes scientific behaviour or needs external
credentials.

- **Promoting the operating point.** `datasets/visualize.py:46-50` hard-codes GT=0.02 /
  penalty=4000 / ST=20 while `WaperConfig` defaults stay GT=3e-5 / penalty=2000
  (`waper/interface/api.py:305-308`). Flagged 2026-03-21 as "implicit, not yet codified";
  unchanged since. Looks like housekeeping, isn't — two configurations are live at once.
- **`track_pruning_threshold=0.3`**, flagged in memory as a likely bug (prunes by distance
  in km, so 0.3 empties the graph; `visualize.py` bypasses it with 8000). Verifying the bug
  needs no input; changing the default does.
- **The group-velocity question** — envelope primitive vs trough-to-trough handoff vs
  dropping the expectation. Blocks the feature-track line of work. See
  `assessments/2026-08-07.md`.
- **Envelope-weighted edge pruning** (`envelope_segmentation_proposal.md`) — and whether it
  still adds anything on top of the shipped `lat_gate` branch resolution.
- **Re-downloading `datasets/v_winds_300mb_nh_2022_2023.nc`** (still a 239 B stub) — needs
  ERA5 credentials.
- **Disposition of two uncommitted files**: `example.png` (unexplained 211 KB binary at repo
  root) and the `method_comparison.ipynb` output-only diff.
