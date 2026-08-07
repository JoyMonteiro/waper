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

## Tier 1 — small and safe (~1 hour) — **done 2026-08-07**

- [x] **Three deferred branch-resolution review findings.** Adjudicated non-correctness-
      affecting during the SDD review; see `.superpowers/sdd/progress.md`.
  - `_lat_ranges_within` — `lo`/`hi` renamed to `overlap_lo`/`overlap_hi` with a comment;
    zero behaviour change.
  - Missing test added to `test_paths_interleave_in_band`: same band, disjoint longitude
    → False.
  - `import pytest` hoisted to the top of `tests/test_rwp_branch_resolution.py`.

- [x] **`_arc_bins` off-by-one** (`waper/identification/rwp_graph.py:389-390`).
      `range(n + 1)` → `range(n)`. New regression test
      `test_arc_bins_do_not_run_past_the_arc_end` pins the boundary (adjacent-but-disjoint
      arcs `[10,30]` / `[31,41]` no longer read as overlapping). Full suite green,
      including the real-data `test_acceptance_t95` — 130 passed, 1 skipped.

- [x] **Doc hygiene.**
  - `datasets/experiments/2026-03-21-initial-audit.md` — dropped the duplicate item, so
    "Next experiments" now numbers 1–7 cleanly.
  - `datasets/experiments/README.md` — added `lat_gate` (15.0, flagged never-swept),
    `hull_method`, `hemisphere` to the parameter list.

## Tier 2 — half a day — **CI / notebooks / changelogs done 2026-08-07**

- [x] **Resurrect CI.** Rewritten in `.github/workflows/test.yaml`. The backlog listed three
      breakages; there was a fourth, fatal one:
  - *(fixed)* Triggered on `master` / `dev` / `ci`. Now runs on **every push** plus PRs into
    `main`, with a `concurrency` group so a newer push cancels the in-flight run.
  - *(fixed)* `checkout@v3` → `v4`, `setup-python@v2` → `v5` (with pip caching),
    `upload-artifact@v2` → `v4`. The retired v2 artifact actions would have hard-failed.
  - *(fixed)* Matrix is now `["3.11", "3.12"]` on `ubuntu-latest`.
  - **Fourth breakage, not previously recorded:** the workflow drove everything through
    `tox`, but `tox.ini` is unmodified cookiecutter scaffolding — `PY_PACKAGE = my_new_project`
    and a `src/` layout that has never existed here (the package is a flat `waper/`). Every
    tox env — `lint`, `type`, `check`, the test envs — points at `src/my_new_project`. CI
    could never have gone green by fixing triggers and action versions alone. The workflow
    now calls `pytest` directly and skips tox entirely.
  - Runs `pytest -m "not slow"`. Installs `libgl1`/`libglx-mesa0` and sets
    `PYVISTA_OFF_SCREEN`, because VTK and PyVista are imported at collection time and need
    an OpenGL runtime even headless.
  - **Not gated on lint.** `ruff check .` reports **319 errors** (193 auto-fixable) against a
    repo with no `[tool.ruff]` config. Gating now would pin CI red forever. See Tier 2 below.
  - **Caveat:** verified only as far as local green (126 passed, 1 skipped, 4 deselected in
    71 s) plus YAML parse. The ubuntu dependency solve for the geo stack (vtk, geovista,
    cartopy, rasterio, datashader) is unverified until the first real run.

- [x] **Guard tests against absent data.** Found while wiring CI, not in the original list:
      `test_acceptance_t95` opened `datasets/forecast_bust_hourly.nc` (652 MB, gitignored)
      with **no `skipif`** — it would hard-fail on CI and on any fresh clone. Same for the
      three slow `test_method_comparison` cases via `run_sweep.DATA_PATH`. All four now skip
      when the file is absent, matching the existing idiom in `test_feature_tracks.py`.

- [x] **Strip notebook outputs.** `.gitattributes` now applies an `nbstripout` clean filter
      to `*.ipynb`; `nbstripout` added to the `dev` extra and the setup step documented in
      `CONTRIBUTING.md`. The filter is **per-clone** — a fresh clone must run
      `nbstripout --install --attributes .gitattributes` or checkouts fail.
      Outputs stay in the working copy and simply stop reaching the index: the notebook's
      diff went from `+3941/-1116` to `+109/-12116` while all 6 output cells remain on disk.
      The five 1.5–6.2 MB blobs already in history are untouched — removing those needs a
      history rewrite, which is **not** housekeeping.

- [x] **Consolidate changelogs.** `HISTORY.md` is now the single changelog, backfilled from
      the 88 commits since 2026-03-12 and grouped by theme. `CHANGELOG.rst` deleted; its two
      real 2022 entries folded in at the bottom. Also fixed three dead `[project.urls]`
      entries in `pyproject.toml` — a malformed bug tracker (`github.com/waper/issues`), and
      two links to `.rst` files on a `master` branch that does not exist.

- [ ] **Ruff/mypy: configure, then gate.** `ruff check .` → 319 errors, 193 auto-fixable, no
      `[tool.ruff]` section in `pyproject.toml`. Wants a deliberate rule selection and a
      formatting pass before CI can enforce it. `mypy` is in the `dev` extra and equally
      unconfigured.

- [ ] **Retire the cookiecutter scaffolding.** Now that nothing depends on it: `tox.ini`
      (400 lines targeting `src/my_new_project`, poetry, prospector, sphinx), `.prospector.yml`,
      `.pylintrc`, `.bettercodehub.yml`. `CONTRIBUTING.md` still says "My New Project" and
      documents the broken tox workflow in places. `requires-python = ">= 3.9"` and the 3.9/3.10
      classifiers claim support that CI does not verify — either test those or narrow the claim.

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
