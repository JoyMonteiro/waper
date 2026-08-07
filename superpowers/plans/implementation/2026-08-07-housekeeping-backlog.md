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
  - *(since gated)* A `lint` job now runs `ruff check .` and `mypy` alongside the suite. See
    the Tier 2 entry below for what had to happen first.
  - **Verified on the real thing.** First-ever run: `31136489671`. The ubuntu dependency
    solve for the geo stack (vtk, geovista, cartopy, rasterio, datashader) succeeded, and
    the OpenGL/off-screen setup was enough — no collection errors. 126 passed on both 3.11
    and 3.12, matching local.
  - One genuine gap it surfaced: `spharm` (pyspharm) is imported lazily by
    `scripts/method_comparison/masks.py::t21_truncate` but was **never a declared
    dependency** — it only ever worked because it is present in the local conda env. It has
    no wheels and needs a Fortran toolchain, so it is now an `importorskip` rather than a
    CI dependency, matching the datashader precedent in `tests/interface/test_explorer.py`.
  - Cosmetic: GitHub warns that `checkout@v4` / `setup-python@v5` / `upload-artifact@v4`
    target Node 20 and are being forced onto Node 24. Not a failure; no action needed yet.

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

- [x] **Ruff/mypy: configure, then gate.** Both are configured in `pyproject.toml`, the tree
      is clean, and the CI `lint` job enforces them.
  - Ruff selects `E, F, I, UP, B, C4, SIM, PIE, RUF` explicitly. Relying on the default
    selection was the trap: it is not `E4,E7,E9,F` in current releases and it moves between
    them, so "319 errors" was never a stable number to work against.
  - `misc/` and `*.ipynb` are excluded (dead pre-package research code; scratch notebooks).
    `E501` is off — no autoformatter runs here, and 34 lines were over 100. The side-effect
    import ordering in `scripts/` (`matplotlib.use("Agg")` before `pyplot`) and the compact
    `a = f(); a.x = 1` idiom in `tests/` are per-file exemptions rather than rewrites; the
    same idiom *was* cleaned up inside `waper/`, which is held to a higher bar.
  - One genuine bug among the 319: `scripts/diag_purple.py::dist` closed over the loop
    variable `head` (B023). The rest were style, dead code, or mid-file imports that had
    accumulated from appending to files.
  - Mypy needed two workarounds before it would even start: 2.3 rejects
    `python_version = "3.9"`, and numpy's bundled stubs use `type` statements that only
    parse on 3.12+. No version pin, and the CI lint job runs on 3.12.
  - It found **8 real errors**, all fixed — three implicit-`Optional` defaults,
    `raster_features` declared `list` but always assigned a `set`, `Feature.footprint`
    typed `object`, an unannotated `_time_step_data`, two colormap dicts whose keys needed
    to narrow to the channel literals. `check_untyped_defs` stays off: the package is
    largely unannotated and turning it on buries these.
  - The lint job installs the **full** dependency set, not just the linters — numpy, pandas
    and matplotlib ship `py.typed`, and mypy silently degrades to `Any` without them, which
    is exactly how these 8 would have been missed. Ruff and mypy are version-pinned there so
    a new release cannot fail an unrelated push; bump deliberately.

- [x] **Retire the cookiecutter scaffolding — done 2026-08-07.** Deleted the four dead config
      files (`tox.ini`, `.prospector.yml`, `.pylintrc`, `.bettercodehub.yml`) and rewrote
      `CONTRIBUTING.md` around the workflow that actually exists here — pytest directly, `ruff`
      and `mypy`, the per-clone `nbstripout` filter — with no "My New Project" left in it.
      `README.rst` followed: the deletions had left dangling `tox` references and a Better Code
      Hub badge with no config file, and its cookiecutter "TODO Document a **Great Feature**"
      placeholders are now real Features/Usage prose. Its Quickstart installs from a source
      checkout: the only PyPI release is `0.0.1`, which predates everything here.
  - `requires-python` narrowed to `>= 3.11`, matching what CI verifies rather than testing
    3.9/3.10; the 3.9/3.10 classifiers went with it, and ruff's inferred `target-version`
    followed to py311, so the tree was re-cleaned under it in the same task. The
    `pypi/pyversions` badge was deleted too — it renders the interpreters of release `0.0.1`
    and so contradicted the new floor.
  - A tree-wide re-grep afterwards found `docs/` was *also* untouched cookiecutter: a Sphinx
    skeleton with `my_new_project` autodoc, a `.readthedocs.yml` building a nonexistent
    `.[docs]` extra, and a `.coveragerc` sourcing `my_new_project`. Joy's ruling: drop
    Sphinx/RTD entirely. `docs/` is now a Quarto site (landing page, the pre-existing
    `docs/algorithm.md`, and a `quartodoc`-generated API reference) deployed to GitHub Pages
    by `.github/workflows/docs.yml`. Manual step outstanding for Joy: enable Pages on the repo.
  - The licence discrepancy surfaced on the way past and was settled: `pyproject.toml`
    classified the project AGPLv3 while `LICENSE` is a genuine BSD 3-Clause. Joy ruled BSD;
    the classifier is now `License :: OSI Approved :: BSD License`, PyPI having no
    3-Clause-specific trove string.
  - Deferred, not done here: `docs/algorithm.md` §1 still describes plain footprint overlap,
    which the energy-weighted tracking on this branch supersedes; and the generated API pages
    are signature-only because no public class carries a docstring yet.

- [x] **Reclaim ~4 GB of local git bloat — done 2026-08-07.** Re-verified before acting:
      `git hash-object datasets/validation.nc` returned `4a5d3e41…`, byte-identical to the
      unreachable 4,015 MB blob, so pruning discarded no data that is not still on disk.
      Joy authorised the reflog expiry. Ran `git reflog expire --expire=now
      --expire-unreachable=now --all` then `git gc --prune=now --aggressive`.
      **`.git`: 4.2 GB → 65 MB** (26 packs → 1, 62 MiB). `git fsck` clean, history intact,
      suite still 126 passed. Commits were pushed to `origin` first.

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
- ~~**`track_pruning_threshold=0.3`**, flagged in memory as a likely bug (prunes by distance
  in km, so 0.3 empties the graph; `visualize.py` bypasses it with 8000).~~ **Closed
  2026-08-07** — Joy authorised the change; the default is now 8000 km everywhere and the
  unit is documented. See `af4874b`.
- **The group-velocity question** — envelope primitive vs trough-to-trough handoff vs
  dropping the expectation. Blocks the feature-track line of work. See
  `assessments/2026-08-07.md`.
- **Envelope-weighted edge pruning** (`envelope_segmentation_proposal.md`) — and whether it
  still adds anything on top of the shipped `lat_gate` branch resolution.
- **Re-downloading `datasets/v_winds_300mb_nh_2022_2023.nc`** (still a 239 B stub) — needs
  ERA5 credentials.
- **Disposition of two uncommitted files**: `example.png` (unexplained 211 KB binary at repo
  root) and the `method_comparison.ipynb` output-only diff.
