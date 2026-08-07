# Version History

This is the project's single changelog. `CHANGELOG.rst` was retired on 2026-08-07 — it
was cookiecutter scaffolding (it referred to "my_new_project" and a placeholder GitHub
account) whose only two real entries have been folded in at the bottom of this file.

## [Unreleased]

### 2026-08-07 — housekeeping

#### Added
- `.github/workflows/test.yaml` rewritten so CI can actually run: triggers on every push
  and on PRs into `main` (it previously watched `master`/`dev`, branches that do not
  exist here, and had never run once). Goes through pytest directly rather than `tox.ini`,
  which was untouched cookiecutter scaffolding targeting a `src/my_new_project` layout and
  has since been deleted (see Removed, below).
- `.gitattributes` with an `nbstripout` clean filter, so notebook outputs stop entering
  git history. `scripts/method_comparison/method_comparison.ipynb` had already deposited
  five output blobs of 1.5–6.2 MB.
- `.gitignore` rules for the large data, paper, and figure artifacts that had been sitting
  untracked in `datasets/` (4.6 GB of NetCDF, with the small test fixtures un-ignored).
- `assessments/`, `datasets/experiments/`, and `results/` are now tracked.
- Regression test `test_arc_bins_do_not_run_past_the_arc_end`.
- Documentation site: `docs/` is now a [Quarto](https://quarto.org) site — a landing page, the
  pre-existing `docs/algorithm.md`, and a `quartodoc`-generated API reference — built and
  deployed to GitHub Pages by the new `.github/workflows/docs.yml`. A `docs` extra
  (`quartodoc`, `griffe`, `jupyter`) installs the Python side; Quarto itself is not a Python
  package.

#### Changed
- `requires-python` narrowed from `>= 3.9` to `>= 3.11`, matching the interpreters CI actually
  verifies. The 3.9/3.10 classifiers went with it, and ruff's inferred `target-version`
  followed to py311, so the tree was re-cleaned under it (chiefly `zip(..., strict=)` on every
  call site, each audited individually, and `itertools.pairwise` where it subsumed one).
- `CONTRIBUTING.md` rewritten around the workflow that exists here — pytest directly, `ruff`
  and `mypy`, the per-clone `nbstripout` filter, and how to build the docs. No "My New Project"
  remains in it.
- `README.rst`: the cookiecutter "TODO Document a **Great Feature**" placeholders are replaced
  with real Features and Usage prose, and Quickstart now installs from a source checkout. The
  sole PyPI release is `0.0.1`, which predates the `>= 3.11` floor, so `pip install waper`
  would have handed users a release incompatible with the stated requirements.
- The landing page's description of tracking now matches the code on this branch: an
  energy-weighted overlap (`waper/tracking/energy_overlap.py`), not plain footprint overlap.

#### Fixed
- `track_pruning_threshold` is a haversine distance **in km**, not an overlap weight. The
  old default of `0.3` pruned every tracking edge and left an empty graph; it is now
  `8000` km — the value `datasets/visualize.py` had been passing directly to work around
  the bug — in `WaperConfig`, `Waper.__init__`, and every script and test. `plot_tracks()`
  falls back to the configured threshold rather than comparing against `None`.
- `_arc_bins` off-by-one in `waper/identification/rwp_graph.py`: `range(n + 1)` → `range(n)`.
  Adjacent-but-disjoint arcs (`[10,30]` and `[31,41]`) no longer read as overlapping.
- `numpy` capped below 2.5 (numba, pulled in via datashader, requires `numpy <= 2.4`), and
  the one explorer test needing datashader now uses `importorskip`.
- Real-data tests that read gitignored NetCDF (`test_acceptance_t95`, and the slow
  `test_method_comparison` cases) are guarded with `skipif` so a fresh clone no longer
  hard-fails on a missing file.
- Licence classifier in `pyproject.toml` said AGPLv3 while `LICENSE` is a genuine BSD
  3-Clause and the README said BSD. Now `License :: OSI Approved :: BSD License` (PyPI's
  trove list has no 3-Clause-specific BSD entry).

#### Removed
- Cookiecutter scaffolding that nothing depended on: `tox.ini` (400 lines targeting a
  `src/my_new_project` layout), `.prospector.yml`, `.pylintrc`, `.bettercodehub.yml`, the
  Sphinx skeleton under `docs/` with its `my_new_project` autodoc, `.readthedocs.yml` (it
  built a `.[docs]` extra that did not exist, and RTD has no finished builds), and an inert
  `[tool.setuptools_scm]` table (the version is static, so it never applied).
- README badges with no service behind them: Read the Docs, Codecov, Code Climate,
  `commits-since`, and `pypi/pyversions` — the last reporting the interpreters of release
  `0.0.1`, which misstates the supported range.

### 2026-06-15 – 2026-06-22 — RWP branch resolution, method comparison, feature tracks

#### Added
- **Latitude-gated branch resolution** (`lat_gate`): longitude/latitude span helpers and an
  in-band interleave test, in-band zonal exclusivity in `get_ranked_paths` pass 1, and
  orphan reassignment wired through config.
- **Three-method agreement study** under `scripts/method_comparison/`: agreement metrics,
  grid helpers and the Zimin Hilbert envelope, Zimin / edge-pruning / node-amplitude mask
  builders, sweep orchestration with CSV output, and a plotting notebook for IoU curves,
  disagreement maps, and case studies. Later extended with Souders T21 spatial and 24-hour
  temporal envelope filtering, an anomaly scan with latitude-stratified disagreement cells,
  an interactive GeoViews time slider with per-RWP colouring, and a poorest-overlap view.
- **Feature-track layer (SP1)**: a `Feature` model with footprint-overlap matching,
  `track_features` handling births, extension, and deaths, weak-pool recovery, latitude-band
  termination, phase velocity and flat-table export, `extract_features` with region-hull
  footprints, and a hemisphere GIF plus an empirical continuity test. Reworked to IoU
  matching, split tracks, and BFS-partitioned footprints.
- **Energy-weighted tracking**: amplitude²-weighted centroids, energy disks at extrema,
  energy-field rasterisation on the stereographic grid, per-timestep energy rasters built
  during identify, and timestep association by energy overlap of cores.

#### Fixed
- `reassign_orphans` no longer re-introduces in-band overlap.
- `get_ranked_paths` skips `source == sink`, so no length-1 RWPs are emitted.
- Paths split only on a true globe wrap.
- Spherical amplitude-weighted RWP centroid.
- Explorer renders in polar-stereographic while polygons stay in their native CRS.

### 2026-06-06 — catalogue I/O and the interactive explorer

#### Added
- `waper.io` catalogue: serialization, extraction, query filtering, and science metrics.
- HoloViews layer builders for the explorer, vendored NL diverging colormaps with
  `bokeh_palette`, and a `run_explorer.py` script for interactive RWP analysis.

#### Fixed
- Coordinates aligned in degrees for the map overlay and Hovmöller; polygons layer uses
  `fill_color` / `line_color`.

### 2026-03-13 – 2026-03-21 — clustering, tracking, and polygon overhaul

#### Added
- Southern Hemisphere support via a `hemisphere` config option.
- `penalty_length_scale_km` parameter on `cluster_extrema`.
- Per-node polygon union in place of a single convex hull.

#### Changed
- Extremum clustering: affinity propagation → DBSCAN → OPTICS (multi-scale).
- Track extraction: `all_simple_paths` → linear-time DAG dynamic programming.
- Path filtering: pairwise filter → greedy maximum-weight independent set.
- Cluster representatives use weighted centroids.

#### Fixed
- Dateline wraparound in longitude delta pruning.
- Distance scaling in cluster representatives.
- Dead zero-crossing penalty replaced with fractional-descent hill-climbing, and
  `penalty_length_scale_km` tuned against dataset validation.
- Quadtree merge computed once per timestep pair, excluding background feature 0.
- Southern-Hemisphere raster orientation normalised.

## [Unreleased] - 2026-03-12

### Added
- Test infrastructure with pytest, including synthetic fixtures for wave fields.
- Comprehensive test suites for:
  - Extrema detection (`test_max_min.py`).
  - Extrema clustering (`test_clustering.py`).
  - Association graph building and pruning (`test_association_graph.py`).
  - Temporal tracking (`test_tracking.py`).
  - Integration pipelines (`test_integration.py`).
- Added robust logging to replace scattered print statements throughout the codebase.
- Explicit dependencies defined in `pyproject.toml` instead of implicit conda `environment.yml`.

### Changed
- Refactored `waper/__init__.py` and `waper/interface/__init__.py` to correctly expose the public API without leaking internal submodules.
- Python version bounds updated in CI to correctly target `3.9` through `3.12`.
- Removed `environment.yml` and transitioned entirely to `pyproject.toml` dependencies.
- Empty datasets edge cases handled appropriately to gracefully bypass pipeline execution instead of throwing `KeyError`s during clustering, graph building, and plotting phases.
  - `max_min.py`: Explicitly check dataset `n_points` instead of `'is max'` existence.
  - `topology.py`: Fast return on empty inputs for `cluster_max`, `cluster_min`, `max_cluster_assign`, `min_cluster_assign`.
  - `rwp_graph.py`: Return empty association graph if extrema inputs are empty.
  - `api.py`: Bypass tracking quadtree instantiation if the raster data lacks tracked fields.

### Removed
- Removed large blocks of dead and commented-out code in `max_min.py`, `rwp_graph.py`, `topology.py`, and `tracking_graph.py` to improve readability.
- Deleted absolute imports inside `api.py`.

## 2022

Carried over from the retired `CHANGELOG.rst`.

- **2022-09-21** — Moved most identification code to appropriate files.
- **2022-06-24** — Some superficial changes.
- **2022-06-04** — First release (`0.0.1`), scaffolded from the
  [Cookiecutter Python Package](https://python-package-generator.readthedocs.io/en/master/)
  template.
