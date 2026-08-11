# Version History

This is the project's single changelog. `CHANGELOG.rst` was retired on 2026-08-07 — it
was cookiecutter scaffolding (it referred to "my_new_project" and a placeholder GitHub
account) whose only two real entries have been folded in at the bottom of this file.

## [Unreleased]

### 2026-08-07 — VTK removal completed (refactoring spec Phase 5, tasks 5.2 and 5.4)

**No module under `waper/` imports VTK any more.** Phase 5 is closed.

#### Changed
- `cluster_extrema` (`waper/identification/topology.py`) computes geodesic distances with
  `scipy.sparse.csgraph.dijkstra` instead of `vtkDijkstraGraphGeodesicPath`. Two new
  helpers do the work: `_surface_graph()` builds the triangle-edge adjacency matrix of the
  clipped surface, and `_path_extremes()` reads the hill-climbing path extreme off the
  shortest-path tree. The old code ran one VTK Dijkstra per pair of extrema; the new code
  runs one scipy pass sourced at every extremum at once, which made identification roughly
  1.5–2× faster per time step.

  **Output is unchanged, and this was checked rather than assumed** — the task alters
  clustering numerics, so a green suite would not have settled it. The pre-refactor VTK
  loop was transcribed verbatim and run side by side with the new code on
  `forecast_bust_hourly.nc`: 6 fields, both signs, two mesh resolutions, 13,551 extrema
  pairs. Geodesic distances agree to ~2 m (VTK accumulates float32 segment lengths, scipy
  sums in float64) and the path extreme feeding the penalty was identical on every pair, so
  Dijkstra tie-breaking does not diverge on these meshes. End-to-end, cluster membership
  and `identified_rwp_paths` are unchanged across 4 time steps.
- The `vtkGeometryFilter` + `vtkTriangleFilter` chain became
  `extract_surface().triangulate()`, verified byte-identical on real data. It is guarded by
  an `isinstance` check: both callers already pass `PolyData`, for which the geometry filter
  was a pass-through, and calling `extract_surface()` on one emits a `PyVistaFutureWarning`
  per time step that the `algorithm=` keyword cannot silence under the declared
  `pyvista >= 0.36` floor.

#### Removed
- `cluster_extrema`'s `base_field` first argument. It fed only the `vtkCellLocator` and a
  `"<name> Cell Value"` fallback that was already unreachable — the point-data lookup
  guarding it never returns `None` on these meshes. Call sites in `api.py` and
  `tests/test_clustering.py` updated.
- `add_connectivity_data_min` (task 5.4), the last raw VTK call left in `topology.py`. It
  had no callers anywhere in the tree, so it was deleted rather than ported to PyVista,
  matching the precedent from 5.1 and 5.3. `identify_connected_regions` was already PyVista
  and is unchanged.

#### Added
- Four tests in `tests/test_clustering.py` pinning the two helpers, including the two traps
  found while writing them: `coo_matrix` **sums** duplicate entries, so the triangle edges
  shared by two faces must be de-duplicated or every interior edge comes out at twice its
  true length; and scipy's "no predecessor" sentinel is `-9999`, not `-1`.

### 2026-08-07 — tracking

#### Added
- `track_weight_threshold`: an optional gate on the envelope-level overlap `weight` of a
  tracking edge, alongside the existing `track_pruning_threshold` distance gate. The two
  are independent and prune in opposite directions — distance from above, weight from
  below — so an edge must clear both. Weight is the energy overlap of two whole-RWP
  footprints normalised to (0, 1] by the larger feature energy, so the gate also
  suppresses merges and splits where a small packet is absorbed into a much larger one.

  **Disabled (`None`) by default**, leaving existing behaviour unchanged: it has not been
  calibrated against a reference dataset. Until now `weight` only influenced track
  *selection* (the heaviest-path DP of §10.3 deprioritises a weak edge but still keeps it
  when it is the only one available); this is the first way to remove such an edge
  outright. Exposed on `WaperConfig`, `Waper.__init__`, `Waper.plot_tracks`, and
  `prune_tracking_graph`.

### 2026-08-07 — quadtree removal

#### Removed
- `waper/tracking/quadtree.py` and the `WaperSingleTimestepData.quadtree` field. The
  quadtree had been orphaned since `e349b42` (2026-06-17) replaced quadtree merge with
  masked NumPy over the raster pair, but it was still built for every time step and read by
  nothing except its own test. This closes refactoring-spec tasks 5.5 / 9.1, though not by
  the route those tasks proposed — tracking did not move to Shapely polygon intersection,
  and the rasters it wanted deleted are now load-bearing for the energy overlap.
- `test_quadtree_pixel_counts`, replaced by two direct tests of the energy primitives that
  superseded it: `test_feature_energies_are_summed_per_feature` (per-feature totals stay
  separate, background 0 excluded — the property the old pixel-count assertion pinned) and
  `test_overlap_energies_keeps_features_separate`.

### 2026-08-07 — VTK removal (refactoring spec Phase 5, tasks 5.1 and 5.3)

#### Removed
- `get_iso_contour` and `compute_gradients` (`waper/identification/utils.py`) and
  `interpolate_cell_values` (`waper/identification/max_min.py`), with the `import vtk` each
  file carried for them. All three were dead — the only references in the tree were their
  own definitions and the refactoring spec. `interpolate_cell_values` in particular was
  redundant rather than merely unused: `get_vtk_object_from_data_array` already fills
  `"<name> Cell Value"` via PyVista's `point_data_to_cell_data()` on every grid it builds.

#### Changed
- `add_maxima_data`/`add_minima_data` use PyVista's `.n_cells` instead of raw
  `GetNumberOfCells()`. The four docstrings in `max_min.py` that declared
  `vtk.vtkUnstructuredGrid` parameters now say `pv.UnstructuredGrid`; those functions have
  taken PyVista objects for as long as they have called `.n_points` and `.extract_points`.

`waper/identification/topology.py` is now the only module importing vtk — task 5.2, the
`vtkDijkstraGraphGeodesicPath` → `scipy.sparse.csgraph` rewrite, is untouched. *(Closed
later the same day; see the Phase 5 completion entry above.)*

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
- `docs/algorithm.md` resynced with the shipped algorithm. Tracking is described as the
  energy-weighted overlap it is throughout — §1, the pipeline diagram, and §10.1's weight
  formula, which still gave `overlap_pixels / max(size_prev, size_curr)` and so contradicted
  §10.2 one screen below it. Two shipped features are documented for the first time: §7.4
  branch resolution and §7.5 orphan reassignment (the `lat_gate` machinery in
  `get_ranked_paths`/`reassign_orphans`), and §9.2 the energy raster (`energy_disks` +
  `rasterize_energy`). §9.3 now records that the quadtree, though still built per time step,
  no longer feeds tracking. Appendix A gains `lat_gate` and `energy_radius_km`.

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
