# WAPER Refactoring Specification

## Document Purpose

This specification captures **every** known issue, improvement, and enhancement for the WAPER (Rossby Wave Packet Extraction and Representation) package. Each item is written as a self-contained task with explicit step-by-step instructions so that any developer — or a code-generation LLM — can pick up a single task and execute it without needing to understand the full history of the project.

**How to use this document:** Work through the phases in order. Within each phase, tasks can often be parallelised (noted where possible). Every task ends with a "Definition of Done" checklist. Do not skip the testing tasks — they are load-bearing.

---

## Table of Contents

- [Phase 0: Scaffolding, CI, and Project Hygiene](#phase-0-scaffolding-ci-and-project-hygiene)
- [Phase 1: Testing Infrastructure](#phase-1-testing-infrastructure)
- [Phase 2: Critical Bug Fixes](#phase-2-critical-bug-fixes)
- [Phase 3: Algorithmic Improvements — Identification](#phase-3-algorithmic-improvements--identification)
- [Phase 4: Algorithmic Improvements — Tracking](#phase-4-algorithmic-improvements--tracking)
- [Phase 5: VTK-to-PyVista / SciPy Refactor](#phase-5-vtk-to-pyvista--scipy-refactor)
- [Phase 6: Performance Optimisation](#phase-6-performance-optimisation)
- [Phase 7: Visualisation Overhaul](#phase-7-visualisation-overhaul)
- [Phase 8: Documentation](#phase-8-documentation)
- [Phase 9: Stretch Goals](#phase-9-stretch-goals)
- [Appendix A: File Inventory](#appendix-a-file-inventory)
- [Appendix B: Magic Numbers Registry](#appendix-b-magic-numbers-registry)
- [Appendix C: Dependency Map](#appendix-c-dependency-map)

---

## Phase 0: Scaffolding, CI, and Project Hygiene

These tasks have **zero** algorithmic risk and can be done first to establish a clean working environment.

### Task 0.1 — Fix `pyproject.toml` Metadata and Dependencies

**Problem:** `pyproject.toml` has no `[project.dependencies]` section. All dependencies are implicit via the conda `environment.yml`. The CI workflow references a package called `cookiecutter_python`, not `waper`. Python 3.6/3.7 are listed in the CI matrix but the project requires `>= 3.9`.

**Files to edit:**
- `pyproject.toml`
- `.github/workflows/test.yaml`

**Steps:**

1. Open `pyproject.toml`.
2. Add a `[project.dependencies]` section listing all runtime dependencies with minimum versions:
   ```
   [project.dependencies]
   numpy >= 1.22
   scipy >= 1.9
   xarray >= 2022.6
   networkx >= 2.8
   pyvista >= 0.36
   geovista >= 0.4
   vtk >= 9.1
   scikit-learn >= 1.1
   shapely >= 1.8
   rasterio >= 1.3
   pyproj >= 3.4
   cartopy >= 0.21
   matplotlib >= 3.6
   tqdm >= 4.64
   ```
3. Add `[project.optional-dependencies]` for dev/test:
   ```
   [project.optional-dependencies]
   dev = ["pytest >= 7.0", "pytest-cov", "mypy", "ruff"]
   ```
4. Remove Python 3.6, 3.7, 3.8 from the `classifiers` list and the CI matrix. The minimum should be `3.9`.
5. In `.github/workflows/test.yaml`, replace all references to `cookiecutter_python` with `waper`. Update the matrix to `["3.9", "3.10", "3.11", "3.12"]`. Replace the install step with:
   ```yaml
   - name: Install package
     run: python -m pip install -e ".[dev]"
   ```
6. Remove the `environment.yml` file (or keep it as a convenience but document that `pyproject.toml` is the source of truth).

**Definition of Done:**
- [x] `pip install -e ".[dev]"` succeeds in a fresh venv.
- [x] CI workflow references `waper` everywhere, not `cookiecutter_python` or `my_new_project`.
- [x] Python version matrix is 3.9–3.12.

---

### Task 0.2 — Remove Dead Code and Commented-Out Blocks

**Problem:** Multiple files contain large blocks of commented-out code and dead functions. This obscures the actual logic and confuses any reader or LLM.

**Files to edit:**
- `waper/identification/max_min.py`
- `waper/identification/rwp_graph.py`
- `waper/identification/topology.py`
- `waper/tracking/tracking_graph.py`

**Steps:**

1. In `max_min.py`:
   - Delete the entire commented-out function `clip_dataset_min` (approx lines 230–248).
   - Delete the commented-out functions `extract_position_ids_minima`, `extract_position_ids_maxima`, `extract_selection_ids_maxima`, `extract_selection_ids_minima` (approx lines 177–264, everything that is commented out).
   - Delete the function `interpolate_cell_values_min` — it is an exact duplicate of `interpolate_cell_values`. Search the codebase for any calls to `interpolate_cell_values_min` and replace them with `interpolate_cell_values`. (At time of writing there are zero calls.)

2. In `rwp_graph.py`:
   - In `compute_association_graph`, remove all commented-out lines referencing `grad_vector`, `curr_max_dir_deriv`, `curr_min_dir_deriv`, `curr_max_scalar`, `curr_min_scalar`, `curr_max_x`, `curr_min_x`, `point_cords_max`, `point_cords_min`, `point_tuple_max`, `point_tuple_min`, `assoc_index_array`, `line_dir_array`.
   - In `edge_weight`, remove all commented-out lines referencing `high_value_threshold`, `scalar_threshold`, `scalar_tolerance`, `high_value_flag`, `cluster_max_pts`, `cluster_min_pts`, and the commented-out nested loop over cluster points.
   - In `get_ranked_paths`, remove the commented-out `best_path` / `max_weight` / `consistent` logic block and the commented-out `return path_list`.

3. In `topology.py`:
   - In `identify_connected_regions`, remove the commented-out VTK connectivity filter block.

4. In `tracking_graph.py`:
   - Remove all commented-out `print` statements.
   - Remove the commented-out `return track_paths` at the end of `get_track_paths`.

5. Run `git diff` to review all removals are comments/dead code only.

**Definition of Done:**
- [x] No commented-out function bodies remain in any `.py` file.
- [x] No `# print(...)` lines remain.
- [x] `interpolate_cell_values_min` no longer exists.
- [x] Package still imports correctly (`python -c "from waper import Waper"`).

---

### Task 0.3 — Fix `__init__.py` and Smoke Test

**Problem:** `waper/__init__.py` exposes internal submodules (`max_min`, `topology`) at the top level, which is unusual. The smoke test imports `my_new_project`, not `waper`.

**Files to edit:**
- `waper/__init__.py`
- `waper/interface/__init__.py`
- `tests/smoke_test.py`

**Steps:**

1. Edit `waper/__init__.py` to:
   ```python
   from .interface.api import Waper, WaperConfig, WaperSingleTimestepData

   __all__ = ["Waper", "WaperConfig", "WaperSingleTimestepData"]
   ```

2. Edit `waper/interface/__init__.py` to:
   ```python
   from .api import Waper, WaperConfig, WaperSingleTimestepData

   __all__ = ["Waper", "WaperConfig", "WaperSingleTimestepData"]
   ```

3. In `api.py`, remove the line `from waper import tracking` (absolute import that is fragile and unused directly). The existing relative import `from ..tracking import quadtree, tracking_graph` already covers it.

4. Edit `tests/smoke_test.py` to:
   ```python
   def test_smoke_import():
       import waper
       assert waper is not None

   def test_smoke_classes_exist():
       from waper import Waper, WaperConfig
       assert Waper is not None
       assert WaperConfig is not None
   ```

5. Run the smoke test: `pytest tests/smoke_test.py -v`.

**Definition of Done:**
- [x] `from waper import Waper, WaperConfig` works.
- [x] `pytest tests/smoke_test.py` passes.
- [x] No absolute `from waper import ...` inside the package source (only relative imports).

---

### Task 0.4 — Add Logging Framework

**Problem:** Debugging output uses bare `print()` statements scattered throughout. There is a `logging()` function in `api.py` that just calls `print()` and is never used.

**Files to edit:**
- `waper/interface/api.py`
- All files that currently contain `print()` calls.

**Steps:**

1. Delete the standalone `logging` function in `api.py` (the one that takes `log_info, config`).

2. At the top of every module that needs logging, add:
   ```python
   import logging
   logger = logging.getLogger(__name__)
   ```

3. Replace every bare `print(...)` call with an appropriate log level:
   - `print(feature)` in `tracking_graph.py` → `logger.warning("Feature %s has no matching rwp_info", feature)`
   - `print(input_xs, input_ys)` in `rwp_polygon.py` → `logger.error("Stereographic transform failed for xs=%s, ys=%s", input_xs, input_ys)`
   - `print('No RWPs found, change thresholds')` in `api.py` → `logger.warning("No RWPs found at this timestep. Consider adjusting thresholds.")`

4. In the `Waper.__init__` method, add:
   ```python
   if debug:
       logging.basicConfig(level=logging.DEBUG)
   ```

**Definition of Done:**
- [x] Zero bare `print()` calls remain in `waper/` source.
- [x] `Waper(debug=True, ...)` produces log output.
- [x] `Waper(debug=False, ...)` is silent by default.

---

## Phase 1: Testing Infrastructure

These tasks create the test harness that all subsequent phases depend on. **Do this before any algorithmic changes.**

### Task 1.1 — Create Synthetic Test Data Fixtures

**Problem:** There are no tests. Before we can safely refactor, we need deterministic test inputs with known answers.

**Files to create:**
- `tests/conftest.py`
- `tests/fixtures/` directory

**Steps:**

1. Create `tests/conftest.py`.

2. Write a pytest fixture `simple_wave_field` that generates a synthetic 2D meridional wind field as an `xarray.DataArray`:
   ```python
   import numpy as np
   import xarray as xr
   import pytest

   @pytest.fixture
   def simple_wave_field():
       """A synthetic v-wind with 3 clear crests and 2 troughs.

       The field is:  v(lon, lat) = A(lon) * sin(k * lon)
       where A(lon) is a Gaussian envelope centered at lon=180
       and k gives ~4 full wavelengths across 360 degrees.

       Latitudes span 20N to 80N. Longitudes span 0 to 359.
       """
       lons = np.arange(0, 360, 2.5)       # 144 points
       lats = np.arange(20, 80.1, 2.5)     # 25 points
       lon2d, lat2d = np.meshgrid(lons, lats)

       k = 2 * np.pi * 4 / 360  # wavenumber 4
       envelope = 30 * np.exp(-((lon2d - 180) ** 2) / (2 * 40 ** 2))
       v = envelope * np.sin(k * np.radians(lon2d) * 360)

       da = xr.DataArray(
           v, dims=["latitude", "longitude"],
           coords={"latitude": lats, "longitude": lons},
       )
       return da
   ```

3. Write a fixture `two_timestep_field` that returns a dataset with 2 timesteps where the wave packet has shifted ~5° east between them. This is for tracking tests.

4. Write a fixture `single_maximum_field` that is a single isolated Gaussian bump (one clear maximum, no minima above threshold). This is for edge-case tests.

5. Write a fixture `flat_field` that is identically zero everywhere. This is for testing graceful failure.

6. Write a fixture `date_line_wave_field` where the wave packet straddles the 0°/360° boundary. This tests wraparound handling.

**Definition of Done:**
- [x] `tests/conftest.py` exists with at least 5 fixtures.
- [x] Each fixture returns an `xarray.DataArray` with latitude/longitude coordinates.
- [x] `pytest tests/conftest.py --collect-only` shows all fixtures.

---

### Task 1.2 — Unit Tests for Extrema Detection

**File to create:** `tests/test_max_min.py`

**Steps:**

1. Write `test_finds_known_maxima`: Using `simple_wave_field`, call `add_maxima_data` and `extract_maxima_points`. Assert that the number of maxima found equals the expected count (manually compute from the synthetic formula). Assert that all maxima have scalar values > 0.

2. Write `test_finds_known_minima`: Same as above but for minima. Assert values < 0.

3. Write `test_threshold_filters_weak_extrema`: Create a field with one strong max (v=30) and one weak max (v=3). Set `extrema_threshold=5`. Assert only the strong maximum survives extraction.

4. Write `test_periodic_boundary_maxima`: Using `date_line_wave_field`, place a maximum at lon=359. Assert it is correctly detected and that the neighbor comparison wraps to lon=0.

5. Write `test_flat_field_no_extrema`: Using `flat_field`, assert zero maxima and zero minima are found.

6. Write `test_maxima_and_minima_do_not_overlap`: For any test field, assert that no grid point is flagged as both a maximum and a minimum.

**Definition of Done:**
- [x] `pytest tests/test_max_min.py -v` passes.
- [x] At least 6 test functions exist.

---

### Task 1.3 — Unit Tests for Clustering

**File to create:** `tests/test_clustering.py`

**Steps:**

1. Write `test_single_extremum_per_region_is_own_cluster`: Create a clipped field with one connected component containing one maximum. Assert it gets cluster ID 0.

2. Write `test_two_close_extrema_same_cluster`: Create a clipped field with two maxima 5° apart in the same connected component. Assert they receive the same cluster ID.

3. Write `test_two_distant_extrema_different_clusters`: Two maxima 60° apart in the same connected component. Assert they receive different cluster IDs.

4. Write `test_isolated_outlier_far_from_group`: Place 5 maxima in a tight group plus 1 maximum 40° away. Assert the outlier is in a different cluster from the group (this will currently FAIL — it documents the AP forcing issue and will be the regression test for Phase 3).

**Definition of Done:**
- [x] `pytest tests/test_clustering.py -v` runs (some tests may be marked `xfail` until Phase 3).
- [x] At least 4 test functions exist.

---

### Task 1.4 — Unit Tests for Association Graph

**File to create:** `tests/test_association_graph.py`

**Steps:**

1. Write `test_alternating_crests_troughs_connected`: Using `simple_wave_field`, run the full identification pipeline up to the association graph. Assert that the graph is bipartite (all edges connect a positive-ID node to a negative-ID node).

2. Write `test_isolated_max_no_adjacent_min`: Create a field with one crest but no trough above threshold. Assert the association graph is empty.

3. Write `test_node_pruning_removes_weak_nodes`: Build an association graph, then prune. Assert nodes with scalar below threshold are gone.

4. Write `test_edge_pruning_removes_low_gradient`: Build and prune. Assert only edges above the gradient threshold survive.

5. Write `test_date_line_association`: Using `date_line_wave_field`, assert that crests/troughs near 0°/360° are correctly associated.

**Definition of Done:**
- [x] `pytest tests/test_association_graph.py -v` passes (some may be `xfail`).
- [x] At least 5 test functions exist.

---

### Task 1.5 — Unit Tests for Tracking

**File to create:** `tests/test_tracking.py`

**Steps:**

1. Write `test_identical_timesteps_full_overlap`: Run identification on the same field twice. Build tracking graph. Assert overlap weight is 1.0 for matching features.

2. Write `test_shifted_field_partial_overlap`: Using `two_timestep_field`, assert overlap weight is between 0 and 1.

3. Write `test_no_overlap_no_edge`: Two timesteps with features in completely different hemispheres. Assert no edges in tracking graph.

4. Write `test_tracking_path_extraction`: Build a tracking graph with 3 timesteps. Assert the extracted path spans all 3.

5. Write `test_quadtree_pixel_counts`: Create a known raster, build quadtree, call `compute_pixels`. Assert pixel counts match expected values.

**Definition of Done:**
- [x] `pytest tests/test_tracking.py -v` passes (some may be `xfail`).
- [x] At least 5 test functions exist.

---

### Task 1.6 — Integration Test

**File to create:** `tests/test_integration.py`

**Steps:**

1. Write `test_full_pipeline_synthetic`: Instantiate `Waper` with `simple_wave_field` (duplicated to 2 timesteps). Call `identify_rwps()`. Assert at least 1 RWP path is found. Call `track_rwps()`. Assert no crash. This is the canary test that the whole pipeline runs end-to-end.

2. Write `test_full_pipeline_flat_field_graceful`: Instantiate with `flat_field`. Assert `identify_rwps()` completes without crash and finds 0 RWPs.

**Definition of Done:**
- [x] `pytest tests/test_integration.py -v` passes.
- [x] Tests run in < 60 seconds on a modern laptop.

---

## Phase 2: Critical Bug Fixes

These are correctness issues that should be fixed before any refactoring. Each fix should be accompanied by a test that would have caught the bug.

### Task 2.1 — Fix Extrema Detection (`max_min.py`)

**Problem 1 — `check` array causes missed extrema:** When point (i,j) is compared to its neighbors, any neighbor that is ≤ the current value gets `check[x][y] = 1`, marking it as "visited." But that neighbor might itself be a valid local maximum relative to *its own* neighbors. The `check` optimisation is incorrect and causes false negatives.

**Problem 2 — `if/if/else` instead of `if/elif/else` for boundary handling:** The `j == 0` block is an `if`, the `j == c-1` block is also an `if` (not `elif`), and the generic case is `else` attached to the second `if`. This means when `j == 0`, the code enters the first block AND then falls through to the `else` block (since `j == 0` is not `c-1`). The point is checked twice with inconsistent neighbor sets.

**Files to edit:**
- `waper/identification/max_min.py`

**Steps:**

1. **Replace the entire body** of `add_maxima_data` with a vectorised implementation:
   ```python
   from scipy.ndimage import maximum_filter

   def add_maxima_data(scalar_values, scalar_name, longitudes, latitudes):
       lons = np.linspace(0, 360, len(longitudes))
       lats = latitudes
       grid_vtk = get_vtk_object_from_data_array(scalar_values, lons, lats, scalar_name)

       numpy_data = scalar_values.values
       r, c = numpy_data.shape

       # maximum_filter with wrap mode on the longitude axis handles periodicity
       local_max = maximum_filter(
           numpy_data, size=3, mode=['constant', 'wrap']
       )
       is_max = (numpy_data == local_max).astype(float)

       # The above finds plateaux too; keep only strict local maxima
       # (equal to the filter output AND not a flat region)
       # For a flat region, all neighbors equal the center, so local_max == center everywhere.
       # We accept this as a maximum only if the value is nonzero.
       # The threshold filtering downstream will remove insignificant ones.

       vertex_identifiers = np.arange(1, r * c + 1, dtype=float)

       cell_number = grid_vtk.GetNumberOfCells()
       cell_id = np.arange(cell_number)

       grid_vtk.point_data["is max"] = is_max.ravel()
       grid_vtk.point_data["Vertex_id"] = vertex_identifiers
       grid_vtk.cell_data["{} Cell ID".format(scalar_name)] = cell_id

       return grid_vtk
   ```

2. **Do the same for `add_minima_data`** using `minimum_filter`:
   ```python
   from scipy.ndimage import minimum_filter

   def add_minima_data(scalar_values, scalar_name, longitudes, latitudes):
       lons = np.linspace(0, 360, len(longitudes))
       lats = latitudes
       grid_vtk = get_vtk_object_from_data_array(scalar_values, lons, lats, scalar_name)

       numpy_data = scalar_values.values
       r, c = numpy_data.shape

       local_min = minimum_filter(
           numpy_data, size=3, mode=['constant', 'wrap']
       )
       is_min = (numpy_data == local_min).astype(float)

       # Exclude the top row (i == 0), matching original behavior
       is_min[0, :] = 0

       vertex_identifiers = np.arange(1, r * c + 1, dtype=float)

       cell_number = grid_vtk.GetNumberOfCells()
       cell_id = np.arange(cell_number)

       grid_vtk.point_data["is min"] = is_min.ravel()
       grid_vtk.point_data["Vertex_id"] = vertex_identifiers
       grid_vtk.cell_data["{} Cell ID".format(scalar_name)] = cell_id

       return grid_vtk
   ```

3. Note on `mode` parameter: `['constant', 'wrap']` means axis 0 (latitude) uses constant padding (no wrap at poles) and axis 1 (longitude) uses wrap (periodic boundary). This exactly matches the intended behavior.

4. Run `pytest tests/test_max_min.py -v` — all tests from Task 1.2 should pass.

5. Run `pytest tests/test_integration.py -v` — the full pipeline should still work.

**Definition of Done:**
- [x] No Python `for` loops remain in `add_maxima_data` or `add_minima_data`.
- [x] `test_periodic_boundary_maxima` passes.
- [x] `test_flat_field_no_extrema` passes.
- [x] `test_maxima_and_minima_do_not_overlap` passes.
- [ ] Integration test passes.

---

### Task 2.2 — Fix Node ID Collision (`min_id == 0` Hack)

**Problem:** In `compute_association_graph`, minima cluster IDs are negated to distinguish them from maxima cluster IDs. Cluster ID 0 maps to node 0, which collides with max cluster 0. The code hacks around this by remapping `min_id = 0` to `min_id = 100`, which breaks if there are ≥100 max clusters.

**Files to edit:**
- `waper/identification/rwp_graph.py`

**Steps:**

1. Change the node ID scheme from plain integers to tuples. A max cluster with ID `k` becomes node `("max", k)`. A min cluster with ID `k` becomes node `("min", k)`.

2. In `compute_association_graph`:
   - Remove the `if min_id == 0: min_id = 100` block and the `if min_id == 100:` special case.
   - When adding nodes:
     ```python
     max_node_id = ("max", max_id)
     min_node_id = ("min", min_id)

     assoc_graph.add_node(
         max_node_id,
         coords=max_centre,
         spherical_coords=max_centre_spherical,
         cluster_id=max_id,
         scalar=max_scalar,
         node_type="max",
         cluster_extrema=cluster_max_dict[max_id],
     )
     assoc_graph.add_node(
         min_node_id,
         coords=min_centre,
         spherical_coords=min_centre_spherical,
         cluster_id=min_id,
         scalar=min_scalar,
         node_type="min",
         cluster_extrema=cluster_min_dict[min_id],
     )
     assoc_graph.add_edge(max_node_id, min_node_id, weight=0)
     ```

3. In `prune_association_graph_nodes`:
   - Replace `if start_node >= 0:` with `if assoc_graph.nodes[start_node]["node_type"] == "max":`.

4. In `edge_weight`:
   - The function currently receives `max_id` and `min_id` — these are now tuple node IDs. The body doesn't depend on the sign of the ID, only on `assoc_graph.nodes[max_id]["scalar"]`, so no change needed to the body.

5. In `prune_association_graph_edges`:
   - Replace `if start_node >= 0:` with `if assoc_graph.nodes[start_node]["node_type"] == "max":`.

6. In `get_ranked_paths`:
   - The `is_to_the_east` call uses `assoc_graph.nodes[source]["coords"][0]` which doesn't depend on node ID format. No change needed.

7. In `rwp_polygon.get_polygon_for_rwp_path`:
   - Replace `if node > 0:` with `if node[0] == "max":`.

8. In `visualization.py` `_plot_rwp_paths`:
   - Replace `if node < 0:` with:
     ```python
     if isinstance(node, tuple):
         color = 'r' if node[0] == 'max' else 'b'
     else:
         color = 'r' if node >= 0 else 'b'
     ```
     (The `else` branch handles the tracking graph, which uses `(time, feature)` tuples with a different structure.)

9. In `visualization.py` `_plot_clusters`:
   - Replace the `if cluster_id == 0: cluster_id = 100` block with just a plain negative sign for display: `str(-cluster_id)`.

10. Search the entire codebase for any remaining `if ... >= 0` or `if ... < 0` or `if ... > 0` checks on node IDs and update them.

**Definition of Done:**
- [ ] The string `min_id = 100` no longer appears anywhere.
- [ ] No node ID is a plain integer in the association graph.
- [ ] `test_alternating_crests_troughs_connected` passes.
- [ ] Integration test passes.

---

### Task 2.3 — Fix `is_to_the_east` Missing Return and Wrong Variable Name

**Problem:** The function returns `True` when `lon1` is east of `lon2`, but implicitly returns `None` (falsy) otherwise. Variable named `delta_lat` should be `delta_lon`.

**File to edit:**
- `waper/identification/utils.py`

**Steps:**

1. Replace the function with:
   ```python
   def is_to_the_east(lon1, lon2):
       """Return True if lon1 is to the east of lon2, handling wraparound."""
       delta_lon = lon1 - lon2

       if abs(delta_lon) > 180:
           delta_lon = -delta_lon

       return delta_lon > 0
   ```

2. Write a test in `tests/test_utils.py`:
   ```python
   from waper.identification.utils import is_to_the_east

   def test_east_simple():
       assert is_to_the_east(10, 5) is True

   def test_west_simple():
       assert is_to_the_east(5, 10) is False

   def test_same_longitude():
       assert is_to_the_east(10, 10) is False

   def test_wraparound_east():
       assert is_to_the_east(5, 355) is True   # 5° is 10° east of 355°

   def test_wraparound_west():
       assert is_to_the_east(355, 5) is False
   ```

**Definition of Done:**
- [ ] `is_to_the_east` always returns a `bool`.
- [ ] Variable is named `delta_lon`.
- [ ] All 5 tests pass.

---

### Task 2.4 — Fix Euclidean Distance in Association Graph

**Problem:** In `compute_association_graph`, the nearest max/min to each isocontour point is found using 2D Euclidean distance on (x, y) components of the 3D spherical mesh coordinates, ignoring z. This is inconsistent with the spherical geometry intent and can give wrong nearest-neighbor results near poles.

**File to edit:**
- `waper/identification/rwp_graph.py`

**Steps:**

1. Import at the top of the file:
   ```python
   from scipy.spatial import cKDTree
   ```

2. In `compute_association_graph`, before the contour-point loop, build KD-trees on the full 3D Cartesian coordinates:
   ```python
   max_tree = cKDTree(max_points_array)  # shape (num_max_pts, 3)
   min_tree = cKDTree(min_points_array)  # shape (num_min_pts, 3)
   ```

3. Replace the inner loops over max/min points with:
   ```python
   for i in range(num_contour_pts):
       contour_point = contour_points[i]

       max_dist, max_idx = max_tree.query(contour_point)
       max_id = int(max_cluster_ids[max_idx])

       min_dist, min_idx = min_tree.query(contour_point)
       min_id = int(min_cluster_ids[min_idx])

       if max_id != -1 and min_id != -1:
           assoc_set.add((max_id, min_id))
   ```

4. This eliminates the O(C × M) + O(C × N) nested loops entirely. Each `query` call is O(log N).

5. Optionally, vectorise fully:
   ```python
   _, max_indices = max_tree.query(contour_points)
   _, min_indices = min_tree.query(contour_points)

   for i in range(num_contour_pts):
       max_id = int(max_cluster_ids[max_indices[i]])
       min_id = int(min_cluster_ids[min_indices[i]])
       assoc_set.add((max_id, min_id))
   ```

**Definition of Done:**
- [ ] No 2-component Euclidean distance calculation remains in `compute_association_graph`.
- [ ] KD-tree uses all 3 Cartesian components.
- [ ] `test_date_line_association` passes.
- [ ] Integration test passes.

---

### Task 2.5 — Fix Feature ID Collision (Rounded Scalar as ID)

**Problem:** `polygon_id = round(path_max, 2)` means two RWPs with the same peak scalar value get the same raster label. One overwrites the other in the raster, and the tracking graph silently loses a feature.

**Files to edit:**
- `waper/tracking/rwp_polygon.py`
- `waper/interface/api.py`
- `waper/tracking/tracking_graph.py`

**Steps:**

1. In `api.py`, in the `_identify_rwps` function, change the loop that creates `rwp_info`:
   ```python
   for index, path in enumerate(time_step_data.identified_rwp_paths):
       (
           polygon,
           _unused_id,
           sample_points,
           weighted_lon,
           weighted_lat,
       ) = rwp_polygon.get_polygon_for_rwp_path(
           path, time_step_data.pruned_graph, time_step_data.vtk_data, config.scalar_name,
           config.min_latitude, config.max_latitude
       )
       # Use a unique monotonic index starting from 1
       rwp_id = index + 1
       time_step_data.rwp_info[tuple(path)] = {
           "polygon": polygon,
           "rwp_id": rwp_id,
           "sample_points": sample_points,
           "weighted_longitude": weighted_lon,
           "weighted_latitude": weighted_lat,
       }
   ```

2. Update `list_polygons` construction to use the new integer ID:
   ```python
   list_polygons.append((
       time_step_data.rwp_info[tuple(path)]["polygon"],
       time_step_data.rwp_info[tuple(path)]["rwp_id"],
   ))
   ```

3. In `rwp_polygon.py`, change `get_polygon_for_rwp_path` to return `None` for `polygon_id` (or simply remove it from the return tuple and adjust callers). The ID is now assigned externally.

4. In `tracking_graph.py`, the feature lookup `abs(feature - rwp_info["rwp_id"]) < 1e-2` now compares integers, so change to exact equality:
   ```python
   if feature == rwp_info["rwp_id"]:
   ```

5. Write a test: create two RWP paths with the same peak scalar value, assert they get different `rwp_id`s, and assert both appear in the raster with distinct labels.

**Definition of Done:**
- [ ] `rwp_id` is a unique integer per timestep.
- [ ] No `round(path_max, 2)` remains as an ID source.
- [ ] Two RWPs with identical peak values are distinguishable in the raster.
- [ ] Integration test passes.

---

### Task 2.6 — Fix Bare `except` in Stereographic Transform

**Problem:** `rwp_polygon.py` has `except:` which catches `KeyboardInterrupt`, `SystemExit`, etc., and re-raises a generic `ValueError` with no message, destroying the traceback.

**File to edit:**
- `waper/tracking/rwp_polygon.py`

**Steps:**

1. Replace:
   ```python
   try:
       return transformer.transform(input_xs, input_ys, errcheck=True)
   except:
       print(input_xs, input_ys)
       raise ValueError()
   ```
   with:
   ```python
   try:
       return transformer.transform(input_xs, input_ys, errcheck=True)
   except Exception as e:
       logger.error(
           "Stereographic transform failed for xs=%s, ys=%s: %s",
           input_xs, input_ys, e,
       )
       raise ValueError(
           f"Stereographic transform failed: {e}"
       ) from e
   ```

**Definition of Done:**
- [ ] No bare `except:` remains anywhere in the codebase.
- [ ] The re-raised error includes the original exception message.

---

## Phase 3: Algorithmic Improvements — Identification

These tasks change the scientific behavior of the identification step. Each must be validated against the synthetic test fixtures AND against real ERA data (manually, comparing plots before/after).

### Task 3.1 — Replace Affinity Propagation with DBSCAN/HDBSCAN

**Problem:** Affinity Propagation forces every extremum into a cluster. Isolated outlier extrema far from the primary group inflate the cluster footprint, making crests/troughs look artificially large. AP also has a global `preference` parameter (`median_dist / 5.0`) that doesn't adapt to per-region structure.

**File to edit:**
- `waper/identification/topology.py`

**Steps:**

1. Add `from sklearn.cluster import DBSCAN` (or `from hdbscan import HDBSCAN` if the dependency is acceptable; DBSCAN is already in sklearn).

2. Create a new unified function `cluster_extrema` that replaces both `cluster_max` and `cluster_min`:
   ```python
   def cluster_extrema(
       base_field,
       connectivity_clipped_scalar_field,
       extrema_points,
       scalar_name,
       sign,   # +1 for maxima, -1 for minima
       eps_km=500,    # DBSCAN neighborhood radius in km
       min_samples=1, # minimum cluster size
   ):
   ```

3. Inside the function, for each connected region:
   - If the region has 1 extremum: assign it to its own cluster (same as before).
   - If the region has ≥2 extrema: run DBSCAN with `eps=eps_km` (converted to the appropriate distance unit for the distance matrix) and `min_samples=min_samples` on the precomputed distance matrix.
   - DBSCAN returns label `-1` for noise points. **Discard noise points** — do not assign them to any cluster.

4. The DBSCAN `eps` parameter has a clear physical interpretation: "two extrema belong to the same crest/trough if they are within `eps` km of each other on the sphere." Expose this as `cluster_eps_km` in `WaperConfig` with a default of 500 km.

5. The `min_samples` parameter controls the minimum number of extrema to form a cluster. Default to 1 (any single point can be a cluster, which matches the original behavior for isolated extrema).

6. Remove `cluster_max` and `cluster_min`. Update all callers in `api.py` to call `cluster_extrema(..., sign=+1)` and `cluster_extrema(..., sign=-1)`.

7. Remove the two-point special case (`if len(region_array[k]) == 2: always merge`). DBSCAN handles this naturally: if two points are within `eps`, they cluster; if not, one or both become noise.

8. Add `cluster_eps_km` and `cluster_min_samples` to `WaperConfig`.

9. Run the clustering tests from Task 1.3. The `test_isolated_outlier_far_from_group` test should now PASS (remove the `xfail` mark).

**Definition of Done:**
- [ ] `cluster_max` and `cluster_min` no longer exist.
- [ ] `cluster_extrema` uses DBSCAN.
- [ ] Noise points (label -1) are excluded from all downstream processing.
- [ ] `WaperConfig` has `cluster_eps_km` and `cluster_min_samples`.
- [ ] `test_isolated_outlier_far_from_group` passes.
- [ ] Integration test passes.

---

### Task 3.2 — Fix the Similarity Penalty (`FindCellsAlongLine` Through Sphere Interior)

**Problem:** In `topology.py`, the similarity matrix penalty for clustering uses `vtkCellLocator.FindCellsAlongLine` to find cells between two extrema. This shoots a straight line through 3D Cartesian space, which passes through the *interior* of the sphere, not along the surface. For widely-separated or near-polar points, the ray may not intersect any surface cells, leaving `min_v` at its initialised value of 1000, which massively corrupts the distance.

**File to edit:**
- `waper/identification/topology.py`

**Steps — short-term fix (if keeping VTK for now):**

1. Replace `FindCellsAlongLine` with a surface-based approach: for each pair of extrema (i, j), the Dijkstra path already provides the sequence of vertices along the shortest surface path. Walk this vertex sequence and sample the scalar value at each vertex. Use the minimum (for maxima) or maximum (for minima) sampled value as the penalty.

2. Concretely, after the Dijkstra path is computed:
   ```python
   pts = dijkstra.GetOutput().GetPoints()
   scalar_arr = scalar_field.GetPointData().GetArray(scalar_name)
   id_list = dijkstra.GetIdList()

   min_v = float('inf')
   for ptIdx in range(id_list.GetNumberOfIds()):
       vid = id_list.GetId(ptIdx)
       val = scalar_arr.GetTuple1(vid)
       if val < min_v:
           min_v = val
   ```
   This samples along the *surface* path, not through the sphere interior.

3. Change `min_v` initialisation from `1000` to `float('inf')` and `max_v` from `-1000` to `float('-inf')`.

**Steps — long-term fix (Phase 5 replaces VTK entirely):**

This will be superseded by Task 5.2 which replaces VTK Dijkstra with scipy sparse graph. At that point, sampling along the path is trivial since we have the grid indices directly.

**Definition of Done:**
- [ ] `FindCellsAlongLine` is no longer used for the similarity penalty.
- [ ] Penalty is sampled along the surface geodesic path.
- [ ] No hardcoded `1000` / `-1000` initialisations remain.

---

### Task 3.3 — Use Weighted Centroid for Cluster Representative

**Problem:** The cluster representative point is the single grid cell with the most extreme scalar value, not the centroid. This puts the "center" of a multi-point cluster at an arbitrary edge of the group, distorting edge weights and graph structure.

**Files to edit:**
- `waper/identification/topology.py` (functions `max_cluster_assign`, `min_cluster_assign`)

**Steps:**

1. In `max_cluster_assign`, compute a value-weighted centroid for each cluster:
   ```python
   for i in range(num_points_max):
       cid = cluster_id_max[i]
       lon = max_points['Longitude'][i]
       lat = max_points['Latitude'][i]
       val = max_scalars[i]

       max_pt_dict[cid].append([lon, lat])

       # Track peak value (for scalar attribute)
       if cluster_max_arr[cid] < val:
           cluster_max_arr[cid] = val

       # Accumulate for weighted centroid
       cluster_lon_sum[cid] += lon * val
       cluster_lat_sum[cid] += lat * val
       cluster_weight_sum[cid] += val

   # Compute weighted centroid
   for cid in range(num_max_clusters):
       if cluster_weight_sum[cid] > 0:
           cluster_max_point[cid][0] = cluster_lon_sum[cid] / cluster_weight_sum[cid]
           cluster_max_point[cid][1] = cluster_lat_sum[cid] / cluster_weight_sum[cid]
   ```

2. **Longitude wraparound:** If a cluster straddles 0°/360°, naive averaging fails. Before accumulating, shift all longitudes in the cluster to be continuous (e.g., if the cluster contains both 350° and 10°, shift 10° to 370° before averaging, then wrap result back). Use `get_consistent_longitudes` from `rwp_polygon.py` or implement a simple version.

3. Do the same for `min_cluster_assign`, weighting by `abs(val)`.

4. The `cluster_max_arr` / `cluster_min_arr` (peak value) should remain unchanged — it's used for pruning. Only the `cluster_max_point` / `cluster_min_point` (representative position) changes.

**Definition of Done:**
- [ ] Cluster representative is a weighted centroid, not the peak-value point.
- [ ] Longitude wraparound is handled.
- [ ] `test_two_close_extrema_same_cluster` uses the centroid as the representative.
- [ ] Integration test passes.

---

### Task 3.4 — Fix Path Ranking to Solve Maximum-Weight Independent Set

**Problem:** The `top_paths` filter in `get_ranked_paths` keeps a path only if no overlapping path has higher weight. This is not transitive: path B can be eliminated by path A, which is then eliminated by path C, even though B and C don't overlap. Valid RWPs are dropped through indirect competition.

**File to edit:**
- `waper/identification/rwp_graph.py`

**Steps:**

1. Replace the current filter with a greedy maximum-weight independent set algorithm:
   ```python
   def get_ranked_paths(assoc_graph, max_weight):
       """Extract non-overlapping paths with maximum total weight."""

       # Step 1: Enumerate all candidate paths (same as before)
       path_list = _enumerate_all_candidate_paths(assoc_graph)

       # Step 2: Compute weight for each path
       path_weights = {}
       for path in path_list:
           w = sum(
               assoc_graph[path[i]][path[i+1]]["weight"]
               for i in range(len(path) - 1)
           )
           path_weights[tuple(path)] = w

       # Step 3: Sort paths by weight descending
       sorted_paths = sorted(path_list, key=lambda p: path_weights[tuple(p)], reverse=True)

       # Step 4: Greedy selection — pick highest-weight path, remove all
       #         paths that share a node with it, repeat.
       selected = []
       used_nodes = set()
       for path in sorted_paths:
           path_nodes = set(path)
           if path_nodes.isdisjoint(used_nodes):
               selected.append(path)
               used_nodes.update(path_nodes)

       return selected
   ```

2. Extract the path enumeration into a helper `_enumerate_all_candidate_paths` to keep the main function clean. This is the existing loop over source/sink pairs with `nx.all_simple_paths`.

3. Apply the same fix to `get_track_paths` in `tracking_graph.py`.

**Definition of Done:**
- [ ] Path selection is greedy by weight, not pairwise filter.
- [ ] A path that doesn't overlap with the winner is never dropped.
- [ ] Integration test passes.

---

### Task 3.5 — Fix Longitude Delta Pruning for Wraparound

**Problem:** `prune_association_graph_edges` uses `abs(lon_0 - lon_1) <= WAPER_MIN_LON_DELTA` which doesn't handle the 0°/360° boundary. Clusters at 358° and 2° compute `abs(358 - 2) = 356` instead of the true 4°.

**File to edit:**
- `waper/identification/rwp_graph.py`

**Steps:**

1. Create a helper function:
   ```python
   def _longitude_separation(lon1, lon2):
       """Compute the shortest angular separation in degrees, handling wraparound."""
       delta = abs(lon1 - lon2) % 360
       return min(delta, 360 - delta)
   ```

2. Replace `abs(lon_0 - lon_1)` with `_longitude_separation(lon_0, lon_1)`.

3. Move `WAPER_MIN_LON_DELTA` into `WaperConfig` as `min_longitude_separation` with default 6.

4. Write tests in `tests/test_utils.py`:
   ```python
   def test_lon_separation_normal():
       assert _longitude_separation(10, 20) == 10

   def test_lon_separation_wraparound():
       assert _longitude_separation(358, 2) == 4

   def test_lon_separation_symmetric():
       assert _longitude_separation(2, 358) == 4
   ```

**Definition of Done:**
- [ ] `_longitude_separation` handles wraparound.
- [ ] All 3 tests pass.
- [ ] Integration test passes.

---

### Task 3.6 — Fix Radius Inconsistency

**Problem:** `utils.py` defines `RADIUS_SPHERE = 63.71` (used for VTK mesh construction) and `RADIUS_EARTH = 6.371e6` (used in haversine). The VTK Dijkstra distances are computed on a sphere of radius 63.71, but haversine returns meters. These distance scales differ by a factor of 100,000, which means the clustering similarity matrix (Dijkstra distances) and the edge weight computation (haversine distances) are on completely different scales.

**File to edit:**
- `waper/identification/utils.py`
- `waper/identification/topology.py` (if the `CLUSTER_MAX_DISTANCE` depends on the scale)

**Steps:**

1. Decide on a single distance convention. The simplest is: keep `RADIUS_SPHERE = 63.71` for VTK mesh visualisation, but normalize all distance computations to kilometres.

2. In `haversine_distance`, return distance in **kilometres**:
   ```python
   RADIUS_EARTH_KM = 6371.0

   def haversine_distance(lat1, lon1, lat2, lon2):
       """Return great-circle distance in kilometres."""
       dlat = math.radians(lat1 - lat2)
       dlon = math.radians(lon1 - lon2)
       a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
       c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
       return RADIUS_EARTH_KM * c
   ```

3. In `topology.py`, scale the Dijkstra distances from the VTK mesh to kilometres:
   ```python
   SCALE_FACTOR = RADIUS_EARTH_KM / RADIUS_SPHERE  # ≈ 100.0
   ```
   Multiply all `dist_matrix` entries by `SCALE_FACTOR` after computing them.

4. Update `CLUSTER_MAX_DISTANCE` to be in kilometres (currently 150 — this would be ~150 × 100 = 15,000 km in real units, which is unreasonable. It's probably meant to be 150 on the scaled sphere, i.e. ~15,000 km. Verify against the science and set appropriately — likely `CLUSTER_MAX_DISTANCE = 15000` in km).

5. Update `WAPER_MAX_NODE_DISTANCE` similarly.

6. In `tracking_graph.py`, remove the `distance / 1000` division (line ~80) since haversine now returns km directly.

**Definition of Done:**
- [ ] All distance values throughout the code are in kilometres.
- [ ] `haversine_distance` returns km.
- [ ] `CLUSTER_MAX_DISTANCE` is documented with its unit.
- [ ] Integration test passes.

---

### Task 3.7 — Replace DBSCAN with OPTICS for Multi-Scale Clustering

**Problem:** DBSCAN with a fixed `eps_km` applies a single density scale globally. Atmospheric wavelengths vary between timesteps, between different RWPs in the same timestep, and seasonally. A fixed radius either merges distinct wave components (eps too large) or fragments large-scale ones (eps too small). With `min_samples=1`, DBSCAN reduces to single-linkage clustering with a hard cutoff, offering no density-based separation between close but distinct features.

OPTICS (Ordering Points To Identify the Clustering Structure) computes a reachability ordering that discovers clusters at varying density scales without committing to a single `eps`.

**Files edited:**
- `waper/identification/topology.py`
- `waper/interface/api.py`

**Changes made:**

1. In `topology.py`, replaced `DBSCAN` with `OPTICS` in `cluster_extrema`:
   - `max_eps_km=1500.0` replaces `eps_km=500.0`
   - Added `xi=0.05` parameter for cluster boundary detection
   - Noise filtering (label `-1`) preserved

2. In `api.py`, updated `WaperConfig`:
   - `cluster_max_eps_km: float = 1500.0` replaces `cluster_eps_km: float = 500.0`
   - Added `cluster_xi: float = 0.05`

**Remaining investigation (requires real data):**

The implementation is complete, but the default parameters (`max_eps_km`, `xi`, `min_samples`) need tuning on real data. Follow the procedure in `conductor/clustering_investigation_plan.md`:

1. Run on 3–5 known synoptic events and verify results are reasonable.
2. If `min_samples=1` produces a flat reachability plot, test `min_samples=2` with a slightly lowered `extrema_threshold`.
3. Run the parameter sensitivity analysis (Section 3 of Pandey et al. 2020) to verify stability.

**Definition of Done:**
- [x] OPTICS implementation complete, parameters exposed in `WaperConfig`.
- [x] Integration test passes.
- [ ] Parameters tuned on real data failure cases.
- [ ] Parameter sensitivity analysis shows stable results.

---

## Phase 4: Algorithmic Improvements — Tracking

### Task 4.1 — Use Concave Hull (Alpha Shape) Instead of Convex Hull

**Problem:** `MultiPoint(...).convex_hull` creates a single convex polygon for the entire RWP path. Since an RWP alternates between positive and negative lobes, the convex hull fills in the gaps, creating an unrealistically large footprint. This inflates overlap during tracking.

**File to edit:**
- `waper/tracking/rwp_polygon.py`

**Steps:**

1. Install/import `shapely`:
   ```python
   from shapely.ops import unary_union
   from shapely.geometry import MultiPoint
   ```

2. If using Shapely ≥ 2.0, use `shapely.concave_hull`:
   ```python
   from shapely import concave_hull

   points = MultiPoint(list(zip(xs, ys)))
   rwp_poly = concave_hull(points, ratio=0.3)  # ratio 0-1, lower = more concave
   ```
   If Shapely < 2.0, use `alphashape` package or compute per-node convex hulls and take their union:
   ```python
   per_node_hulls = []
   for node_points in node_point_groups:
       if len(node_points) >= 3:
           per_node_hulls.append(MultiPoint(node_points).convex_hull)
   rwp_poly = unary_union(per_node_hulls)
   ```

3. The second approach (per-node union) is scientifically better because each crest/trough gets its own hull. Refactor `get_polygon_for_rwp_path` to collect points **per node** and create per-node polygons, then union them.

4. Expose the hull method as a config option: `WaperConfig.hull_method = "concave" | "convex" | "per_node"`.

**Definition of Done:**
- [ ] Default footprint is not a single convex hull.
- [ ] Footprints visually match the RWP structure (narrow along the wave, not filled in).
- [ ] Integration test passes.
- [ ] Tracking results are not degraded.

---

### Task 4.2 — Support Southern Hemisphere

**Problem:** `transform_to_stereographic` hardcodes North Pole stereographic projection. The raster bounds are hardcoded for NH. The `TODO` comment acknowledges this.

**File to edit:**
- `waper/tracking/rwp_polygon.py`

**Steps:**

1. Add a `hemisphere` parameter to `transform_to_stereographic`:
   ```python
   def transform_to_stereographic(input_xs, input_ys, hemisphere="north", inverse=False):
       from_crs = pyproj.crs.CRS(4326)
       if hemisphere == "north":
           to_crs = pyproj.crs.CRS("+proj=stere +lat_0=90 +lon_0=0")
       elif hemisphere == "south":
           to_crs = pyproj.crs.CRS("+proj=stere +lat_0=-90 +lon_0=0")
       else:
           raise ValueError(f"hemisphere must be 'north' or 'south', got '{hemisphere}'")
       ...
   ```

2. Compute `WAPER_X_BOUNDS` and `WAPER_Y_BOUNDS` dynamically from the hemisphere, or make them functions of `hemisphere`.

3. Add `hemisphere` to `WaperConfig` with default `"north"`.

4. Thread the hemisphere through all calls to `transform_to_stereographic`.

5. Write a test: create a wave field in the Southern Hemisphere (latitudes -20 to -80), run the pipeline, assert no crash and polygons are in the correct hemisphere.

**Definition of Done:**
- [ ] `WaperConfig.hemisphere` exists, defaults to `"north"`.
- [ ] SH test passes.
- [ ] NH behavior is unchanged.

---

### Task 4.3 — Replace `all_simple_paths` in Tracking with DAG Longest Path

**Problem:** `get_track_paths` uses `nx.all_simple_paths` which has factorial worst-case complexity. The tracking graph is a DAG (edges go strictly forward in time), so a linear-time DP longest-path algorithm exists.

**File to edit:**
- `waper/tracking/tracking_graph.py`

**Steps:**

1. Replace `get_track_paths` with:
   ```python
   def get_track_paths(tracking_graph):
       """Extract tracks as longest-weight paths in the tracking DAG."""

       # Topological sort (linear time)
       topo_order = list(nx.topological_sort(tracking_graph))

       # DP: for each node, store (best_weight_to_here, predecessor)
       best_weight = {node: 0 for node in topo_order}
       predecessor = {node: None for node in topo_order}

       for node in topo_order:
           for succ in tracking_graph.successors(node):
               edge_wt = tracking_graph[node][succ]["weight"]
               candidate = best_weight[node] + edge_wt
               if candidate > best_weight[succ]:
                   best_weight[succ] = candidate
                   predecessor[succ] = node

       # Extract paths by backtracking from end nodes
       end_nodes = [n for n in tracking_graph if tracking_graph.out_degree(n) == 0]

       track_paths = []
       for end in end_nodes:
           path = [end]
           current = end
           while predecessor[current] is not None:
               current = predecessor[current]
               path.append(current)
           path.reverse()
           if len(path) > 1:
               track_paths.append(path)

       # Deduplicate: if two paths share segments, keep the higher-weight one
       # (greedy independent set, same as Task 3.4)
       return _greedy_select_independent_paths(track_paths, tracking_graph)
   ```

2. Complexity: O(V + E), versus O(V! / (V-k)!) worst case for `all_simple_paths`.

3. Write a benchmark test with a synthetic tracking graph of 20 timesteps × 5 features each, and assert it completes in < 1 second.

**Definition of Done:**
- [ ] `nx.all_simple_paths` is no longer called in `tracking_graph.py`.
- [ ] Tracking of 20 timesteps completes in < 1 second.
- [ ] Integration test passes.

---

### Task 4.4 — Decouple Quadtree Merge from Per-Feature Loop

**Problem:** In `build_tracking_graph`, the quadtree `merge` is called once per timestep (correct), but the `edge_list` is the Cartesian product of all features including 0 (background). Also, the merge is recomputed for every feature node at the current timestep, even though it only depends on the pair of timesteps.

**File to edit:**
- `waper/tracking/tracking_graph.py`

**Steps:**

1. Move the merge computation **outside** the feature loop:
   ```python
   if time > 0:
       merge_graph = merge(
           time_step_data[time].quadtree,
           time_step_data[time - 1].quadtree,
       )
       merge_feature_size = compute_size_features(merge_graph)
       prev_feature_size = compute_size_features(time_step_data[time - 1].quadtree)
       curr_feature_size = compute_size_features(time_step_data[time].quadtree)
   ```

2. Filter out feature 0 from the Cartesian product:
   ```python
   prev_features = time_step_data[time - 1].raster_features - {0}
   curr_features = time_step_data[time].raster_features - {0}
   edge_list = list(product(prev_features, curr_features))
   ```

3. This is both a correctness fix (avoids creating spurious edges involving the background) and a performance fix (merge is computed once, not once per feature).

**Definition of Done:**
- [ ] `merge` is called exactly once per pair of consecutive timesteps.
- [ ] Feature 0 is never in the Cartesian product.
- [ ] Integration test passes.

---

## Phase 5: VTK-to-PyVista / SciPy Refactor

The current code mixes raw VTK API calls (e.g., `vtk.vtkDijkstraGraphGeodesicPath`, `vtk.vtkGeometryFilter`, `vtk.vtkTriangleFilter`, `vtk.vtkCellLocator`, `vtk.vtkContourFilter`, `vtk.vtkGradientFilter`, `vtk.vtkConnectivityFilter`, `vtk.vtkIntArray`, `vtk.vtkFloatArray`, `vtk.vtkIdList`) with PyVista's high-level API. The goal is to eliminate all raw VTK calls and use either PyVista wrappers or SciPy equivalents.

### Task 5.1 — Replace VTK Contour and Gradient Filters

**File to edit:**
- `waper/identification/utils.py`

**Steps:**

1. Replace `get_iso_contour` with PyVista's `.contour()`:
   ```python
   def get_iso_contour(scalar_field, value, scalar_name):
       """Extract isocontour at given value."""
       return scalar_field.contour([value], scalars=scalar_name)
   ```
   Note: `api.py` already uses `time_step_data.vtk_data.contour(...)` in one place. The `get_iso_contour` function in `utils.py` may be unused — check and remove if so.

2. Replace `compute_gradients` with PyVista's `.compute_derivative()`:
   ```python
   def compute_gradients(scalar_field, scalar_name):
       return scalar_field.compute_derivative(scalars=scalar_name)
   ```
   Check if this function is called anywhere. If not, delete it.

3. Remove `import vtk` from `utils.py`.

**Definition of Done:**
- [ ] No `vtk.vtkContourFilter` or `vtk.vtkGradientFilter` in the codebase.
- [ ] `import vtk` is removed from `utils.py`.

---

### Task 5.2 — Replace VTK Dijkstra with SciPy Sparse Graph

**File to edit:**
- `waper/identification/topology.py`

This is the largest single refactoring task. Take it in sub-steps.

**Sub-step A: Build a sparse adjacency matrix from the PyVista mesh.**

1. The clipped scalar field is a PyVista `UnstructuredGrid`. Extract its cell connectivity to build a sparse graph:
   ```python
   from scipy.sparse import lil_matrix
   from scipy.sparse.csgraph import shortest_path

   def build_adjacency_matrix(mesh):
       """Build a sparse adjacency matrix from mesh connectivity.

       Edge weights are the Euclidean distances between connected points.
       """
       n = mesh.n_points
       adj = lil_matrix((n, n), dtype=float)

       # Extract edges from cells
       for i in range(mesh.n_cells):
           cell = mesh.get_cell(i)
           point_ids = [cell.point_ids[j] for j in range(cell.n_points)]
           for a_idx in range(len(point_ids)):
               for b_idx in range(a_idx + 1, len(point_ids)):
                   pa = point_ids[a_idx]
                   pb = point_ids[b_idx]
                   dist = np.linalg.norm(mesh.points[pa] - mesh.points[pb])
                   adj[pa, pb] = dist
                   adj[pb, pa] = dist

       return adj.tocsr()
   ```

2. Alternatively, use `mesh.extract_all_edges()` which returns a PolyData of line segments, then parse those.

**Sub-step B: Replace Dijkstra computation.**

1. In `cluster_extrema` (the unified function from Task 3.1), after building the adjacency matrix:
   ```python
   adj = build_adjacency_matrix(mesh)

   # Get point IDs of the extrema
   extrema_ids = extrema_points.point_data["vtkOriginalPointIds"].astype(int)

   # Compute shortest paths between all extrema pairs within each region
   for region_id in range(num_regions):
       region_extrema = [idx for idx in region_indices if regions[idx] == region_id]
       if len(region_extrema) <= 1:
           continue

       # Extract subgraph for this region
       region_point_ids = np.where(all_region_ids == region_id)[0]
       sub_adj = adj[np.ix_(region_point_ids, region_point_ids)]

       # Map extrema IDs to subgraph indices
       id_map = {pid: i for i, pid in enumerate(region_point_ids)}
       local_extrema = [id_map[eid] for eid in extrema_ids_in_region]

       # Compute pairwise shortest paths (only for the extrema rows)
       dist_matrix_sub = shortest_path(sub_adj, indices=local_extrema)
       # dist_matrix_sub[i, j] = shortest path from local_extrema[i] to local_extrema[j]
   ```

**Sub-step C: Replace the similarity penalty.**

1. Along the shortest path (which is now a sequence of grid indices), sample the scalar field values and find the min (for maxima) or max (for minima):
   ```python
   # Use predecessors to reconstruct the path
   dist, predecessors = shortest_path(sub_adj, indices=local_extrema, return_predecessors=True)

   def get_path_penalty(predecessors, source_local, target_local, scalar_values, sign):
       """Walk the shortest path and find the extremal scalar value."""
       path = []
       current = target_local
       while current != source_local:
           path.append(current)
           current = predecessors[source_local, current]
       path.append(source_local)

       path_values = scalar_values[path]
       if sign == +1:  # maxima: penalty is min value along path
           return np.min(path_values)
       else:           # minima: penalty is max value along path
           return np.max(path_values)
   ```

**Sub-step D: Remove all VTK imports from topology.py.**

1. Remove `import vtk`.
2. Remove `vtk.vtkGeometryFilter`, `vtk.vtkTriangleFilter`, `vtk.vtkDijkstraGraphGeodesicPath`, `vtk.vtkCellLocator`, `vtk.vtkIdList`, `vtk.vtkIntArray`.
3. Use `mesh.point_data["Cluster ID"] = cluster_assign` instead of VTK arrays.

**Definition of Done:**
- [ ] `import vtk` is removed from `topology.py`.
- [ ] No VTK Dijkstra, geometry filter, triangle filter, or cell locator remains.
- [ ] Clustering results match (or improve upon) the VTK-based version.
- [ ] Integration test passes.

---

### Task 5.3 — Replace Remaining VTK Calls in `max_min.py`

**File to edit:**
- `waper/identification/max_min.py`

**Steps:**

1. Remove `import vtk`.
2. The `interpolate_cell_values` function uses raw VTK API (`GetNumberOfCells`, `GetCell`, `GetPointId`, `vtkFloatArray`). After Task 2.1, check if this function is still called. If it is, replace with:
   ```python
   def interpolate_cell_values(dataset, scalar_name):
       """Interpolate point data to cell data using PyVista."""
       cell_data = dataset.point_data_to_cell_data()
       dataset.cell_data[f"{scalar_name} Cell Value"] = cell_data[scalar_name]
       return dataset
   ```
3. The `clip_dataset` function already uses PyVista's `clip_scalar` — no change needed.

**Definition of Done:**
- [ ] `import vtk` is removed from `max_min.py`.
- [ ] `interpolate_cell_values` uses PyVista API.

---

### Task 5.4 — Replace VTK Connectivity Filter

**File to edit:**
- `waper/identification/topology.py`

**Steps:**

1. `identify_connected_regions` already uses PyVista:
   ```python
   return dataset.connectivity(largest=False)
   ```
   This is fine. No change needed.

2. `add_connectivity_data_min` uses raw VTK:
   ```python
   connectivity_filter = vtk.vtkConnectivityFilter()
   ```
   Check if this function is called anywhere. If not, delete it. If it is, replace with:
   ```python
   def add_connectivity_data_min(dataset):
       return pv.wrap(dataset).connectivity(largest=False)
   ```

**Definition of Done:**
- [ ] No `vtk.vtkConnectivityFilter` in the codebase.
- [ ] `add_connectivity_data_min` is either deleted or uses PyVista.

---

### Task 5.5 — Replace NetworkX Quadtree with Spatial Index

**Problem:** The quadtree is implemented as a `networkx.DiGraph` with substantial per-node dictionary overhead. For spatial intersection, an R-tree or STRtree operating directly on the polygons would be faster and simpler.

**File to edit:**
- `waper/tracking/quadtree.py`
- `waper/tracking/tracking_graph.py`

**This is a stretch goal** — the current quadtree works correctly (if slowly). Defer to Phase 9 unless performance is a blocking issue.

**Steps (sketch):**

1. Replace the quadtree merge + pixel counting with direct Shapely polygon intersection:
   ```python
   from shapely.ops import unary_union

   def compute_overlap(poly_a, poly_b):
       intersection = poly_a.intersection(poly_b)
       return intersection.area / max(poly_a.area, poly_b.area)
   ```

2. In `build_tracking_graph`, iterate over pairs of polygons from consecutive timesteps and compute overlap directly.

3. Remove `quadtree.py` entirely.
4. Remove `raster_data`, `raster_features`, `quadtree` from `WaperSingleTimestepData`.
5. Remove `rasterize_all_rwps` from `rwp_polygon.py`.
6. Remove `WAPER_IMAGE_SIZE`, `WAPER_NUM_PIXELS`, `WAPER_X_BOUNDS`, `WAPER_Y_BOUNDS`, `WAPER_RASTER_TRANSFORM`.

**Definition of Done:**
- [ ] `quadtree.py` is deleted.
- [ ] Tracking uses direct polygon intersection.
- [ ] Raster-related code is removed.
- [ ] Integration test passes.

---

## Phase 6: Performance Optimisation

### Task 6.1 — Vectorise `interpolate_cell_values`

Already covered by Task 5.3 (using `point_data_to_cell_data()`).

### Task 6.2 — Vectorise Association Graph Construction

Already covered by Task 2.4 (using KD-trees).

### Task 6.3 — Profile and Benchmark

**File to create:** `tests/test_benchmark.py`

**Steps:**

1. Write a benchmark that runs the full pipeline on a synthetic 721×1440 grid (ERA5 resolution) for 1 timestep and measures wall-clock time.

2. Write a benchmark for tracking: 10 timesteps of 721×1440.

3. Use `pytest-benchmark` or simple `time.time()` assertions:
   ```python
   def test_identification_benchmark(large_wave_field):
       import time
       start = time.time()
       result = _identify_rwps(large_wave_field, default_config)
       elapsed = time.time() - start
       assert elapsed < 30  # Should complete in <30s for one timestep
   ```

4. Record baseline timings. After each optimisation task, re-run and verify improvement.

**Definition of Done:**
- [ ] Benchmark tests exist for identification and tracking.
- [ ] Baseline timings are recorded in a comment or markdown file.

---

## Phase 7: Visualisation Overhaul

### Task 7.1 — Make All Plots Accept User-Provided Axes and Projection

**Problem:** Most plot functions create their own `plt.subplot(projection=...)`, preventing integration into user figure layouts. Projections are hardcoded (PlateCarree, Orthographic, Stereographic — inconsistently).

**File to edit:**
- `waper/interface/visualization.py`
- `waper/interface/api.py`

**Steps:**

1. For every `_plot_*` function, ensure the signature accepts `ax=None` and `projection=None`:
   ```python
   def _plot_graph(rwp_graph, scalar_data=None, ax=None, projection=None):
       if projection is None:
           projection = ccrs.PlateCarree()
       if ax is None:
           fig, ax = plt.subplots(subplot_kw={"projection": projection})
       ...
       return ax
   ```

2. For `_plot_clusters`, which currently creates two subplots internally (211, 212), refactor to accept `fig=None, axes=None`:
   ```python
   def _plot_clusters(..., fig=None, axes=None, projection=None):
       if projection is None:
           projection = ccrs.PlateCarree(central_longitude=180)
       if axes is None:
           fig, axes = plt.subplots(2, 1, subplot_kw={"projection": projection})
       ax_top, ax_bottom = axes
       ...
       return fig, axes
   ```

3. In the `Waper` class methods (`plot_clusters`, `plot_association_graph`, etc.), pass through `ax=None` and `projection=None` to the underlying function.

**Definition of Done:**
- [ ] Every `_plot_*` function accepts `ax` and `projection`.
- [ ] No function calls `plt.subplot(...)` with a hardcoded projection unless `ax` is None.
- [ ] User can pass their own axes into any plot method.

---

### Task 7.2 — Add Coastlines and Gridlines

**File to edit:**
- `waper/interface/visualization.py`

**Steps:**

1. In every `_plot_*` function, after creating or receiving the axes, add:
   ```python
   ax.coastlines(linewidth=0.5, color='gray')
   ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
   ```

2. Make this optional via a `geographic_features=True` parameter.

**Definition of Done:**
- [ ] All map plots show coastlines and gridlines by default.
- [ ] `geographic_features=False` suppresses them.

---

### Task 7.3 — Add Hovmöller Diagram for Tracking

**File to create:**
- New function in `waper/interface/visualization.py`
- New method in `Waper` class

**Steps:**

1. Write `_plot_hovmoller(tracking_graph, track_paths)`:
   - X-axis: longitude.
   - Y-axis: time (timestep index).
   - Plot each track as a line showing the weighted longitude at each timestep.
   - Color-code by track index or by intensity.

2. Add `Waper.plot_hovmoller()` method that calls the above.

**Definition of Done:**
- [ ] `plot_hovmoller` produces a longitude-vs-time diagram.
- [ ] Each track is a distinct colored line.

---

### Task 7.4 — Fix Polygon Plotting Projection Mismatch

**Problem:** `_plot_polygons` plots polygon boundaries in stereographic coordinates but scatter points (weighted centers) in PlateCarree. This works by accident because Cartopy transforms, but it's confusing and fragile.

**File to edit:**
- `waper/interface/visualization.py`

**Steps:**

1. Convert polygon exterior coordinates back to lat/lon before plotting:
   ```python
   from ..tracking.rwp_polygon import transform_to_stereographic

   for poly in poly_list:
       stereo_lons, stereo_lats = poly.exterior.coords.xy
       geo_lons, geo_lats = transform_to_stereographic(
           np.array(stereo_lons), np.array(stereo_lats), inverse=True
       )
       ax.plot(geo_lons, geo_lats, transform=ccrs.PlateCarree())
   ```

2. This allows the polygon to be plotted on any projection, not just stereographic.

**Definition of Done:**
- [ ] All plot elements use a consistent geographic CRS.
- [ ] Polygons render correctly on PlateCarree, Orthographic, and Stereographic projections.

---

## Phase 8: Documentation

### Task 8.1 — Fix Template Documentation

**Problem:** `docs/contents/30_usage.rst` references `my_new_project`. `docs/contents/40_modules.rst` references `my_new_project`. `README.rst` has TODO placeholders.

**Files to edit:**
- `docs/contents/30_usage.rst`
- `docs/contents/40_modules.rst`
- `docs/contents/my_new_project.rst`
- `README.rst`

**Steps:**

1. Replace all `my_new_project` references with `waper`.
2. In `30_usage.rst`, write a basic usage example:
   ```rst
   =====
   Usage
   =====

   .. code-block:: python

      import xarray as xr
      from waper import Waper

      ds = xr.open_dataset("era5_v_wind.nc")

      w = Waper(
          data_array=ds,
          scalar_name="v",
          latitude_label="latitude",
          longitude_label="longitude",
          time_label="time",
      )

      w.identify_rwps()
      w.track_rwps()

      ax = w.plot_rwp_graphs(time_index=0)
   ```
3. In `40_modules.rst`, reference `waper` not `my_new_project`.
4. Rename or delete `my_new_project.rst`.
5. In `README.rst`, fill in the TODO placeholders with actual feature descriptions.

**Definition of Done:**
- [ ] `my_new_project` does not appear anywhere in the docs.
- [ ] Usage example is correct and runnable.
- [ ] README lists actual features.

---

### Task 8.2 — Add Docstrings to All Public Functions

**Problem:** Many functions have incomplete or missing docstrings. Type hints are sparse despite `py.typed` marker.

**Files to edit:** All `.py` files in `waper/`.

**Steps:**

1. For every public function (not starting with `_`), ensure a Google-style docstring exists with:
   - One-line summary.
   - `Args:` section with type and description for each parameter.
   - `Returns:` section with type and description.
   - `Raises:` section if applicable.

2. Add type hints to all function signatures. At minimum:
   - `api.py`: `Waper.__init__`, `identify_rwps`, `track_rwps`, all `plot_*` methods.
   - `rwp_graph.py`: `compute_association_graph`, `prune_association_graph_nodes`, `prune_association_graph_edges`, `get_ranked_paths`.
   - `topology.py`: `cluster_extrema` (after Task 3.1).
   - `utils.py`: `haversine_distance`, `is_to_the_east`.

3. Run `mypy waper/ --ignore-missing-imports` and fix any type errors that arise from the new annotations.

**Definition of Done:**
- [ ] Every public function has a docstring.
- [ ] Every public function has type hints on arguments and return.
- [ ] `mypy` reports zero errors (or only errors from third-party stubs).

---

### Task 8.3 — Write an Algorithm Overview in the README

**Steps:**

1. Add a section to `README.rst` (or a separate `docs/algorithm.rst`) that summarises the algorithm in 2-3 paragraphs, referencing Pandey et al. (2020) and Malavika's thesis.

2. Include a diagram (ASCII or image) showing the pipeline:
   ```
   Input (v-wind field)
     → Extrema Detection
     → Connected Region Labeling
     → Clustering (DBSCAN)
     → Association Graph
     → Node Pruning → Edge Pruning
     → Path Extraction
     → Polygon Footprints
     → Rasterisation / Quadtree
     → Temporal Tracking Graph
     → Track Extraction
   ```

3. Link to the full `architecture_and_algorithm.md` for detailed math.

**Definition of Done:**
- [ ] README or docs contains an algorithm overview.
- [ ] Both papers are cited.

---

## Phase 9: Stretch Goals

These are desirable but not critical. They can be tackled after all other phases.

### Task 9.1 — Replace NetworkX Quadtree with Direct Polygon Intersection

See Task 5.5 for full specification.

### Task 9.2 — Parallel Timestep Processing

**Problem:** `identify_rwps` processes timesteps sequentially in a `for` loop. Each timestep is independent during identification.

**Steps:**

1. Use `concurrent.futures.ProcessPoolExecutor` or `joblib.Parallel`:
   ```python
   from joblib import Parallel, delayed

   def identify_rwps(self):
       self._time_step_data = Parallel(n_jobs=-1)(
           delayed(_identify_rwps)(
               self.data_array[self._config.scalar_name][i], self._config
           )
           for i in range(self._num_time_steps)
       )
   ```

2. Ensure `WaperConfig` and `DataArray` are picklable (they should be).

3. VTK objects are NOT picklable. This task depends on Phase 5 (removing VTK).

**Definition of Done:**
- [ ] Identification runs in parallel across timesteps.
- [ ] Results are identical to sequential execution.

---

### Task 9.3 — xarray Integration for Output

**Problem:** Results are stored in ad-hoc dicts and dataclasses. Users must manually extract coordinates.

**Steps:**

1. Add a method `Waper.to_dataset()` that returns an `xarray.Dataset` with:
   - A variable `rwp_id(time, rwp_index)` — the ID of each RWP at each timestep.
   - A variable `rwp_longitude(time, rwp_index)` — weighted longitude.
   - A variable `rwp_latitude(time, rwp_index)` — weighted latitude.
   - A variable `rwp_peak_value(time, rwp_index)` — peak scalar value.

2. Add a method `Waper.tracks_to_dataframe()` that returns a `pandas.DataFrame` with columns: `track_id`, `time`, `longitude`, `latitude`, `peak_value`.

**Definition of Done:**
- [ ] `to_dataset()` returns a well-formed xarray Dataset.
- [ ] `tracks_to_dataframe()` returns a well-formed DataFrame.

---

### Task 9.4 — Add `WaperConfig.from_yaml()` and `WaperConfig.to_yaml()`

**Steps:**

1. Use `dataclasses.asdict` + `yaml.dump` for serialisation.
2. Use `yaml.safe_load` + `WaperConfig(**d)` for deserialisation.
3. This allows reproducible runs with config files.

**Definition of Done:**
- [ ] Round-trip: `WaperConfig.from_yaml(config.to_yaml()) == config`.

---

## Appendix A: File Inventory

| File | Purpose | Key Issues |
|------|---------|------------|
| `waper/__init__.py` | Package root | Exposes internal submodules (Task 0.3) |
| `waper/interface/__init__.py` | Interface subpackage | Same issue |
| `waper/interface/api.py` | Main `Waper` class, `WaperConfig`, orchestration | Dead `logging` function; absolute import; Feature ID bug |
| `waper/interface/visualization.py` | All plotting functions | Hardcoded projections; no coastlines; projection mismatch |
| `waper/identification/__init__.py` | Empty | — |
| `waper/identification/max_min.py` | Extrema detection | O(N²) loops; `check` array bug; boundary bug; dead code |
| `waper/identification/topology.py` | Clustering | AP forcing; VTK Dijkstra; `FindCellsAlongLine` bug; code duplication |
| `waper/identification/rwp_graph.py` | Association graph | Node ID collision; Euclidean distance; lon wraparound; path ranking |
| `waper/identification/utils.py` | Utilities (mesh, distance) | `is_to_the_east` bug; radius inconsistency; raw VTK calls |
| `waper/tracking/__init__.py` | Empty | — |
| `waper/tracking/quadtree.py` | Quadtree spatial index | NetworkX overhead (stretch goal) |
| `waper/tracking/rwp_polygon.py` | Polygon footprints | Feature ID bug; convex hull; NH-only; bare except |
| `waper/tracking/tracking_graph.py` | Temporal tracking graph | `all_simple_paths`; redundant merge; path ranking |
| `tests/smoke_test.py` | Smoke test | Tests `my_new_project` |
| `pyproject.toml` | Build config | No dependencies; wrong metadata |
| `.github/workflows/test.yaml` | CI | References `cookiecutter_python`; wrong Python versions |
| `docs/` | Documentation | All template/placeholder |

---

## Appendix B: Magic Numbers Registry

Every hardcoded constant that should be either configurable or documented with its unit.

| Constant | File | Current Value | Unit | Proposed Location |
|----------|------|---------------|------|-------------------|
| `CLUSTER_MAX_DISTANCE` | `topology.py` | 150 | VTK-sphere units (~15000 km) | `WaperConfig.cluster_max_distance_km` |
| `WAPER_MAX_SCALAR_VALUE` | `rwp_graph.py` | 100 | m/s | `WaperConfig.max_scalar_value` |
| `WAPER_MAX_NODE_DISTANCE` | `rwp_graph.py` | 1000 | VTK-sphere units | `WaperConfig.max_node_distance_km` |
| `WAPER_MIN_LON_DELTA` | `rwp_graph.py` | 6 | degrees | `WaperConfig.min_longitude_separation` |
| `WAPER_SUBSAMPLE` | `rwp_polygon.py` | 5 | points | `WaperConfig.polygon_subsample` |
| `WAPER_IMAGE_SIZE` | `rwp_polygon.py` | 512 | pixels | `WaperConfig.raster_size` |
| `WAPER_CLUSTER_WIDTH` | `rwp_polygon.py` | 60 | degrees | `WaperConfig.cluster_width_degrees` |
| `WAPER_X_BOUNDS` | `rwp_polygon.py` | hardcoded | meters (stereo) | Compute dynamically from hemisphere |
| `WAPER_Y_BOUNDS` | `rwp_polygon.py` | hardcoded | meters (stereo) | Compute dynamically from hemisphere |
| `RADIUS_SPHERE` | `utils.py` | 63.71 | arbitrary (100 km?) | Document clearly; keep as internal |
| `RADIUS_EARTH` | `utils.py` | 6.371e6 | meters | Change to `RADIUS_EARTH_KM = 6371.0` |
| `median_dist / 5.0` | `topology.py` | varies | AP preference | Remove when switching to DBSCAN |
| `path_max / 3.0` | `rwp_polygon.py` | varies | clip fraction | `WaperConfig.polygon_clip_fraction` |
| `0.001` (ray tolerance) | `topology.py` | 0.001 | VTK units | Remove when switching to SciPy |

---

## Appendix C: Dependency Map

Shows which tasks depend on which. **Independent tasks within a phase can be parallelised.**

```
Phase 0 (all independent of each other)
  ├── 0.1 (pyproject.toml)
  ├── 0.2 (dead code removal)
  ├── 0.3 (__init__.py + smoke test)
  └── 0.4 (logging)

Phase 1 (depends on Phase 0)
  ├── 1.1 (fixtures) ← all other Phase 1 tasks depend on this
  ├── 1.2 (test extrema) ← depends on 1.1
  ├── 1.3 (test clustering) ← depends on 1.1
  ├── 1.4 (test association graph) ← depends on 1.1
  ├── 1.5 (test tracking) ← depends on 1.1
  └── 1.6 (integration test) ← depends on 1.1

Phase 2 (depends on Phase 1 for validation)
  ├── 2.1 (extrema detection) ← independent
  ├── 2.2 (node ID collision) ← independent
  ├── 2.3 (is_to_the_east) ← independent
  ├── 2.4 (Euclidean distance) ← independent
  ├── 2.5 (feature ID) ← independent
  └── 2.6 (bare except) ← independent

Phase 3 (depends on Phase 2 for correctness baseline)
  ├── 3.1 (DBSCAN) ← independent
  ├── 3.2 (similarity penalty) ← depends on 3.1 (uses cluster_extrema)
  ├── 3.3 (weighted centroid) ← depends on 3.1 (uses cluster_extrema)
  ├── 3.4 (path ranking) ← independent
  ├── 3.5 (lon wraparound) ← independent
  └── 3.6 (radius units) ← independent

Phase 4 (depends on Phase 3)
  ├── 4.1 (concave hull) ← independent
  ├── 4.2 (southern hemisphere) ← independent
  ├── 4.3 (DAG longest path) ← independent
  └── 4.4 (decouple merge) ← independent

Phase 5 (depends on Phase 3 for clean topology.py)
  ├── 5.1 (VTK contour/gradient) ← independent
  ├── 5.2 (VTK Dijkstra → SciPy) ← depends on 3.1, 3.2
  ├── 5.3 (VTK in max_min) ← depends on 2.1
  ├── 5.4 (VTK connectivity) ← independent
  └── 5.5 (quadtree → R-tree) ← stretch, depends on 4.1

Phase 6 (depends on Phase 5)
  └── 6.3 (benchmark) ← depends on all optimisations

Phase 7 (can start after Phase 2; no algorithmic dependency)
  ├── 7.1 (axes/projection) ← independent
  ├── 7.2 (coastlines) ← independent
  ├── 7.3 (Hovmöller) ← independent
  └── 7.4 (polygon projection fix) ← depends on 4.1

Phase 8 (can start any time)
  ├── 8.1 (fix template docs) ← independent
  ├── 8.2 (docstrings) ← depends on final API shape (Phase 3+)
  └── 8.3 (algorithm overview) ← independent

Phase 9 (after everything else)
  ├── 9.1 (quadtree removal) = 5.5
  ├── 9.2 (parallel timesteps) ← depends on Phase 5 (no VTK)
  ├── 9.3 (xarray output) ← depends on final API shape
  └── 9.4 (YAML config) ← independent
```

---

## Appendix D: Edge Cases to Test

A checklist of boundary conditions that must have explicit test coverage.

| Edge Case | Where It Matters | Expected Behavior |
|-----------|-----------------|-------------------|
| Flat field (all zeros) | Extrema detection | Zero extrema found; no crash |
| Single grid cell above threshold | Clustering | One cluster with one point |
| Wave packet straddling 0°/360° | Association graph, polygon, tracking | Correct associations; polygon wraps; tracking connects |
| Wave packet at the North Pole | VTK mesh, distance computation | Distances are correct; no NaN |
| Wave packet at the South Pole | Stereographic projection | Works if hemisphere="south" |
| Two RWPs with identical peak scalar | Feature ID assignment | Distinct IDs; both in raster |
| Very weak field (all below threshold) | Identification | No RWPs; informative log message |
| Dense field (100+ extrema per region) | Clustering, AP/DBSCAN convergence | Completes in reasonable time; clusters are physically plausible |
| Single timestep (no tracking) | Tracking | Graceful no-op or informative error |
| Timestep with zero RWPs followed by timestep with RWPs | Tracking graph | No spurious edges; no crash |
| Very high resolution grid (0.25° global) | Performance | Completes identification in < 60s |
| Non-global domain (regional subset) | Periodicity assumptions | No wraparound artifacts |
| NaN values in input | Extrema detection | NaNs are ignored or raise clear error |
| Non-uniform latitude spacing | VTK mesh construction | Mesh handles it correctly |
| Longitude starting from -180 instead of 0 | `np.linspace(0, 360, ...)` assumption | Longitudes are correctly mapped |

---

## Appendix E: Complete `WaperConfig` After All Phases

```python
@dataclass(eq=False, frozen=True)
class WaperConfig:

    # --- User-required ---
    scalar_name: str
    latitude_label: str
    longitude_label: str
    time_label: str

    # --- Identification thresholds ---
    clip_value: float = 2.0
    extrema_threshold: float = 10.0
    node_pruning_threshold: float = 20.0
    edge_pruning_threshold: float = 3e-5
    max_edge_weight: float = 1.0
    min_longitude_separation: float = 6.0    # degrees
    max_scalar_value: float = 100.0          # m/s

    # --- Latitude bounds ---
    max_latitude: float | None = None
    min_latitude: float | None = None

    # --- Clustering ---
    cluster_eps_km: float = 500.0            # DBSCAN eps in km
    cluster_min_samples: int = 1             # DBSCAN min_samples

    # --- Polygon footprints ---
    polygon_clip_fraction: float = 3.0       # clip at peak / this
    hull_method: str = "per_node"            # "convex", "concave", "per_node"
    polygon_subsample: int = 5

    # --- Tracking ---
    track_pruning_threshold: float = 0.3     # km (distance threshold)
    hemisphere: str = "north"                # "north" or "south"
    raster_size: int = 512                   # pixels (if using quadtree)

    # --- Internal VTK labels ---
    vtk_latitude_label: str = "Latitude"
    vtk_longitude_label: str = "Longitude"
    vtk_region_label: str = "RegionId"

    # --- Debug ---
    debug: bool = False
```
