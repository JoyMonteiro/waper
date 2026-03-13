# Version History

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
