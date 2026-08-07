"""Smoke coverage for the `zip(..., strict=True)` sites the rest of the suite misses.

Ruff's B905 required an explicit `strict=` on every `zip()`. Where the operands are
equal-length by construction we chose `strict=True`, which turns a silent truncation
into a `ValueError`. Most of those sites are already exercised by the suite; the ones
below were not, so this module runs them once to confirm the lengths really do match:

* `waper/interface/visualization.py` — the extrema-annotation zips in `_plot_clusters`
  and the weighted-centroid zip in `_plot_polygons` (the module is otherwise untested).
* `waper/tracking/rwp_polygon.py` — the `concave` and `convex` hull branches of
  `get_rwp_polygon`, which the default `per_node` path never reaches.
"""
import matplotlib
import pytest
import xarray as xr

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_clusters_zips_equal_length_point_arrays(tracked_waper):
    """`_plot_clusters` zips three point-data arrays of one PolyData."""
    assert tracked_waper.plot_clusters(0) is not None


def test_plot_rwp_polygons_zips_parallel_centroid_lists(tracked_waper):
    """`_plot_polygons` zips the weighted lon/lat lists built in lockstep."""
    assert tracked_waper.plot_rwp_polygons(0, plot_samples=True) is not None


@pytest.mark.parametrize("hull_method", ["per_node", "convex", "concave"])
def test_all_hull_methods_zip_transform_outputs(two_timestep_field, hull_method):
    """Every hull branch zips the paired outputs of one stereographic transform."""
    from waper.interface.api import Waper

    w = Waper(data_array=xr.Dataset({"v": two_timestep_field}), scalar_name="v",
              latitude_label="latitude", longitude_label="longitude", time_label="time",
              clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80.1,
              node_pruning_threshold=15, edge_pruning_threshold=3e-5)
    # WaperConfig is a frozen dataclass and `hull_method` has no constructor argument.
    object.__setattr__(w._config, "hull_method", hull_method)
    w.identify_rwps()
    assert w._time_step_data[0].rwp_info
