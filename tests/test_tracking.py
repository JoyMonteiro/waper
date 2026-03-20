import time as time_module
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from waper.interface.api import WaperConfig, _identify_rwps
from waper.tracking import tracking_graph
from waper.tracking import quadtree as qt_module
from waper.tracking.quadtree import compute_size_features, create_quadtree
from waper.tracking.rwp_polygon import WAPER_IMAGE_SIZE


@pytest.fixture
def default_config():
    return WaperConfig(
        debug=False,
        scalar_name="v",
        latitude_label="latitude",
        longitude_label="longitude",
        time_label="time",
        clip_value=2,
        extrema_threshold=10,
        max_latitude=80.1,
        min_latitude=20,
        node_pruning_threshold=15,
        edge_pruning_threshold=3e-5,
        track_pruning_threshold=0.3,
        max_edge_weight=1,
    )


def test_identical_timesteps_full_overlap(simple_wave_field, default_config):
    ts_data = _identify_rwps(simple_wave_field, default_config)
    ts_list = [ts_data, ts_data]

    track_g = tracking_graph.build_tracking_graph(ts_list, 2)

    # Check edges
    # Weight should be 1.0 because the quadtrees are exactly the same
    for u, v, data in track_g.edges(data=True):
        assert pytest.approx(data["weight"], 0.01) == 1.0


def test_shifted_field_partial_overlap(two_timestep_field, default_config):
    ts_data_0 = _identify_rwps(two_timestep_field.isel(time=0), default_config)
    ts_data_1 = _identify_rwps(two_timestep_field.isel(time=1), default_config)
    ts_list = [ts_data_0, ts_data_1]

    track_g = tracking_graph.build_tracking_graph(ts_list, 2)

    # Since it shifted, the overlap should be between 0 and 1.
    edges_found = False
    for u, v, data in track_g.edges(data=True):
        if 0 < data["weight"] < 1.0:
            edges_found = True

    assert edges_found


def test_no_overlap_no_edge(single_maximum_field, default_config):
    ts_data_0 = _identify_rwps(single_maximum_field, default_config)

    # Move the bump far away
    da2 = single_maximum_field.copy(deep=True)
    lons = da2["longitude"].values
    lats = da2["latitude"].values
    lon2d, lat2d = np.meshgrid(lons, lats)
    v2 = 30 * np.exp(-((lon2d - 90) ** 2 + (lat2d - 50) ** 2) / (2 * 10**2))
    da2.values = v2

    ts_data_1 = _identify_rwps(da2, default_config)

    ts_list = [ts_data_0, ts_data_1]
    track_g = tracking_graph.build_tracking_graph(ts_list, 2)

    assert len(track_g.edges) == 0


def test_tracking_path_extraction(simple_wave_field, default_config):
    ts_data = _identify_rwps(simple_wave_field, default_config)
    ts_list = [ts_data, ts_data, ts_data]

    track_g = tracking_graph.build_tracking_graph(ts_list, 3)
    paths = tracking_graph.get_track_paths(track_g)

    found_long_path = False
    for p in paths:
        if len(p) == 3:
            found_long_path = True
            assert p[0][0] == 0
            assert p[1][0] == 1
            assert p[2][0] == 2

    assert found_long_path


def test_dag_dp_completes_fast():
    """DAG DP must process 20 timesteps x 5 features in under 1 second."""
    g = nx.DiGraph()
    for t in range(20):
        for f in range(1, 6):
            g.add_node((t, f), coords=(float(f * 10), 50.0))
            if t > 0:
                g.add_edge((t - 1, f), (t, f), weight=0.8, distance=500.0)
                if f < 5:
                    g.add_edge((t - 1, f), (t, f + 1), weight=0.3, distance=600.0)

    start = time_module.monotonic()
    paths = tracking_graph.get_track_paths(g)
    elapsed = time_module.monotonic() - start

    assert elapsed < 1.0, f"get_track_paths took {elapsed:.2f}s — too slow"
    assert len(paths) > 0


def test_quadtree_pixel_counts():
    raster = np.zeros((WAPER_IMAGE_SIZE, WAPER_IMAGE_SIZE), dtype=int)
    # Feature 1: 10x10 block
    raster[10:20, 10:20] = 1
    # Feature 2: 20x20 block
    raster[50:70, 50:70] = 2

    qt = create_quadtree(raster)
    sizes = compute_size_features(qt)

    assert sizes[(1,)] == 100
    assert sizes[(2,)] == 400


def test_merge_called_once_per_timestep_pair(simple_wave_field, default_config):
    """merge() must be called exactly (number_steps - 1) times, not once per feature."""
    ts_data = _identify_rwps(simple_wave_field, default_config)
    ts_list = [ts_data, ts_data, ts_data]  # 3 timesteps → 2 pairs

    with patch.object(tracking_graph, "merge", wraps=qt_module.merge) as mock_merge:
        tracking_graph.build_tracking_graph(ts_list, number_steps=3)
        assert mock_merge.call_count == 2, (
            f"Expected merge() called 2 times (once per timestep pair), "
            f"got {mock_merge.call_count}"
        )


def test_feature_zero_not_in_edges(simple_wave_field, default_config):
    """Feature 0 (background) must never appear as an endpoint in tracking graph edges."""
    ts_data = _identify_rwps(simple_wave_field, default_config)
    ts_list = [ts_data, ts_data]

    track_g = tracking_graph.build_tracking_graph(ts_list, 2)
    for u, v in track_g.edges():
        assert u[1] != 0, f"Feature 0 found as source in edge {u} -> {v}"
        assert v[1] != 0, f"Feature 0 found as target in edge {u} -> {v}"
