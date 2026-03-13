import networkx as nx
import numpy as np
import pytest

from waper.interface.api import WaperConfig, _identify_rwps
from waper.tracking import tracking_graph
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


@pytest.mark.xfail(reason="Phase 2/4 tracking bugs")
def test_identical_timesteps_full_overlap(simple_wave_field, default_config):
    ts_data = _identify_rwps(simple_wave_field, default_config)
    ts_list = [ts_data, ts_data]

    track_g = tracking_graph.build_tracking_graph(ts_list, 2)

    # Check edges
    # Weight should be 1.0 because the quadtrees are exactly the same
    for u, v, data in track_g.edges(data=True):
        assert pytest.approx(data["weight"], 0.01) == 1.0


@pytest.mark.xfail(reason="Phase 2/4 tracking bugs")
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


@pytest.mark.xfail(reason="Phase 2/4 tracking bugs")
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


@pytest.mark.xfail(reason="Phase 2/4 tracking bugs")
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
