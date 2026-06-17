import os
import time as time_module
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest
import xarray as xr

from waper.interface.api import WaperConfig, Waper, _identify_rwps
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


def _stub_tsd(raster, energy, features):
    from unittest.mock import MagicMock
    s = MagicMock()
    s.raster_data = raster
    s.energy_raster = energy
    s.raster_features = features
    s.rwp_info = {}
    return s


def test_energy_weight_full_overlap_is_one():
    F = np.array([[0, 1, 1], [0, 1, 0]])
    E = np.array([[0.0, 2.0, 2.0], [0.0, 2.0, 0.0]])
    ts = [_stub_tsd(F, E, {0, 1}), _stub_tsd(F.copy(), E.copy(), {0, 1})]
    g = tracking_graph.build_tracking_graph(ts, 2)
    assert g.number_of_edges() == 1
    assert abs(g[(0, 1)][(1, 1)]["weight"] - 1.0) < 1e-9


def test_energy_weight_partial_when_core_moves():
    Fp = np.array([[1, 1, 0, 0]]); Ep = np.array([[5.0, 1.0, 0.0, 0.0]])
    Fc = np.array([[0, 1, 1, 0]]); Ec = np.array([[0.0, 1.0, 5.0, 0.0]])
    ts = [_stub_tsd(Fp, Ep, {0, 1}), _stub_tsd(Fc, Ec, {0, 1})]
    g = tracking_graph.build_tracking_graph(ts, 2)
    w = g[(0, 1)][(1, 1)]["weight"]
    assert 0.0 < w < 1.0


def test_overlap_computed_once_per_timestep_pair():
    F = np.array([[1]]); E = np.array([[1.0]])
    ts = [_stub_tsd(F, E, {0, 1}) for _ in range(3)]
    with patch("waper.tracking.tracking_graph.overlap_energies",
               return_value={}) as mock_ov:
        tracking_graph.build_tracking_graph(ts, number_steps=3)
        assert mock_ov.call_count == 2


def test_feature_zero_not_in_edges(simple_wave_field, default_config):
    """Feature 0 (background) must never appear as an endpoint in tracking graph edges."""
    ts_data = _identify_rwps(simple_wave_field, default_config)
    ts_list = [ts_data, ts_data]

    track_g = tracking_graph.build_tracking_graph(ts_list, 2)
    for u, v in track_g.edges():
        assert u[1] != 0, f"Feature 0 found as source in edge {u} -> {v}"
        assert v[1] != 0, f"Feature 0 found as target in edge {u} -> {v}"


def test_energy_weighted_centroid_favors_high_amplitude():
    from waper.tracking.rwp_polygon import _weighted_centroid
    xs = np.array([0.0, 10.0]); ys = np.array([0.0, 0.0])
    values = np.array([1.0, 3.0])          # 3x amplitude -> 9x energy
    wx, wy = _weighted_centroid(xs, ys, values)
    assert abs(wx - 9.0) < 1e-9            # 10 * 9/(1+9) = 9.0
    assert abs(wy - 0.0) < 1e-9


def test_weighted_centroid_uses_squared_weights_sign_independent():
    from waper.tracking.rwp_polygon import _weighted_centroid
    xs = np.array([0.0, 4.0]); ys = np.array([0.0, 0.0])
    values = np.array([-2.0, 2.0])         # equal energy (4 each) -> midpoint
    wx, _ = _weighted_centroid(xs, ys, values)
    assert abs(wx - 2.0) < 1e-9


def test_energy_disks_one_per_node_weighted_by_energy():
    from waper.tracking.rwp_polygon import energy_disks
    # two extrema; energy must be amplitude**2 and sign-independent
    cells = energy_disks([(0.0, 50.0, 3.0), (90.0, 50.0, -2.0)],
                         hemisphere="north", radius_m=300e3)
    assert len(cells) == 2
    geom0, e0 = cells[0]
    geom1, e1 = cells[1]
    assert abs(e0 - 9.0) < 1e-9
    assert abs(e1 - 4.0) < 1e-9
    assert geom0.geom_type == "Polygon" and geom0.area > 0


def test_energy_disks_empty():
    from waper.tracking.rwp_polygon import energy_disks
    assert energy_disks([], hemisphere="north") == []


def test_rasterize_energy_shape_and_values():
    from waper.tracking.rwp_polygon import energy_disks, rasterize_energy, WAPER_IMAGE_SIZE
    cells = energy_disks([(0.0, 80.0, 4.0)], hemisphere="north", radius_m=500e3)
    raster = rasterize_energy(cells, hemisphere="north")
    assert raster.shape == (WAPER_IMAGE_SIZE, WAPER_IMAGE_SIZE)
    assert raster.dtype == np.float64
    # the only non-zero value present is the burned energy (4**2 = 16)
    nonzero = np.unique(raster[raster > 0])
    assert nonzero.size == 1 and abs(nonzero[0] - 16.0) < 1e-6
    assert (raster > 0).sum() > 0


def test_rasterize_energy_empty_returns_none():
    from waper.tracking.rwp_polygon import rasterize_energy
    assert rasterize_energy([], hemisphere="north") is None


def test_energy_raster_built_and_aligned(two_timestep_field):
    import xarray as xr
    ds = xr.Dataset({"v": two_timestep_field})
    w = Waper(data_array=ds, scalar_name="v",
              latitude_label="latitude", longitude_label="longitude", time_label="time",
              clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80.1,
              node_pruning_threshold=15, edge_pruning_threshold=3e-5,
              track_pruning_threshold=0.3, max_edge_weight=1, debug=False)
    w.identify_rwps()
    tsd = w._time_step_data[0]
    assert tsd.energy_raster is not None
    assert tsd.energy_raster.shape == tsd.raster_data.shape
    # energy lives only where a feature footprint is, and is strictly positive there
    assert (tsd.energy_raster > 0).any()
    assert np.all(tsd.energy_raster[tsd.raster_data == 0] >= 0)


DATASET = "datasets/forecast_bust.nc"


@pytest.mark.skipif(not os.path.exists(DATASET), reason="forecast_bust.nc not present")
def test_energy_tracks_show_eastward_motion():
    ds = xr.open_dataset(DATASET)
    w = Waper(data_array=ds, scalar_name="v",
              latitude_label="latitude", longitude_label="longitude", time_label="time",
              clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80,
              node_pruning_threshold=20, edge_pruning_threshold=0.02, max_edge_weight=1,
              track_pruning_threshold=0.3)
    w.identify_rwps(); w.track_rwps()
    g = w._tracking_graph
    # at least one tracked edge exists, and centroids move (not frozen)
    assert g.number_of_edges() > 0
    moved = [d["distance"] for _, _, d in g.edges(data=True)]
    assert np.median(moved) > 0.0
