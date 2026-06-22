import networkx as nx
from waper.identification.rwp_graph import get_ranked_paths
from waper.identification.rwp_graph import (
    _path_lon_span, _arcs_overlap, _path_lat_range,
    _lat_ranges_within, _paths_interleave_in_band,
)


def _g():
    """A→B edge (a valid 2-node RWP) plus an isolated node C."""
    g = nx.Graph()
    g.add_node(("max", 0), coords=(10.0, 50.0), node_type="max", scalar=30.0)
    g.add_node(("min", 0), coords=(30.0, 50.0), node_type="min", scalar=-30.0)
    g.add_node(("max", 1), coords=(200.0, 50.0), node_type="max", scalar=25.0)  # isolated
    g.add_edge(("max", 0), ("min", 0), weight=1.0)
    return g


def test_no_length_one_paths():
    paths = get_ranked_paths(_g(), max_weight=10.0)
    assert all(len(p) >= 2 for p in paths)
    # the isolated node must not appear as its own RWP
    assert not any(p == [("max", 1)] for p in paths)


def _node(g, name, lon, lat):
    g.add_node(name, coords=(lon, lat), node_type=name[0], scalar=10.0)


def test_lon_span_and_lat_range():
    g = nx.Graph()
    _node(g, ("max", 0), 10.0, 40.0)
    _node(g, ("min", 0), 30.0, 55.0)
    _node(g, ("max", 1), 60.0, 45.0)
    path = [("max", 0), ("min", 0), ("max", 1)]
    start, length = _path_lon_span(g, path)
    assert start == 10.0
    assert abs(length - 50.0) < 1e-6          # 20 + 30 degrees eastward
    assert _path_lat_range(g, path) == (40.0, 55.0)


def test_arcs_overlap():
    assert _arcs_overlap(10.0, 40.0, 30.0, 40.0) is True     # [10,50] vs [30,70]
    assert _arcs_overlap(10.0, 20.0, 100.0, 20.0) is False   # [10,30] vs [100,120]
    assert _arcs_overlap(350.0, 30.0, 10.0, 10.0) is True    # wrap: [350,20] vs [10,20]


def test_lat_ranges_within():
    assert _lat_ranges_within((40, 55), (45, 60), 15) is True    # overlapping
    assert _lat_ranges_within((30, 35), (45, 50), 15) is True    # gap 10 <= 15
    assert _lat_ranges_within((20, 25), (45, 50), 15) is False   # gap 20 > 15


def test_paths_interleave_in_band():
    g = nx.Graph()
    _node(g, ("max", 0), 10.0, 50.0); _node(g, ("min", 0), 40.0, 52.0)   # band A 50-52
    _node(g, ("max", 1), 20.0, 51.0); _node(g, ("min", 1), 50.0, 53.0)   # overlaps lon, same band
    _node(g, ("max", 2), 20.0, 25.0); _node(g, ("min", 2), 50.0, 27.0)   # overlaps lon, far band
    a = [("max", 0), ("min", 0)]
    b = [("max", 1), ("min", 1)]
    c = [("max", 2), ("min", 2)]
    assert _paths_interleave_in_band(g, a, b, 15.0) is True    # same band, overlapping lon
    assert _paths_interleave_in_band(g, a, c, 15.0) is False   # far band -> allowed


def test_pass1_rejects_in_band_interleaver():
    g = nx.Graph()
    # strong train, lat ~50, lon 10..70
    _node(g, ("max", 0), 10.0, 50.0); _node(g, ("min", 0), 40.0, 51.0); _node(g, ("max", 2), 70.0, 50.0)
    g.add_edge(("max", 0), ("min", 0), weight=5.0)
    g.add_edge(("min", 0), ("max", 2), weight=5.0)
    # weak interleaver, lat ~52 (same band), lon 20..50
    _node(g, ("min", 1), 20.0, 52.0); _node(g, ("max", 1), 50.0, 52.0)
    g.add_edge(("min", 1), ("max", 1), weight=0.5)
    paths = get_ranked_paths(g, max_weight=10.0, lat_gate=15.0)
    node_sets = [set(p) for p in paths]
    assert {("max", 0), ("min", 0), ("max", 2)} in node_sets        # strong train kept
    assert {("min", 1), ("max", 1)} not in node_sets                # weak interleaver rejected


def test_pass1_keeps_different_waveguide():
    g = nx.Graph()
    # midlat train lat ~50, lon 10..70
    _node(g, ("max", 0), 10.0, 50.0); _node(g, ("min", 0), 40.0, 50.0); _node(g, ("max", 2), 70.0, 50.0)
    g.add_edge(("max", 0), ("min", 0), weight=5.0)
    g.add_edge(("min", 0), ("max", 2), weight=5.0)
    # subtropical train lat ~25 (>15 away), overlapping lon 20..60
    _node(g, ("min", 1), 20.0, 25.0); _node(g, ("max", 1), 60.0, 25.0)
    g.add_edge(("min", 1), ("max", 1), weight=4.0)
    paths = get_ranked_paths(g, max_weight=10.0, lat_gate=15.0)
    node_sets = [set(p) for p in paths]
    assert {("max", 0), ("min", 0), ("max", 2)} in node_sets        # midlat kept
    assert {("min", 1), ("max", 1)} in node_sets                    # subtropical also kept


# ---------------------------------------------------------------------------
# Task 4: reassign_orphans tests
# ---------------------------------------------------------------------------
from waper.identification.rwp_graph import reassign_orphans


def test_orphan_extends_chain_end():
    # orphan max east of the path's eastmost min -> clean extension (no branch)
    g = nx.Graph()
    _node(g, ("max", 0), 10.0, 50.0); _node(g, ("min", 0), 40.0, 50.0)
    _node(g, ("max", 1), 70.0, 50.0)  # orphan, east of min0
    g.add_edge(("max", 0), ("min", 0), weight=5.0)
    g.add_edge(("min", 0), ("max", 1), weight=4.0)   # discarded edge available
    out = reassign_orphans(g, [[("max", 0), ("min", 0)]], lat_gate=15.0)
    assert out == [[("max", 0), ("min", 0), ("max", 1)]]


def test_orphan_weak_branch_dropped():
    # interior junction: orphan competes with a strong existing east arm and loses
    g = nx.Graph()
    _node(g, ("max", 0), 10.0, 50.0); _node(g, ("min", 0), 40.0, 50.0); _node(g, ("max", 2), 80.0, 50.0)
    g.add_edge(("max", 0), ("min", 0), weight=5.0)
    g.add_edge(("min", 0), ("max", 2), weight=5.0)          # strong east arm
    _node(g, ("max", 1), 55.0, 50.0)                         # orphan east of min0, weak edge
    g.add_edge(("min", 0), ("max", 1), weight=0.5)
    out = reassign_orphans(g, [[("max", 0), ("min", 0), ("max", 2)]], lat_gate=15.0)
    # orphan dropped; strong train intact
    assert out == [[("max", 0), ("min", 0), ("max", 2)]]


def test_orphan_outside_gate_dropped():
    # orphan within longitude but >15 deg latitude away, with an edge -> not absorbed, dropped
    g = nx.Graph()
    _node(g, ("max", 0), 10.0, 50.0); _node(g, ("min", 0), 40.0, 50.0)
    g.add_edge(("max", 0), ("min", 0), weight=5.0)
    _node(g, ("max", 1), 60.0, 25.0)                         # far band
    g.add_edge(("min", 0), ("max", 1), weight=9.0)           # strong edge, but gated out
    out = reassign_orphans(g, [[("max", 0), ("min", 0)]], lat_gate=15.0)
    assert out == [[("max", 0), ("min", 0)]]


def test_orphan_west_side_absorb():
    # West-side absorb (direction == -1): path starts with a min at its western end.
    # An orphan max strictly west of that min has an edge to it; the west arm is empty
    # so the orphan is prepended without competition.
    #
    # Path:  (min,0)[lon=30] -> (max,0)[lon=60]
    # Orphan: (max,1)[lon=10]  -- west of (min,0)
    # Edge:  (max,1) -- (min,0), weight=3.0
    g = nx.Graph()
    _node(g, ("min", 0), 30.0, 50.0)
    _node(g, ("max", 0), 60.0, 50.0)
    _node(g, ("max", 1), 10.0, 50.0)   # orphan, strictly west of (min,0)
    g.add_edge(("min", 0), ("max", 0), weight=5.0)
    g.add_edge(("max", 1), ("min", 0), weight=3.0)   # edge from orphan to path's west end

    initial_path = [("min", 0), ("max", 0)]
    out = reassign_orphans(g, [initial_path], lat_gate=15.0)

    # Orphan should be prepended: result is [orphan, *original_path]
    assert out == [[("max", 1), ("min", 0), ("max", 0)]]


def test_orphan_cascade_multi_iteration():
    # Cascade / multi-iteration: absorbing a strong orphan D into path A->B->C
    # strips the weak east arm (C), re-orphaning C. On a subsequent iteration C
    # loses its competition against D and is dropped permanently.
    #
    # Path:  (max,0)[lon=10] -w=5- (min,0)[lon=40] -w=2- (max,2)[lon=70]
    # Orphan D=(max,1)[lon=55], edge (min,0)--(max,1) weight=8
    #
    # Iteration 1: D competes with east arm B->C (weight 2). D wins (8>2).
    #   Path becomes [(max,0),(min,0),(max,1)]; (max,2) re-orphaned.
    # Iteration 2: (max,2) tries to re-attach to (min,0) (east side, weight 2).
    #   Existing east arm is now (min,0)--(max,1) weight 8. 2 < 8 -> (max,2) dropped.
    # Final: [[(max,0),(min,0),(max,1)]]
    g = nx.Graph()
    _node(g, ("max", 0), 10.0, 50.0)
    _node(g, ("min", 0), 40.0, 50.0)
    _node(g, ("max", 2), 70.0, 50.0)   # originally in path, becomes orphan after cascade
    _node(g, ("max", 1), 55.0, 50.0)   # strong orphan that triggers cascade

    g.add_edge(("max", 0), ("min", 0), weight=5.0)
    g.add_edge(("min", 0), ("max", 2), weight=2.0)   # weak east arm
    g.add_edge(("min", 0), ("max", 1), weight=8.0)   # strong orphan edge

    initial_path = [("max", 0), ("min", 0), ("max", 2)]
    out = reassign_orphans(g, [initial_path], lat_gate=15.0)

    assert out == [[("max", 0), ("min", 0), ("max", 1)]]


# ---------------------------------------------------------------------------
# Task 5: acceptance test on real data (timestep 95)
# ---------------------------------------------------------------------------
import pytest


def _load_t95():
    import warnings; warnings.filterwarnings("ignore")
    import xarray as xr
    raw = xr.open_dataset("datasets/forecast_bust_hourly.nc")
    da = (raw["v"].rename({"valid_time": "time"})
          .squeeze("pressure_level", drop=True)
          .coarsen(latitude=4, longitude=4, boundary="trim").mean()
          .assign_coords(longitude=lambda d: d.longitude % 360)
          .sortby("longitude"))
    return da.isel(time=[95])


@pytest.mark.slow
def test_acceptance_t95():
    from waper import Waper
    from waper.identification.rwp_graph import (
        _paths_interleave_in_band,
    )
    da = _load_t95()
    w = Waper(data_array=da.to_dataset(name="v"), scalar_name="v",
              latitude_label="latitude", longitude_label="longitude",
              time_label="time", clip_value=2, extrema_threshold=10,
              min_latitude=20, max_latitude=80,
              node_pruning_threshold=20, edge_pruning_threshold=0.02,
              lat_gate=15.0)
    w.identify_rwps()
    tsd = w._time_step_data[0]
    paths = tsd.identified_rwp_paths
    g = tsd.pruned_graph

    # no length-1 RWPs
    assert all(len(p) >= 2 for p in paths)
    # the strong western train survived (a long path remains)
    assert max(len(p) for p in paths) >= 8
    # no in-band spatial overlap between any two returned RWPs
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            assert not _paths_interleave_in_band(g, paths[i], paths[j], 15.0)
