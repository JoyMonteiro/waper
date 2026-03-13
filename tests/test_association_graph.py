import networkx as nx
import numpy as np
import pytest

from waper.interface.api import WaperConfig, _identify_rwps


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


@pytest.mark.xfail(reason="Phase 2.2: Node ID Collision")
def test_alternating_crests_troughs_connected(simple_wave_field, default_config):
    # simple_wave_field has 3 crests and 2 troughs
    # The association graph connects max nodes (positive IDs) to min nodes (negative IDs)
    ts_data = _identify_rwps(simple_wave_field, default_config)
    G = ts_data.association_graph

    assert len(G.nodes) > 0
    # Check it is bipartite: edges only connect positive to negative
    for u, v in G.edges():
        assert (u > 0 and v < 0) or (u < 0 and v > 0)


@pytest.mark.xfail(reason="Phase 2.1: empty dataset causes KeyError")
def test_isolated_max_no_adjacent_min(single_maximum_field, default_config):
    # Field only has one positive bump, no minima above the threshold
    ts_data = _identify_rwps(single_maximum_field, default_config)
    G = ts_data.association_graph

    # Association graph requires both maxima and minima near the 0-contour
    # It should have 0 edges if there are no minima
    assert len(G.edges) == 0


def test_node_pruning_removes_weak_nodes(simple_wave_field):
    config = WaperConfig(
        debug=False,
        scalar_name="v",
        latitude_label="latitude",
        longitude_label="longitude",
        time_label="time",
        clip_value=2,
        extrema_threshold=5,  # low threshold to catch more
        max_latitude=80.1,
        min_latitude=20,
        node_pruning_threshold=25,  # high pruning threshold
        edge_pruning_threshold=1e-5,
        track_pruning_threshold=0.3,
        max_edge_weight=1,
    )

    ts_data = _identify_rwps(simple_wave_field, config)
    G_assoc = ts_data.association_graph
    from waper.identification import rwp_graph

    G_pruned = rwp_graph.prune_association_graph_nodes(G_assoc, 25)

    # Assert any node in G_pruned has |scalar| >= 25
    for node in G_pruned.nodes():
        assert abs(G_pruned.nodes[node]["scalar"]) >= 25


def test_edge_pruning_removes_low_gradient(simple_wave_field):
    config = WaperConfig(
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
        edge_pruning_threshold=0.5,  # very high gradient threshold
        track_pruning_threshold=0.3,
        max_edge_weight=100,
    )

    ts_data = _identify_rwps(simple_wave_field, config)
    G_pruned = ts_data.pruned_graph

    # High threshold should leave very few or zero edges
    # compared to association_graph
    assert len(G_pruned.edges) < len(ts_data.association_graph.edges)

    for u, v, data in G_pruned.edges(data=True):
        assert data["weight"] >= 0.5


@pytest.mark.xfail(
    reason="Date line wrapping association may have bugs to fix in Phase 3"
)
def test_date_line_association(date_line_wave_field, default_config):
    ts_data = _identify_rwps(date_line_wave_field, default_config)
    G = ts_data.association_graph
    # We expect an edge connecting a max near 360 to a min near 0, or vice versa.
    # We'll just assert that the graph has edges, showing the feature was identified.
    assert len(G.edges) > 0
