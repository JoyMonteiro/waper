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


def test_alternating_crests_troughs_connected(simple_wave_field, default_config):
    # simple_wave_field has 3 crests and 2 troughs
    # The association graph connects max nodes to min nodes (tuple-based IDs)
    ts_data = _identify_rwps(simple_wave_field, default_config)
    G = ts_data.association_graph

    assert len(G.nodes) > 0
    # Check it is bipartite: edges only connect max to min nodes
    for u, v in G.edges():
        types = {u[0], v[0]}
        assert types == {"max", "min"}


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


def test_date_line_association(date_line_wave_field, default_config):
    ts_data = _identify_rwps(date_line_wave_field, default_config)
    G = ts_data.association_graph
    # We expect an edge connecting a max near 360 to a min near 0, or vice versa.
    # We'll just assert that the graph has edges, showing the feature was identified.
    assert len(G.edges) > 0


def test_get_ranked_paths_independent_set():
    from waper.identification.rwp_graph import get_ranked_paths

    G = nx.Graph()
    # Path A: nodes 1-2-3 (Weight 10)
    # Path B: nodes 3-4-5 (Weight 8)
    # Path C: nodes 5-6-7 (Weight 6)
    # A and B overlap at 3. B and C overlap at 5. A and C do not overlap.
    # We expect A and C to be kept.
    G.add_edge(1, 2, weight=5)
    G.add_edge(2, 3, weight=5)
    
    G.add_edge(3, 4, weight=4)
    G.add_edge(4, 5, weight=4)
    
    G.add_edge(5, 6, weight=3)
    G.add_edge(6, 7, weight=3)
    
    # We also need to add start/end attributes because get_ranked_paths checks for "degree == 1" leaves.
    # Actually get_ranked_paths enumerates all simple paths between any degree 1 nodes.
    # So 1 and 7 will be leaves. But there are no branches, so there's only one simple path 1-2-3-4-5-6-7.
    # Let's mock a graph with branching to create overlapping paths.
    
    G2 = nx.Graph()
    # Path A: 1-2-3-4
    # Path B: 2-5-6
    # Path C: 6-7-8
    G2.add_edge(1, 2, weight=10)
    G2.add_edge(2, 3, weight=10)
    G2.add_edge(3, 4, weight=10) # A weight = 30
    
    G2.add_edge(2, 5, weight=2)
    G2.add_edge(5, 6, weight=2) # B weight = 4 (actually B shares node 2 with A)
    
    G2.add_edge(6, 7, weight=20)
    G2.add_edge(7, 8, weight=20) # C weight = 40 (C shares node 6 with B, but not with A)
    
    # Leaves are 1, 4, 8. 
    # Paths from leaves:
    # 1 to 4: 1-2-3-4 (w=30)  -> A
    # 1 to 8: 1-2-5-6-7-8 (w=10+2+2+20+20=54) -> this is actually the heaviest path and spans the whole thing.
    # To properly isolate paths, we need separate subcomponents or we just test the greedy selection itself.
    
    # Let's write a targeted test:
    # We will build a graph where all candidate paths are extracted.
    # To test the ranking/filtering logic specifically, we can see if get_ranked_paths works.
    
    # Let's just create a star graph from a central node 0
    G_star = nx.Graph()
    for i in range(8):
        G_star.add_node(i, coords=(180 + i, 50))  # dummy coords so is_to_the_east works, left to right
    
    G_star.add_edge(1, 0, weight=10)
    G_star.add_edge(0, 2, weight=10)  # Path 1-0-2 (w=20)
    
    G_star.add_edge(3, 0, weight=5)
    G_star.add_edge(0, 4, weight=5)   # Path 3-0-4 (w=10)
    
    G_star.add_edge(5, 6, weight=15)
    G_star.add_edge(6, 7, weight=15)  # Path 5-6-7 (w=30, disjoint)
    
    # All paths from leaves to leaves.
    paths = get_ranked_paths(G_star, max_weight=100)
    
    # Expected: 5-6-7 (w=30) and 1-0-2 (w=20) should be selected.
    # 3-0-4 (w=10) overlaps with 1-0-2 at node 0, so it should be dropped.
    
    # Check that 5-6-7 is in there
    assert [5, 6, 7] in paths or [7, 6, 5] in paths
    # Check that 1-0-2 is in there
    assert [1, 0, 2] in paths or [2, 0, 1] in paths
    # Check that 3-0-4 is NOT in there
    assert [3, 0, 4] not in paths
    assert [4, 0, 3] not in paths
