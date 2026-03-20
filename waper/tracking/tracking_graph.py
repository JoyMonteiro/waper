import logging
from itertools import product

import networkx as nx
from matplotlib.font_manager import weight_dict
from networkx import Graph
from tqdm import tqdm

from ..identification.utils import haversine_distance
from .quadtree import compute_size_features, merge

logger = logging.getLogger(__name__)


def build_tracking_graph(time_step_data, number_steps: int = None) -> Graph:
    """Build tracking graph based on overlap between quadtrees

    Args:
        time_step_data (list): list of identification data
        number_steps (int): number of timesteps to track over

    Returns:
        Graph: tracking graph with nodes corresponding to RWP features
        and edges connecting features in different time steps.
    """

    tracking_graph = nx.DiGraph()

    if number_steps is None:
        number_steps = len(time_step_data)

    for time in tqdm(range(number_steps)):
        for feature in time_step_data[time].raster_features:
            if feature == 0:
                continue

            lon = 0
            lat = 0
            for rwp_info in time_step_data[time].rwp_info.values():
                if feature == rwp_info["rwp_id"]:
                    lon = rwp_info["weighted_longitude"]
                    lat = rwp_info["weighted_latitude"]

            if lon == 0:
                logger.warning("Feature %s has no matching rwp_info", feature)

            tracking_graph.add_node((time, feature), coords=(lon, lat))
            if time > 0:
                edge_list = list(
                    product(
                        time_step_data[time - 1].raster_features,
                        time_step_data[time].raster_features,
                    )
                )
                merge_graph = merge(
                    time_step_data[time].quadtree, time_step_data[time - 1].quadtree
                )
                merge_feature_size = compute_size_features(merge_graph)
                prev_feature_size = compute_size_features(
                    time_step_data[time - 1].quadtree
                )
                curr_feature_size = compute_size_features(time_step_data[time].quadtree)

                for edge in edge_list:
                    if (edge in merge_feature_size) or (
                        edge[::-1] in merge_feature_size
                    ):
                        weight = merge_feature_size[edge] / max(
                            prev_feature_size[tuple([edge[0]])],
                            curr_feature_size[tuple([edge[1]])],
                        )
                        tracking_graph.add_edge(
                            (time - 1, edge[0]), (time, edge[1]), weight=weight
                        )

    for edge in tracking_graph.edges:
        lon1, lat1 = tracking_graph.nodes[edge[0]]["coords"]
        lon2, lat2 = tracking_graph.nodes[edge[1]]["coords"]
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        tracking_graph.edges[edge]["distance"] = distance

    return tracking_graph


def prune_tracking_graph(tracking_graph, threshold) -> Graph:
    """Remove edges with weight below threshold

    Args:
        tracking_graph (Graph): tracking graph
        threshold (float): threshold to prune at

    Returns:
        Graph: pruned tracking graph
    """

    pruned_graph = nx.DiGraph()

    for edge in tracking_graph.edges:
        if tracking_graph.edges[edge]["distance"] < threshold:
            pruned_graph.add_node(
                edge[0], coords=tracking_graph.nodes[edge[0]]["coords"]
            )
            pruned_graph.add_node(
                edge[1], coords=tracking_graph.nodes[edge[1]]["coords"]
            )
            pruned_graph.add_edge(
                edge[0],
                edge[1],
                weight=tracking_graph.edges[edge]["weight"],
                distance=tracking_graph.edges[edge]["distance"],
            )

    return pruned_graph


def get_path_weight(track_graph, path):
    return sum(
        track_graph[path[i]][path[i + 1]]["weight"]
        for i in range(len(path) - 1)
    )


def _greedy_select_independent_paths(track_paths, tracking_graph):
    """Keep highest-weight paths that share no nodes (greedy independent set)."""
    path_wt = {
        tuple(p): get_path_weight(tracking_graph, p) for p in track_paths
    }
    sorted_paths = sorted(track_paths, key=lambda p: path_wt[tuple(p)], reverse=True)

    top_paths = []
    used_nodes = set()
    for path in sorted_paths:
        path_nodes = set(path)
        if path_nodes.isdisjoint(used_nodes):
            top_paths.append(path)
            used_nodes.update(path_nodes)
    return top_paths


def get_track_paths(tracking_graph):
    """Extract tracks as longest-weight paths in the tracking DAG.

    Uses topological-sort DP: O(V + E) instead of factorial all_simple_paths.
    """
    if len(tracking_graph) == 0:
        return []

    topo_order = list(nx.topological_sort(tracking_graph))

    best_weight = {node: 0.0 for node in topo_order}
    predecessor = {node: None for node in topo_order}

    for node in topo_order:
        for succ in tracking_graph.successors(node):
            edge_wt = tracking_graph[node][succ]["weight"]
            candidate = best_weight[node] + edge_wt
            if candidate > best_weight[succ]:
                best_weight[succ] = candidate
                predecessor[succ] = node

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

    return _greedy_select_independent_paths(track_paths, tracking_graph)
