from matplotlib.font_manager import weight_dict
import networkx as nx
from networkx import Graph
from itertools import product
from tqdm import tqdm

from .quadtree import merge, compute_size_features
from ..identification.utils import haversine_distance


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
                if abs(feature - rwp_info["rwp_id"]) < 1e-2:
                    lon = rwp_info["weighted_longitude"]
                    lat = rwp_info["weighted_latitude"]

            if lon == 0:
                print(feature)

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
                prev_feature_size = compute_size_features(time_step_data[time - 1].quadtree)
                curr_feature_size = compute_size_features(time_step_data[time].quadtree)
                # print(prev_feature_size)
                # print(curr_feature_size)

                for edge in edge_list:

                    if (edge in merge_feature_size) or (edge[::-1] in merge_feature_size):
                        # print(edge, merge_feature_size[edge])
                        # print(edge)
                        # print(prev_feature_size[tuple([edge[0]])], curr_feature_size[tuple([edge[1]])])
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
        tracking_graph.edges[edge]["distance"] = distance / 1000

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
            pruned_graph.add_node(edge[0], coords=tracking_graph.nodes[edge[0]]["coords"])
            pruned_graph.add_node(edge[1], coords=tracking_graph.nodes[edge[1]]["coords"])
            pruned_graph.add_edge(
                edge[0], edge[1],
                weight=tracking_graph.edges[edge]["weight"],
                distance=tracking_graph.edges[edge]["distance"],
            )
            
    return pruned_graph


def get_track_paths(tracking_graph):

    track_paths = []

    end_nodes = [
        node
        for node in tracking_graph.nodes()
        if tracking_graph.in_degree(node) != 0 and tracking_graph.out_degree(node) == 0
    ]

    start_nodes = [
        node
        for node in tracking_graph.nodes()
        if tracking_graph.in_degree(node) == 0 and tracking_graph.out_degree(node) > 0
    ]

    all_combinations = product(start_nodes, end_nodes)

    for start_end in all_combinations:

        if nx.has_path(tracking_graph, source=start_end[0], target=start_end[1]):
            for path in nx.all_simple_paths(
                tracking_graph, source=start_end[0], target=start_end[1]
            ):
                track_paths.append(path)

    return track_paths
