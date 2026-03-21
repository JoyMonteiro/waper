import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict
from .utils import haversine_distance, is_to_the_east, _longitude_separation

WAPER_MAX_SCALAR_VALUE = 100
WAPER_MAX_NODE_DISTANCE = 1000


def compute_association_graph(max_points, min_points, iso_contour, scalar_name):
    """Compute the association graph by identifying the closest maxima/minima to a point on the
    given isocontour

    Args:
        max_points (vtk): object containing all the maxima identified
        min_points (vtk): object containing all the minima identified
        iso_contour (vtk): object containing all points on the isocontour
        scalar_name (string): name of the scalar field

    Returns:
        nx.Graph: the association graph
    """
    # creating an empty graph
    assoc_graph = nx.Graph()

    if max_points.n_points == 0 or min_points.n_points == 0:
        return assoc_graph

    num_contour_pts = iso_contour.n_points

    max_cluster_ids = max_points["Cluster ID"]
    min_cluster_ids = min_points["Cluster ID"]

    num_max_pts = max_points.n_points
    num_min_pts = min_points.n_points
    num_max_clusters = int(np.max(max_cluster_ids) + 1)
    num_min_clusters = int(np.max(min_cluster_ids) + 1)

    cluster_max_arr = np.full(num_max_clusters, 0.0)
    cluster_min_arr = np.full(num_min_clusters, 100.0)
    cluster_max_point = np.full((num_max_clusters, 2), 0.0)
    cluster_min_point = np.full((num_min_clusters, 2), 0.0)
    cluster_max_spherical_coord = np.full((num_max_clusters, 3), 0.0)
    cluster_min_spherical_coord = np.full((num_min_clusters, 3), 0.0)


    assoc_set = set()

    max_scalars = max_points[scalar_name]
    min_scalars = min_points[scalar_name]

    cluster_max_dict = defaultdict(list)
    cluster_min_dict = defaultdict(list)

    for i in range(num_max_pts):
        point_coords = max_points["Longitude"][i], max_points["Latitude"][i]
        cluster_id = max_cluster_ids[i]
        scalar = max_scalars[i]
        point_tuple = (point_coords, cluster_id, scalar)
        cluster_max_dict[cluster_id].append(point_tuple)
        if cluster_max_arr[max_cluster_ids[i]] < max_scalars[i]:
            cluster_max_arr[max_cluster_ids[i]] = max_scalars[i]
            cluster_max_point[max_cluster_ids[i]][0] = point_coords[0]
            cluster_max_point[max_cluster_ids[i]][1] = point_coords[1]
            cluster_max_spherical_coord[max_cluster_ids[i]][:] = max_points.points[i]

    for i in range(num_min_pts):
        point_coords = min_points["Longitude"][i], min_points["Latitude"][i]
        cluster_id = min_cluster_ids[i]
        scalar = min_scalars[i]
        point_tuple = (point_coords, cluster_id, scalar)
        cluster_min_dict[cluster_id].append(point_tuple)
        if cluster_min_arr[int(min_cluster_ids[i])] > min_scalars[i]:
            cluster_min_arr[int(min_cluster_ids[i])] = min_scalars[i]
            cluster_min_point[int(min_cluster_ids[i])][0] = point_coords[0]
            cluster_min_point[int(min_cluster_ids[i])][1] = point_coords[1]
            cluster_min_spherical_coord[min_cluster_ids[i]][:] = min_points.points[i]

    contour_points = iso_contour.points
    min_points_array = min_points.points
    max_points_array = max_points.points

    max_tree = cKDTree(max_points_array)
    min_tree = cKDTree(min_points_array)

    _, max_indices = max_tree.query(contour_points)
    _, min_indices = min_tree.query(contour_points)

    for i in range(num_contour_pts):
        max_id = int(max_cluster_ids[max_indices[i]])
        min_id = int(min_cluster_ids[min_indices[i]])
        if max_id != -1 and min_id != -1:
            assoc_set.add((max_id, min_id))

    count = 0

    for elem in assoc_set:
        count += 1
        max_id = elem[0]
        min_id = elem[1]
        max_centre = cluster_max_point[max_id]
        min_centre = cluster_min_point[min_id]
        max_scalar = cluster_max_arr[max_id]
        min_scalar = cluster_min_arr[min_id]

        max_centre_spherical = cluster_max_spherical_coord[max_id]
        min_centre_spherical = cluster_min_spherical_coord[min_id]

        max_node_id = ("max", max_id)
        min_node_id = ("min", min_id)

        assoc_graph.add_node(
            max_node_id,
            coords=max_centre,
            spherical_coords=max_centre_spherical,
            cluster_id=max_id,
            scalar=max_scalar,
            node_type="max",
            cluster_extrema=cluster_max_dict[max_id],
        )
        assoc_graph.add_node(
            min_node_id,
            coords=min_centre,
            spherical_coords=min_centre_spherical,
            cluster_id=min_id,
            scalar=min_scalar,
            node_type="min",
            cluster_extrema=cluster_min_dict[min_id],
        )

        assoc_graph.add_edge(max_node_id, min_node_id, weight=0)
    return assoc_graph


def prune_association_graph_nodes(assoc_graph, scalar_threshold):
    """Remove nodes from the association graph that fall below the
    threshold value

    Args:
        assoc_graph (nx.Graph): Association graph
        scalar_threshold (float): Threshold value

    Returns:
        nx.Graph: association graph with only nodes above threshold
    """

    pruned_graph = nx.Graph()
    edges = [e for e in assoc_graph.edges()]
    for e in edges:
        start_node = e[0]
        end_node = e[1]
        if assoc_graph.nodes[start_node]["node_type"] == "max":
            if (
                assoc_graph.nodes[start_node]["scalar"]
                < -assoc_graph.nodes[end_node]["scalar"]
            ):
                min_scalar = assoc_graph.nodes[start_node]["scalar"]

            else:
                min_scalar = -assoc_graph.nodes[end_node]["scalar"]
        else:
            if (
                -assoc_graph.nodes[start_node]["scalar"]
                < assoc_graph.nodes[end_node]["scalar"]
            ):
                min_scalar = -assoc_graph.nodes[start_node]["scalar"]
            else:
                min_scalar = assoc_graph.nodes[end_node]["scalar"]

        if min_scalar >= scalar_threshold and min_scalar <= WAPER_MAX_SCALAR_VALUE:
            pruned_graph.add_node(
                start_node,
                coords=assoc_graph.nodes[start_node]["coords"],
                spherical_coords=assoc_graph.nodes[start_node]["spherical_coords"],
                cluster_id=assoc_graph.nodes[start_node]["cluster_id"],
                scalar=assoc_graph.nodes[start_node]["scalar"],
                node_type=assoc_graph.nodes[start_node]["node_type"],
                cluster_extrema=assoc_graph.nodes[start_node]["cluster_extrema"],
            )
            pruned_graph.add_node(
                end_node,
                coords=assoc_graph.nodes[end_node]["coords"],
                spherical_coords=assoc_graph.nodes[end_node]["spherical_coords"],
                cluster_id=assoc_graph.nodes[end_node]["cluster_id"],
                scalar=assoc_graph.nodes[end_node]["scalar"],
                node_type=assoc_graph.nodes[end_node]["node_type"],
                cluster_extrema=assoc_graph.nodes[end_node]["cluster_extrema"],
            )
            pruned_graph.add_edge(start_node, end_node)

    return pruned_graph


def edge_weight(
    assoc_graph,
    max_id,
    min_id
):


    max_scalar = assoc_graph.nodes[max_id]["scalar"]
    min_scalar = assoc_graph.nodes[min_id]["scalar"]


    curr_dist = 0.0

    edge_weight = 0.0

    curr_dist = haversine_distance(
        assoc_graph.nodes[max_id]["coords"][1],
        assoc_graph.nodes[max_id]["coords"][0],
        assoc_graph.nodes[min_id]["coords"][1],
        assoc_graph.nodes[min_id]["coords"][0],
    )

    # Ensure we don't divide by zero if centroids overlap exactly
    edge_weight = (max_scalar - min_scalar) / max(curr_dist, 1e-6)

    return edge_weight


def prune_association_graph_edges(
    assoc_graph, threshold, max_weight,
    min_longitude_separation=6.0, max_aspect_ratio=1.5,
):
    """Remove edges which fall below edge weight thresholds

    Args:
        assoc_graph (nx.Graph): current association graph
        threshold (float): weight threshold for pruning
        max_weight (float): maximum likely value for edge weight
        min_longitude_separation (float): minimum angular distance between extrema
        max_aspect_ratio (float): maximum |Δlat|/|Δlon| — edges steeper than
            this are discarded as nearly-vertical connections

    Returns:
        nx.Graph: association graph with low weight edges pruned
    """

    pruned_graph = nx.Graph()
    edges = [e for e in assoc_graph.edges()]

    for e in edges:
        start_node = e[0]
        end_node = e[1]

        lon_0 = assoc_graph.nodes[start_node]["coords"][0]
        lon_1 = assoc_graph.nodes[end_node]["coords"][0]
        lat_0 = assoc_graph.nodes[start_node]["coords"][1]
        lat_1 = assoc_graph.nodes[end_node]["coords"][1]

        dlon = _longitude_separation(lon_0, lon_1)
        if dlon <= min_longitude_separation:
            continue

        dlat = abs(lat_0 - lat_1)
        if dlat / max(dlon, 1e-6) > max_aspect_ratio:
            continue

        if assoc_graph.nodes[start_node]["node_type"] == "max":
            weight = edge_weight(assoc_graph, start_node, end_node)
        else:
            weight = edge_weight(assoc_graph, end_node, start_node)
        assoc_graph[start_node][end_node]["weight"] = weight

        if weight >= threshold and weight <= max_weight:
            pruned_graph.add_node(
                start_node,
                coords=assoc_graph.nodes[start_node]["coords"],
                spherical_coords=assoc_graph.nodes[start_node]["spherical_coords"],
                cluster_id=assoc_graph.nodes[start_node]["cluster_id"],
                scalar=assoc_graph.nodes[start_node]["scalar"],
                node_type=assoc_graph.nodes[start_node]["node_type"],
                cluster_extrema=assoc_graph.nodes[start_node]["cluster_extrema"],
            )
            pruned_graph.add_node(
                end_node,
                coords=assoc_graph.nodes[end_node]["coords"],
                spherical_coords=assoc_graph.nodes[end_node]["spherical_coords"],
                cluster_id=assoc_graph.nodes[end_node]["cluster_id"],
                scalar=assoc_graph.nodes[end_node]["scalar"],
                node_type=assoc_graph.nodes[end_node]["node_type"],
                cluster_extrema=assoc_graph.nodes[end_node]["cluster_extrema"],
            )
            pruned_graph.add_edge(start_node, end_node, weight=weight)
    return pruned_graph

def _is_monotonic_east(assoc_graph, path):
    """Return True if every successive node in *path* is east of the previous."""
    for i in range(len(path) - 1):
        lon_a = assoc_graph.nodes[path[i]]["coords"][0]
        lon_b = assoc_graph.nodes[path[i + 1]]["coords"][0]
        if is_to_the_east(lon_a, lon_b):
            return False
    return True


def get_ranked_paths(assoc_graph, max_weight):

    path_list = []

    start_leaves = [x for x in assoc_graph.nodes()]
    end_leaves = [x for x in assoc_graph.nodes()]

    for source in start_leaves:
        for sink in end_leaves:
            # eliminate sinks to the west of source node
            if is_to_the_east(
                assoc_graph.nodes[source]["coords"][0], assoc_graph.nodes[sink]["coords"][0]
            ):
                continue

            if nx.has_path(assoc_graph, source=source, target=sink):
                for path in nx.all_simple_paths(assoc_graph, source=source, target=sink):
                    if _is_monotonic_east(assoc_graph, path):
                        path_list.append(path)

    path_wt_dict = {}

    for path in path_list:
        curr_wt = 0
        for i in range(len(path) - 1):
            curr_wt += assoc_graph[path[i]][path[i + 1]]["weight"]
        path_wt_dict[tuple(path)] = curr_wt

    sorted_paths = sorted(path_list, key=lambda p: path_wt_dict[tuple(p)], reverse=True)

    top_paths = []
    used_nodes = set()

    for path in sorted_paths:
        path_nodes = set(path)
        if path_nodes.isdisjoint(used_nodes):
            top_paths.append(path)
            used_nodes.update(path_nodes)

    return top_paths
