import networkx as nx
import numpy as np
from collections import defaultdict
from .utils import haversine_distance, is_to_the_east

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

    num_contour_pts = iso_contour.n_points
    # point_grad = iso_contour.GetPointData().GetArray("Gradients")

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
    # assoc_index_array = np.full((num_max_clusters, num_min_clusters), 0.0)

    # line_dir_array = np.full((num_max_clusters, num_min_clusters), 0.0)

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
    for i in range(num_contour_pts):
        contour_point = contour_points[i]
        max_dist = WAPER_MAX_NODE_DISTANCE
        min_dist = WAPER_MAX_NODE_DISTANCE
        max_id = -1
        min_id = -1
        # curr_max_dir_deriv = 0
        # curr_min_dir_deriv = 0
        # grad_vector = [point_grad.GetTuple3(i)[0], point_grad.GetTuple3(i)[1]]
        # curr_max_scalar = 0
        # curr_min_scalar = 0

        for j in range(num_max_pts):
            max_point = max_points_array[j]
            curr_max_id = max_cluster_ids[j]
            max_dir_vector = [max_point[0] - contour_point[0], max_point[1] - contour_point[1]]
            # max_dir_deriv = (
            #     max_dir_vector[0] * grad_vector[0] + max_dir_vector[1] * grad_vector[1]
            # )
            curr_max_dist = (max_dir_vector[0] ** 2 + max_dir_vector[1] ** 2) ** 0.5
            # if(max_dir_deriv>0):
            if curr_max_dist < max_dist:
                max_dist = curr_max_dist
                max_id = curr_max_id
                # curr_max_dir_deriv = max_dir_deriv
                # curr_max_scalar = max_scalars.GetTuple1(j)
                # curr_max_x = max_point[0]

        max_id = int(max_id)
        #         point_cords_max = cluster_max_point[max_id]
        #         point_tuple_max = (point_cords_max, max_id, cluster_max_arr[max_id])

        for j in range(num_min_pts):
            min_point = min_points_array[j]
            curr_min_id = min_cluster_ids[j]
            min_dir_vector = [min_point[0] - contour_point[0], min_point[1] - contour_point[1]]
            # min_dir_deriv = (
            #     min_dir_vector[0] * grad_vector[0] + min_dir_vector[1] * grad_vector[1]
            # )
            curr_min_dist = (min_dir_vector[0] ** 2 + min_dir_vector[1] ** 2) ** 0.5
            # if(min_dir_deriv > 0):
            if curr_min_dist < min_dist:
                min_dist = curr_min_dist
                min_id = curr_min_id
                # curr_min_dir_deriv = min_dir_deriv
                # curr_min_scalar = min_scalars.GetTuple1(j)
                # curr_min_x = min_point[0]

        min_id = int(min_id)
        #         point_cords_min = cluster_min_point[min_id]
        #         point_tuple_min = (point_cords_min, min_id, cluster_min_arr[min_id])
        if max_id != -1 and min_id != -1:
            assoc_set.add((int(max_id), int(min_id)))

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

        if min_id == 0:
            min_id = 100

        assoc_graph.add_node(
            max_id,
            coords=max_centre,
            spherical_coords=max_centre_spherical,
            cluster_id=max_id,
            scalar=max_scalar,
            cluster_extrema=cluster_max_dict[max_id],
        )

        if min_id == 100:
            assoc_graph.add_node(
                -min_id,
                coords=min_centre,
                spherical_coords=min_centre_spherical,
                cluster_id=min_id,
                scalar=min_scalar,
                cluster_extrema=cluster_min_dict[0],
            )
        else:
            assoc_graph.add_node(
                -min_id,
                coords=min_centre,
                spherical_coords=min_centre_spherical,
                cluster_id=min_id,
                scalar=min_scalar,
                cluster_extrema=cluster_min_dict[min_id],
            )

        assoc_graph.add_edge(max_id, -min_id, weight=0)
        # print("no. of associations", count)
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
        # min_node = 0
        min_scalar = 0

        if start_node >= 0:
            if (
                assoc_graph.nodes[start_node]["scalar"]
                < -assoc_graph.nodes[end_node]["scalar"]
            ):
                # min_node = start_node
                min_scalar = assoc_graph.nodes[start_node]["scalar"]

            else:
                # min_node = end_node
                min_scalar = -assoc_graph.nodes[end_node]["scalar"]
        else:
            if (
                -assoc_graph.nodes[start_node]["scalar"]
                < assoc_graph.nodes[end_node]["scalar"]
            ):
                # min_node = start_node
                min_scalar = -assoc_graph.nodes[start_node]["scalar"]
            else:
                # min_node = end_node
                min_scalar = assoc_graph.nodes[end_node]["scalar"]

        if min_scalar >= scalar_threshold and min_scalar <= WAPER_MAX_SCALAR_VALUE:
            pruned_graph.add_node(
                start_node,
                coords=assoc_graph.nodes[start_node]["coords"],
                spherical_coords=assoc_graph.nodes[start_node]["spherical_coords"],
                cluster_id=assoc_graph.nodes[start_node]["cluster_id"],
                scalar=assoc_graph.nodes[start_node]["scalar"],
                cluster_extrema=assoc_graph.nodes[start_node]["cluster_extrema"],
            )
            pruned_graph.add_node(
                end_node,
                coords=assoc_graph.nodes[end_node]["coords"],
                spherical_coords=assoc_graph.nodes[end_node]["spherical_coords"],
                cluster_id=assoc_graph.nodes[end_node]["cluster_id"],
                scalar=assoc_graph.nodes[end_node]["scalar"],
                cluster_extrema=assoc_graph.nodes[end_node]["cluster_extrema"],
            )
            pruned_graph.add_edge(start_node, end_node)

    return pruned_graph


def edge_weight(
    assoc_graph,
    max_id,
    min_id
    # , high_value_threshold,
    # scalar_threshold, scalar_tolerance
):

    # scalar_tol = 30

    max_scalar = assoc_graph.nodes[max_id]["scalar"]
    min_scalar = assoc_graph.nodes[min_id]["scalar"]

    # cluster_max_pts = assoc_graph.nodes[max_id]["cluster_extrema"]
    # cluster_min_pts = assoc_graph.nodes[min_id]["cluster_extrema"]

    curr_dist = 0.0

    edge_weight = 0.0
    # high_value_flag = 0

    # if max_scalar > high_value_threshold and min_scalar > high_value_threshold:
    # high_value_flag = 1

    curr_dist = haversine_distance(
        assoc_graph.nodes[max_id]["coords"][1],
        assoc_graph.nodes[max_id]["coords"][0],
        assoc_graph.nodes[min_id]["coords"][1],
        assoc_graph.nodes[min_id]["coords"][0],
    )

    edge_weight = (max_scalar - min_scalar) / curr_dist

    # for max_pt in cluster_max_pts:
    #     if max_pt[2] < scalar_threshold:
    #         continue
    #     if max_pt[2] < max_scalar - scalar_tolerance and high_value_flag == 0:
    #         continue

    #     for min_pt in cluster_min_pts:
    #         if min_pt[2] > -scalar_threshold:
    #             continue
    #         if min_pt[2] > -min_scalar + scalar_tolerance and high_value_flag == 0:
    #             continue
    #         curr_dist = haversine_distance(
    #             max_pt[0][0], max_pt[0][1], min_pt[0][0], min_pt[0][1]
    #         )
    #         curr_weight = (max_pt[2] - min_pt[2]) / curr_dist

    #         if curr_weight > edge_weight:
    #             edge_weight = curr_weight

    return edge_weight


def prune_association_graph_edges(assoc_graph, threshold, max_weight):
    """Remove edges which fall below edge weight thresholds

    Args:
        assoc_graph (nx.Graph): current association graph
        threshold (float): weight threshold for pruning
        max_weight (float): maximum likely value for edge weight

    Returns:
        nx.Graph: association graph with low weight edges pruned
    """

    pruned_graph = nx.Graph()
    edges = [e for e in assoc_graph.edges()]

    for e in edges:
        start_node = e[0]
        end_node = e[1]
        if start_node >= 0:
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
                cluster_extrema=assoc_graph.nodes[start_node]["cluster_extrema"],
            )
            pruned_graph.add_node(
                end_node,
                coords=assoc_graph.nodes[end_node]["coords"],
                spherical_coords=assoc_graph.nodes[end_node]["spherical_coords"],
                cluster_id=assoc_graph.nodes[end_node]["cluster_id"],
                scalar=assoc_graph.nodes[end_node]["scalar"],
                cluster_extrema=assoc_graph.nodes[end_node]["cluster_extrema"],
            )
            pruned_graph.add_edge(start_node, end_node, weight=weight)
    return pruned_graph


def get_ranked_paths(assoc_graph, max_weight):

    # H = nx.Graph()
    # H = assoc_graph
    path_list = []

    start_leaves = [x for x in assoc_graph.nodes()]
    end_leaves = [x for x in assoc_graph.nodes()]

    # print(len(start_leaves), "number of nodes in graph for rankedPaths")

    for source in start_leaves:
        # print(source)
        for sink in end_leaves:
            # eliminate sinks to the west of source node
            if is_to_the_east(
                assoc_graph.nodes[source]["coords"][0], assoc_graph.nodes[sink]["coords"][0]
            ):
                continue

            if nx.has_path(assoc_graph, source=source, target=sink):
                for path in nx.all_simple_paths(assoc_graph, source=source, target=sink):
                    consistent = True
                    for node in path[:-1]:
                        if is_to_the_east(
                            assoc_graph.nodes[node]["coords"][0], assoc_graph.nodes[path[-1]]["coords"][0]
                        ):
                            consistent = False
                    
                    if consistent:
                        path_list.append(path)

    path_wt_dict = {}

    # print(len(path_list), "number of paths found")

    for path in path_list:
        curr_wt = 0
        # print(path)
        for i in range(len(path) - 1):
            # print(assoc_graph.nodes[path[i]]["coords"][0], assoc_graph.nodes[path[i+1]]["coords"][0])
            curr_wt += assoc_graph[path[i]][path[i + 1]]["weight"]
        path_wt_dict[tuple(path)] = curr_wt

    top_paths = list(
        filter(
            lambda f: not any(
                [
                    (  # Condition reduces to "True if path weight is less than reference and both are part of the same path"
                        path_wt_dict[tuple(f)] < path_wt_dict[tuple(g)]
                        and len(set(f) & set(g)) != 0
                    )
                    for g in path_list
                ]
            ),
            path_list,
        )
    )

    return top_paths
