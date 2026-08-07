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
    max_region_ids = max_points["RegionId"]
    min_region_ids = min_points["RegionId"]

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

    # Map each cluster to its connected region (for wrap detection).
    cluster_max_region = {}
    cluster_min_region = {}


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
        cluster_max_region[int(cluster_id)] = int(max_region_ids[i])
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
        cluster_min_region[int(cluster_id)] = int(min_region_ids[i])
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
            region_id=cluster_max_region.get(max_id, -1),
        )
        assoc_graph.add_node(
            min_node_id,
            coords=min_centre,
            spherical_coords=min_centre_spherical,
            cluster_id=min_id,
            scalar=min_scalar,
            node_type="min",
            cluster_extrema=cluster_min_dict[min_id],
            region_id=cluster_min_region.get(min_id, -1),
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
                region_id=assoc_graph.nodes[start_node]["region_id"],
            )
            pruned_graph.add_node(
                end_node,
                coords=assoc_graph.nodes[end_node]["coords"],
                spherical_coords=assoc_graph.nodes[end_node]["spherical_coords"],
                cluster_id=assoc_graph.nodes[end_node]["cluster_id"],
                scalar=assoc_graph.nodes[end_node]["scalar"],
                node_type=assoc_graph.nodes[end_node]["node_type"],
                cluster_extrema=assoc_graph.nodes[end_node]["cluster_extrema"],
                region_id=assoc_graph.nodes[end_node]["region_id"],
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

    lon_max = assoc_graph.nodes[max_id]["coords"][0]
    lat_max = assoc_graph.nodes[max_id]["coords"][1]
    lon_min = assoc_graph.nodes[min_id]["coords"][0]
    lat_min = assoc_graph.nodes[min_id]["coords"][1]

    curr_dist = haversine_distance(lat_max, lon_max, lat_min, lon_min)

    # Orientation penalty: downweight edges that are more N-S oriented.
    # zonal_fraction = cos(atan(dlat/dlon)) = dlon / sqrt(dlon² + dlat²)
    dlon = _longitude_separation(lon_max, lon_min)
    dlat = abs(lat_max - lat_min)
    zonal_fraction = dlon / max((dlon**2 + dlat**2)**0.5, 1e-6)

    # Ensure we don't divide by zero if centroids overlap exactly
    edge_weight = (max_scalar - min_scalar) / max(curr_dist, 1e-6) * zonal_fraction

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
                region_id=assoc_graph.nodes[start_node]["region_id"],
            )
            pruned_graph.add_node(
                end_node,
                coords=assoc_graph.nodes[end_node]["coords"],
                spherical_coords=assoc_graph.nodes[end_node]["spherical_coords"],
                cluster_id=assoc_graph.nodes[end_node]["cluster_id"],
                scalar=assoc_graph.nodes[end_node]["scalar"],
                node_type=assoc_graph.nodes[end_node]["node_type"],
                cluster_extrema=assoc_graph.nodes[end_node]["cluster_extrema"],
                region_id=assoc_graph.nodes[end_node]["region_id"],
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


def _path_circles_globe(assoc_graph, path, full_circle=360.0):
    """Return True if the path wraps all the way around the globe.

    A genuine wrap is when the cumulative eastward longitude travelled along the
    path reaches a full circle, meaning it has looped back to revisit the same
    longitudes — which is non-physical for a single RWP.

    This replaces the old shared-``region_id`` test: ``region_id`` is the
    connected-region label of the *thresholded field*, and the successive
    troughs (or crests) of one coherent wave train legitimately lie on the same
    continuous anomaly ribbon, so they share a region. Splitting on that shredded
    real wave trains; only an actual globe-circling loop is a wrap.
    """
    total = 0.0
    for i in range(len(path) - 1):
        lon_a = assoc_graph.nodes[path[i]]["coords"][0]
        lon_b = assoc_graph.nodes[path[i + 1]]["coords"][0]
        total += _longitude_separation(lon_a, lon_b)
        if total >= full_circle:
            return True
    return False


def _split_at_weakest_edge(assoc_graph, path):
    """Split *path* at its lowest-weight edge, returning two sub-paths.

    Each sub-path must have at least 2 nodes to be valid.
    """
    if len(path) < 3:
        return [path]

    min_weight = float("inf")
    min_idx = -1
    for i in range(len(path) - 1):
        w = assoc_graph[path[i]][path[i + 1]]["weight"]
        if w < min_weight:
            min_weight = w
            min_idx = i

    left = path[: min_idx + 1]
    right = path[min_idx + 1 :]

    result = []
    if len(left) >= 2:
        result.append(left)
    if len(right) >= 2:
        result.append(right)
    return result


def _unwrap_path(assoc_graph, path):
    """Recursively split a path until no sub-path circles the globe."""
    if not _path_circles_globe(assoc_graph, path):
        return [path]

    sub_paths = _split_at_weakest_edge(assoc_graph, path)
    result = []
    for sp in sub_paths:
        result.extend(_unwrap_path(assoc_graph, sp))
    return result


def _path_lon_span(assoc_graph, path):
    """Eastward longitude arc of a monotonic-east path: (start_lon, arc_length_deg)."""
    start = assoc_graph.nodes[path[0]]["coords"][0]
    length = 0.0
    for i in range(len(path) - 1):
        a = assoc_graph.nodes[path[i]]["coords"][0]
        b = assoc_graph.nodes[path[i + 1]]["coords"][0]
        length += _longitude_separation(a, b)
    return start, length


def _arc_bins(start, length, full=360.0, step=1.0):
    """Integer-degree bins covered by the eastward arc [start, start+length] (mod 360)."""
    n = int(length // step) + 1
    return {int(round((start + k * step) % full)) % 360 for k in range(n)}


def _arcs_overlap(start_a, len_a, start_b, len_b):
    """True if two eastward longitude arcs share any longitude (wrap-aware)."""
    return not _arc_bins(start_a, len_a).isdisjoint(_arc_bins(start_b, len_b))


def _path_lat_range(assoc_graph, path):
    lats = [assoc_graph.nodes[n]["coords"][1] for n in path]
    return min(lats), max(lats)


def _lat_ranges_within(range_a, range_b, gate):
    """True if the gap between two [min,max] latitude ranges is <= gate (overlap -> 0)."""
    overlap_lo = max(range_a[0], range_b[0])
    overlap_hi = min(range_a[1], range_b[1])
    if overlap_lo <= overlap_hi:          # ranges intersect -> gap is 0
        return True
    return (overlap_lo - overlap_hi) <= gate


def _paths_interleave_in_band(assoc_graph, path_a, path_b, lat_gate):
    """True if two paths overlap in longitude AND lie within lat_gate of each other."""
    sa, la = _path_lon_span(assoc_graph, path_a)
    sb, lb = _path_lon_span(assoc_graph, path_b)
    if not _arcs_overlap(sa, la, sb, lb):
        return False
    return _lat_ranges_within(
        _path_lat_range(assoc_graph, path_a),
        _path_lat_range(assoc_graph, path_b),
        lat_gate,
    )


def get_ranked_paths(assoc_graph, max_weight, lat_gate=15.0):

    path_list = []

    start_leaves = [x for x in assoc_graph.nodes()]
    end_leaves = [x for x in assoc_graph.nodes()]

    for source in start_leaves:
        for sink in end_leaves:
            if source == sink:
                continue
            # eliminate sinks to the west of source node
            if is_to_the_east(
                assoc_graph.nodes[source]["coords"][0], assoc_graph.nodes[sink]["coords"][0]
            ):
                continue

            if nx.has_path(assoc_graph, source=source, target=sink):
                for path in nx.all_simple_paths(assoc_graph, source=source, target=sink):
                    if _is_monotonic_east(assoc_graph, path):
                        # Split paths that wrap around to the same region.
                        for unwrapped in _unwrap_path(assoc_graph, path):
                            path_list.append(unwrapped)

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
        if not path_nodes.isdisjoint(used_nodes):
            continue
        if any(_paths_interleave_in_band(assoc_graph, path, ap, lat_gate)
               for ap in top_paths):
            continue
        top_paths.append(path)
        used_nodes.update(path_nodes)

    return reassign_orphans(assoc_graph, top_paths, lat_gate=lat_gate)


def reassign_orphans(assoc_graph, top_paths, lat_gate=15.0, max_iter=50):
    """Absorb leftover (orphan) nodes into the stronger branch, drop the weaker.

    An orphan attaches to an in-RWP neighbour within ``lat_gate`` degrees of
    latitude. If it would extend a chain end (the existing arm on its side is
    empty) it is absorbed. Otherwise it competes with that arm by summed edge
    weight: the weaker arm is dropped (its nodes re-orphan and may re-attach on a
    later iteration). Orphans with no eligible neighbour are dropped.
    """
    paths = [list(p) for p in top_paths]

    def arm_weight(path, j, direction):
        w = 0.0
        i = j
        while 0 <= i + direction < len(path):
            a, b = path[i], path[i + direction]
            w += assoc_graph[a][b]["weight"]
            i += direction
        return w

    dropped = set()
    for _ in range(max_iter):
        assigned = {n for p in paths for n in p}
        orphans = [n for n in assoc_graph.nodes()
                   if n not in assigned and n not in dropped]
        progressed = False

        for o in orphans:
            o_lon, o_lat = assoc_graph.nodes[o]["coords"]
            cands = [
                (nb, assoc_graph[o][nb]["weight"])
                for nb in assoc_graph.neighbors(o)
                if nb in assigned
                and abs(assoc_graph.nodes[nb]["coords"][1] - o_lat) <= lat_gate
            ]
            if not cands:
                continue

            nb, w_o = max(cands, key=lambda c: c[1])
            pi = next(i for i, p in enumerate(paths) if nb in p)
            path = paths[pi]
            j = path.index(nb)
            direction = 1 if is_to_the_east(o_lon, assoc_graph.nodes[nb]["coords"][0]) else -1
            existing = arm_weight(path, j, direction)

            if w_o <= existing:
                dropped.add(o)                      # orphan's branch is weaker -> drop it
            elif direction == 1:
                candidate = path[: j + 1] + [o]
                # Guard: reject if the extended path now interleaves-in-band with
                # any other currently-accepted path (reproducing pass-1 invariant).
                if any(_paths_interleave_in_band(assoc_graph, candidate, paths[k], lat_gate)
                       for k in range(len(paths)) if k != pi):
                    dropped.add(o)                  # would re-introduce overlap -> drop
                else:
                    paths[pi] = candidate           # safe: splice orphan on east end
            else:
                candidate = [o] + path[j:]
                # Same guard for west-side splice.
                if any(_paths_interleave_in_band(assoc_graph, candidate, paths[k], lat_gate)
                       for k in range(len(paths)) if k != pi):
                    dropped.add(o)                  # would re-introduce overlap -> drop
                else:
                    paths[pi] = candidate           # safe: splice orphan on west end
            progressed = True
            break                                   # recompute assignment after each change

        if not progressed:
            break

    return [p for p in paths if len(p) >= 2]
