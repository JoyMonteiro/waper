from collections import defaultdict

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn import cluster

from .utils import RADIUS_EARTH_KM, RADIUS_SPHERE

CLUSTER_MAX_DISTANCE = 15000.0
SCALE_FACTOR = RADIUS_EARTH_KM / RADIUS_SPHERE

# scipy.sparse.csgraph marks "no predecessor" — the source itself, or a vertex
# unreachable from it — with this sentinel rather than -1.
_NO_PREDECESSOR = -9999


def _surface_graph(mesh):
    """Build the point-adjacency graph of a triangulated surface mesh.

    Nodes are mesh point IDs; edges are triangle edges weighted by the
    Euclidean distance between their endpoints. This is the graph
    ``vtkDijkstraGraphGeodesicPath`` walked internally, so shortest-path lengths
    over it are the same geodesic distances — in mesh units, i.e. on the scaled
    sphere of radius ``RADIUS_SPHERE``, which ``SCALE_FACTOR`` converts to km.

    Args:
        mesh (pv.PolyData): triangulated surface

    Returns:
        scipy.sparse.csr_matrix: symmetric (n_points, n_points) weight matrix
    """
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    edges = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    # Interior edges are shared by two triangles. De-duplicate them, or the
    # coo -> csr conversion would sum the duplicates into a doubled weight.
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]

    lengths = np.linalg.norm(
        mesh.points[edges[:, 0]] - mesh.points[edges[:, 1]], axis=1
    )
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    weights = np.concatenate([lengths, lengths])

    n = mesh.n_points
    return coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()


def _path_extremes(predecessors, values, sign):
    """Extremal scalar value along each shortest path out of one source.

    Given the predecessor row scipy returns for a single source, produce an
    array ``ext`` where ``ext[v]`` is the minimum (``sign > 0``) or maximum
    (``sign < 0``) of ``values`` over every vertex on the shortest path from the
    source to ``v``, both endpoints included.

    A shortest-path tree gives each vertex exactly one parent, so this is a
    bottleneck query up the tree. It is answered for every vertex at once by
    pointer doubling: round k folds in the ancestor 2^k steps up, so the whole
    tree resolves in O(log depth) vectorised passes rather than one Python walk
    per (source, target) pair.

    Args:
        predecessors (ndarray): parent of each vertex in the shortest-path tree
        values (ndarray): scalar value at each vertex
        sign (int): +1 for maxima (take the minimum), -1 for minima (the maximum)

    Returns:
        ndarray: the extremal value along the path to each vertex
    """
    ancestor = predecessors.astype(np.intp, copy=True)
    # Point roots and unreachable vertices at themselves, so doubling is a no-op
    # for them. Unreachable vertices are never queried — their pair distance is
    # infinite and the pair stays at CLUSTER_MAX_DISTANCE.
    detached = ancestor == _NO_PREDECESSOR
    ancestor[detached] = np.flatnonzero(detached)

    combine = np.minimum if sign > 0 else np.maximum
    ext = values.astype(float, copy=True)
    while True:
        ext = combine(ext, ext[ancestor])
        nxt = ancestor[ancestor]
        if np.array_equal(nxt, ancestor):
            return ext
        ancestor = nxt


def cluster_extrema(
    connectivity_clipped_scalar_field,
    extrema_points,
    scalar_name,
    sign,
    max_eps_km=1500,
    min_samples=2,
    xi=0.05,
    penalty_length_scale_km=2000.0,
):
    """Cluster extrema (maxima or minima) in the scalar field.

    Extrema in the same connected region are separated by the geodesic distance
    between them across the clipped surface, plus a hill-climbing penalty for
    paths that leave the shared ridge (or trough). DBSCAN over that precomputed
    distance then groups them.

    Args:
        connectivity_clipped_scalar_field (pv.PolyData): scalar field labelled
            by connected region
        extrema_points (pv.PolyData): the extrema, carrying "vtkOriginalPointIds"
        scalar_name (string): name of the variable
        sign (int): +1 for maxima, -1 for minima
        max_eps_km (float): DBSCAN neighbourhood radius in km
        min_samples (int): unused — DBSCAN runs with min_samples=1 so that every
            extremum lands in a cluster
        xi (float): unused — retained from the earlier OPTICS implementation
        penalty_length_scale_km (float): km of extra distance charged per unit of
            fractional descent along the path

    Returns:
        pv.PolyData: extrema points with a "Cluster ID" point array
    """
    num_points = extrema_points.n_points
    if num_points == 0:
        extrema_points.point_data["Cluster ID"] = np.zeros(0, dtype=int)
        return extrema_points

    # The geodesic graph is built from triangle edges, so the field has to be a
    # triangulated surface. Both callers already pass PolyData (clip_scalar ->
    # connectivity keeps it), for which the old vtkGeometryFilter step was a
    # pass-through; extract the surface only if that is not the case.
    surface = connectivity_clipped_scalar_field
    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface()
    surface = surface.triangulate()

    extrema_node = np.asarray(
        extrema_points.point_data["vtkOriginalPointIds"], dtype=np.intp
    )
    extrema_regions = np.asarray(extrema_points.point_data["RegionId"])
    point_region_id = np.asarray(surface.point_data["RegionId"])
    num_regions = int(np.max(point_region_id) + 1)

    point_scalar_values = np.asarray(surface.point_data[scalar_name], dtype=float)

    # Scalar value at each extremum, for the hill-climbing penalty.
    if scalar_name in extrema_points.point_data:
        extrema_scalar_values = np.asarray(
            extrema_points.point_data[scalar_name], dtype=float
        )
    else:
        # Fall back: look up via original point ID in the clipped scalar field
        extrema_scalar_values = point_scalar_values[extrema_node]

    dist_matrix = np.full((num_points, num_points), CLUSTER_MAX_DISTANCE)

    graph = _surface_graph(surface)
    geodesic, predecessors = dijkstra(
        graph, directed=False, indices=extrema_node, return_predecessors=True
    )

    for i in range(num_points):
        partners = np.flatnonzero(extrema_regions[i + 1 :] == extrema_regions[i]) + (
            i + 1
        )
        if partners.size == 0:
            continue

        # Extreme scalar value along the path from extremum i to every vertex:
        # the minimum for maxima, the maximum for minima.
        path_extremes = _path_extremes(predecessors[i], point_scalar_values, sign)

        for j in partners:
            dist = geodesic[i, extrema_node[j]]
            if not np.isfinite(dist):
                # Same region label, but no path across the triangulated
                # surface. Leave the pair at CLUSTER_MAX_DISTANCE.
                continue

            path_extreme_v = path_extremes[extrema_node[j]]
            if sign > 0:
                path_extreme_v = min(path_extreme_v, extrema_scalar_values[i])
            else:
                path_extreme_v = max(path_extreme_v, extrema_scalar_values[i])

            # Hill-climbing penalty: fractional descent from weaker endpoint.
            #
            # For maxima (sign>0): reference is the weaker (smaller) peak value.
            #   descent = reference - path_minimum. Positive when path dips below
            #   the weaker peak. Example: peaks at 30 and 25, path dips to 10.
            #   reference=25, descent=15, f=0.6.
            #
            # For minima (sign<0): reference is the weaker (least negative) trough.
            #   descent = path_maximum - reference. Positive when path rises above
            #   the weaker trough. Example: troughs at -20 and -18, path rises to -5.
            #   reference=-18, descent=(-5)-(-18)=13, f=13/18=0.72.
            val_i = extrema_scalar_values[i]
            val_j = extrema_scalar_values[j]

            if sign > 0:
                reference = min(val_i, val_j)
                descent = reference - path_extreme_v
            else:
                reference = max(val_i, val_j)
                descent = path_extreme_v - reference

            abs_ref = abs(reference)
            f = max(0.0, descent / abs_ref) if abs_ref > 0 else 0.0

            penalty_km = f * penalty_length_scale_km

            final_dist = dist * SCALE_FACTOR + penalty_km
            dist_matrix[i][j] = final_dist
            dist_matrix[j][i] = final_dist

    region_array = [[0 for _ in range(0)] for _ in range(num_regions)]
    cluster_assign = np.full(num_points, -1)

    for i in range(num_points):
        region_array[int(point_region_id[extrema_node[i]])].append(i)

    prev_cluster_id = 0

    for k in range(num_regions):
        num_cluster = len(region_array[k])
        if num_cluster == 0:
            continue

        if num_cluster == 1:
            cluster_assign[region_array[k][0]] = prev_cluster_id
            prev_cluster_id += 1
            continue

        new_dist = np.zeros((num_cluster, num_cluster))
        for i in range(num_cluster):
            for j in range(i + 1, num_cluster):
                new_dist[i][j] = dist_matrix[region_array[k][i]][region_array[k][j]]
                new_dist[j][i] = new_dist[i][j]

        dbscan = cluster.DBSCAN(
            eps=max_eps_km, min_samples=1, metric="precomputed",
        )
        labels = dbscan.fit_predict(new_dist)

        for i in range(num_cluster):
            if labels[i] != -1:
                cluster_assign[region_array[k][i]] = labels[i] + prev_cluster_id

        if np.max(labels) >= 0:
            prev_cluster_id += np.max(labels) + 1

    # Reassign any remaining unassigned points as singleton clusters.
    for i in range(num_points):
        if cluster_assign[i] == -1:
            cluster_assign[i] = prev_cluster_id
            prev_cluster_id += 1

    extrema_points.point_data["Cluster ID"] = cluster_assign.astype(int)
    return extrema_points


def identify_connected_regions(dataset):
    """Identify connected regions in the data

    Args:
        dataset (pv.PolyData): scalar field

    Returns:
        pv.PolyData: scalar field labeled by connected regions
    """

    return dataset.connectivity(largest=False)


def min_cluster_assign(min_points, scalar_name):
    """Get points in each minima cluster

    Args:
        min_points (pv.PolyData): clustered minima points in scalar field
        scalar_name (string): name of the variable
    """

    num_points_min = min_points.n_points
    if num_points_min == 0:
        return (np.array([]), np.array([]), defaultdict(list), 0)

    cluster_id_min = min_points["Cluster ID"]
    num_min_clusters = np.max(cluster_id_min) + 1

    min_pt_dict = defaultdict(list)
    cluster_min_arr = np.full(num_min_clusters, 0.0)
    cluster_min_point = np.full((num_min_clusters, 2), 0.0)
    min_scalars = min_points[scalar_name]

    cluster_lon_sum = np.zeros(num_min_clusters)
    cluster_lat_sum = np.zeros(num_min_clusters)
    cluster_weight_sum = np.zeros(num_min_clusters)
    cluster_base_lon = np.full(num_min_clusters, -1.0)

    for i in range(num_points_min):
        cid = cluster_id_min[i]
        lon = min_points["Longitude"][i]
        lat = min_points["Latitude"][i]
        val = min_scalars[i]
        weight = abs(val)

        min_pt_dict[cid].append([lon, lat])

        if cluster_min_arr[cid] > val:
            cluster_min_arr[cid] = val

        if cluster_base_lon[cid] == -1.0:
            cluster_base_lon[cid] = lon

        shifted_lon = lon
        if abs(lon - cluster_base_lon[cid]) > 180:
            if lon > cluster_base_lon[cid]:
                shifted_lon -= 360
            else:
                shifted_lon += 360

        cluster_lon_sum[cid] += shifted_lon * weight
        cluster_lat_sum[cid] += lat * weight
        cluster_weight_sum[cid] += weight

    for cid in range(num_min_clusters):
        if cluster_weight_sum[cid] > 0:
            avg_lon = cluster_lon_sum[cid] / cluster_weight_sum[cid]
            avg_lat = cluster_lat_sum[cid] / cluster_weight_sum[cid]
            cluster_min_point[cid][0] = avg_lon % 360
            cluster_min_point[cid][1] = avg_lat

    return (cluster_min_arr, cluster_min_point, min_pt_dict, num_min_clusters)


def max_cluster_assign(max_points, scalar_name):
    """Get points in each maxima cluster

    Args:
        max_points (pv.PolyData): clustered maxima points in scalar field
        scalar_name (string): name of the variable
    """

    num_points_max = max_points.n_points
    if num_points_max == 0:
        return (np.array([]), np.array([]), defaultdict(list), 0)

    cluster_id_max = max_points["Cluster ID"]
    num_max_clusters = np.max(cluster_id_max) + 1

    max_pt_dict = defaultdict(list)
    cluster_max_arr = np.full(num_max_clusters, 0.0)
    cluster_max_point = np.full((num_max_clusters, 2), 0.0)
    max_scalars = max_points[scalar_name]

    cluster_lon_sum = np.zeros(num_max_clusters)
    cluster_lat_sum = np.zeros(num_max_clusters)
    cluster_weight_sum = np.zeros(num_max_clusters)
    cluster_base_lon = np.full(num_max_clusters, -1.0)

    for i in range(num_points_max):
        cid = cluster_id_max[i]
        lon = max_points["Longitude"][i]
        lat = max_points["Latitude"][i]
        val = max_scalars[i]

        max_pt_dict[cid].append([lon, lat])

        if cluster_max_arr[cid] < val:
            cluster_max_arr[cid] = val

        if cluster_base_lon[cid] == -1.0:
            cluster_base_lon[cid] = lon

        # Shift longitude if it wraps around
        shifted_lon = lon
        if abs(lon - cluster_base_lon[cid]) > 180:
            if lon > cluster_base_lon[cid]:
                shifted_lon -= 360
            else:
                shifted_lon += 360

        cluster_lon_sum[cid] += shifted_lon * val
        cluster_lat_sum[cid] += lat * val
        cluster_weight_sum[cid] += val

    for cid in range(num_max_clusters):
        if cluster_weight_sum[cid] > 0:
            avg_lon = cluster_lon_sum[cid] / cluster_weight_sum[cid]
            avg_lat = cluster_lat_sum[cid] / cluster_weight_sum[cid]
            cluster_max_point[cid][0] = avg_lon % 360
            cluster_max_point[cid][1] = avg_lat

    return (cluster_max_arr, cluster_max_point, max_pt_dict, num_max_clusters)
