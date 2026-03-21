import math
from collections import defaultdict

import numpy as np
import pyvista as pv
import vtk
from sklearn import cluster
from .utils import RADIUS_EARTH_KM, RADIUS_SPHERE

CLUSTER_MAX_DISTANCE = 15000.0
SCALE_FACTOR = RADIUS_EARTH_KM / RADIUS_SPHERE


def cluster_extrema(
    base_field,
    connectivity_clipped_scalar_field,
    extrema_points,
    scalar_name,
    sign,
    max_eps_km=1500,
    min_samples=2,
    xi=0.05,
    penalty_length_scale_km=2000.0,
):
    """Cluster extrema (maxima or minima) in the scalar field using OPTICS.

    Args:
        base_field (object): vtk object containing the unclipped scalar field data
        connectivity_clipped_scalar_field (object): vtk object containing connectivity information
        extrema_points (object): vtk object containing the extrema
        scalar_name (string): name of the variable
        sign (int): +1 for maxima, -1 for minima
        max_eps_km (float): OPTICS maximum neighborhood radius in km
        min_samples (int): OPTICS minimum cluster size
        xi (float): OPTICS steepness threshold for cluster boundary detection

    Returns:
        object: extrema points with cluster IDs (noise points discarded)
    """
    if extrema_points.GetNumberOfPoints() == 0:
        cluster_id = vtk.vtkIntArray()
        cluster_id.SetNumberOfComponents(1)
        cluster_id.SetNumberOfTuples(0)
        cluster_id.SetName("Cluster ID")
        extrema_points.GetPointData().AddArray(cluster_id)
        return pv.wrap(extrema_points)

    scalar_field = connectivity_clipped_scalar_field

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(scalar_field)
    geometry_filter.Update()
    scalar_field = geometry_filter.GetOutput()

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(scalar_field)
    triangle_filter.Update()
    scalar_field = triangle_filter.GetOutput()

    extrema_point_id = extrema_points.GetPointData().GetArray("vtkOriginalPointIds")
    num_points = extrema_points.GetNumberOfPoints()
    extrema_regions = extrema_points.GetPointData().GetArray("RegionId")
    point_region_id = scalar_field.GetPointData().GetArray("RegionId")
    num_regions = int(np.max(point_region_id) + 1)

    dist_matrix = np.full((num_points, num_points), CLUSTER_MAX_DISTANCE)

    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(scalar_field)

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(base_field)
    locator.BuildLocator()
    cell_ids = vtk.vtkIdList()

    cell_v = base_field.GetCellData().GetArray(f"{scalar_name} Cell Value")

    point_coords = np.empty((0, 3))
    for i in range(num_points):
        point_coords = np.append(point_coords, [extrema_points.GetPoint(i)], axis=0)

    # Extract scalar values at each extremum for hill-climbing penalty.
    extrema_scalar_values = np.zeros(num_points)
    # Try point data on extrema_points first
    extrema_scalar_arr = extrema_points.GetPointData().GetArray(scalar_name)
    if extrema_scalar_arr is not None:
        for i in range(num_points):
            extrema_scalar_values[i] = extrema_scalar_arr.GetTuple1(i)
    else:
        # Fall back: look up via original point ID in the clipped scalar field
        sf_scalar_arr = scalar_field.GetPointData().GetArray(scalar_name)
        for i in range(num_points):
            orig_id = int(extrema_point_id.GetTuple1(i))
            extrema_scalar_values[i] = sf_scalar_arr.GetTuple1(orig_id)

    for i in range(num_points):
        for j in range(i + 1, num_points):
            p0 = [0, 0, 0]
            p1 = [0, 0, 0]
            dist = 0.0
            
            region_1 = extrema_regions.GetTuple1(i)
            region_2 = extrema_regions.GetTuple1(j)
            if region_1 != region_2:
                continue

            dijkstra.SetStartVertex(int(extrema_point_id.GetTuple1(i)))
            dijkstra.SetEndVertex(int(extrema_point_id.GetTuple1(j)))
            dijkstra.Update()
            
            dijkstra_output = dijkstra.GetOutput()
            pts = dijkstra_output.GetPoints()
            id_list = dijkstra.GetIdList()
            
            # Track the extreme value along the Dijkstra path for hill-climbing penalty.
            # For maxima (sign>0): find minimum along path.
            # For minima (sign<0): find maximum along path.
            path_extreme_v = extrema_scalar_values[i]  # initialize to endpoint value
            
            for ptId in range(pts.GetNumberOfPoints() - 1):
                pts.GetPoint(ptId, p0)
                pts.GetPoint(ptId + 1, p1)
                dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
                
            # Look up point_scalar_arr once (moved out of inner loop — same result,
            # since the array doesn't change per iteration).
            point_scalar_arr = scalar_field.GetPointData().GetArray(scalar_name)
            for ptIdx in range(id_list.GetNumberOfIds()):
                vid = id_list.GetId(ptIdx)
                if point_scalar_arr:
                    val = point_scalar_arr.GetTuple1(vid)
                else:
                    val = cell_v.GetTuple1(vid)
                    
                if sign > 0:
                    if val < path_extreme_v:
                        path_extreme_v = val
                else:
                    if val > path_extreme_v:
                        path_extreme_v = val

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
            if abs_ref > 0:
                f = max(0.0, descent / abs_ref)
            else:
                f = 0.0
                
            penalty_km = f * penalty_length_scale_km
            
            final_dist = dist * SCALE_FACTOR + penalty_km
            dist_matrix[i][j] = final_dist
            dist_matrix[j][i] = final_dist

    region_array = [[0 for _ in range(0)] for _ in range(num_regions)]
    cluster_assign = np.full(num_points, -1)

    for i in range(num_points):
        region_array[
            int(point_region_id.GetTuple1(int(extrema_point_id.GetTuple1(i))))
        ].append(i)

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

    cluster_id = vtk.vtkIntArray()
    cluster_id.SetNumberOfComponents(1)
    cluster_id.SetNumberOfTuples(num_points)
    cluster_id.SetName("Cluster ID")

    for i in range(num_points):
        cluster_id.SetTuple1(i, cluster_assign[i])

    extrema_points.GetPointData().AddArray(cluster_id)
    return pv.wrap(extrema_points)


def identify_connected_regions(dataset):
    """Identify connected regions in the data

    Args:
        dataset (pv.PolyData): scalar field

    Returns:
        pv.PolyData: scalar field labeled by connected regions
    """

    return dataset.connectivity(largest=False)


def add_connectivity_data_min(dataset):
    """Identify connected regions in the data

    Args:
        dataset (vtk.UnstructuredGrid): scalar field

    Returns:
        vtk.UnstructuredGrid: scalar field labeled by connected regions
    """

    connectivity_filter = vtk.vtkConnectivityFilter()
    connectivity_filter.SetInputData(dataset)
    connectivity_filter.SetExtractionModeToAllRegions()
    connectivity_filter.ColorRegionsOn()
    connectivity_filter.Update()
    return connectivity_filter.GetOutput()


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
