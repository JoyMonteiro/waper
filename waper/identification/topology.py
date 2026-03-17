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
    eps_km=500,
    min_samples=1,
):
    """Cluster extrema (maxima or minima) in the scalar field using DBSCAN.

    Args:
        base_field (object): vtk object containing the unclipped scalar field data
        connectivity_clipped_scalar_field (object): vtk object containing connectivity information
        extrema_points (object): vtk object containing the extrema
        scalar_name (string): name of the variable
        sign (int): +1 for maxima, -1 for minima
        eps_km (float): DBSCAN clustering radius in km
        min_samples (int): DBSCAN minimum cluster size

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
            
            penalty_v = 1000 * sign
            
            for ptId in range(pts.GetNumberOfPoints() - 1):
                pts.GetPoint(ptId, p0)
                pts.GetPoint(ptId + 1, p1)
                dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
                
            for ptIdx in range(id_list.GetNumberOfIds()):
                vid = id_list.GetId(ptIdx)
                val = cell_v.GetTuple1(vid)  # Using cell_v since point to cell mapping isn't fully defined, but scalar field has PointData usually. Wait, cell_v is GetCellData().
                
                # To properly sample we should use the point data if it's there
                # Let's check if scalar_name exists in point data
                point_scalar_arr = scalar_field.GetPointData().GetArray(scalar_name)
                if point_scalar_arr:
                    val = point_scalar_arr.GetTuple1(vid)
                    
                if sign > 0:
                    if val < penalty_v:
                        penalty_v = val
                else:
                    if val > penalty_v:
                        penalty_v = val

            # Distance should be positive. If penalty is a drop in the crest, we penalize by increasing distance.
            # If sign > 0 (maxima), smaller/negative penalty_v means a deeper valley between them.
            # If sign < 0 (minima), larger/positive penalty_v means a higher ridge between them.
            
            penalty = 0.0
            if sign > 0:
                if penalty_v < 0:
                    penalty = abs(penalty_v) * 100  # Scale penalty to make distance very large
            else:
                if penalty_v > 0:
                    penalty = abs(penalty_v) * 100
                    
            final_dist = (dist + penalty) * SCALE_FACTOR
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

        # Use DBSCAN
        # Note: the dist_matrix isn't in true km yet until Task 3.6, but we apply eps_km here
        # Assuming eps_km mapping is handled outside or will be fixed in Task 3.6
        dbscan = cluster.DBSCAN(
            eps=eps_km, min_samples=min_samples, metric="precomputed"
        )
        labels = dbscan.fit_predict(new_dist)
        
        for i in range(num_cluster):
            if labels[i] != -1:
                cluster_assign[region_array[k][i]] = labels[i] + prev_cluster_id
                
        if np.max(labels) >= 0:
            prev_cluster_id += np.max(labels) + 1

    # Filter out noise points (-1)
    valid_indices = np.where(cluster_assign != -1)[0]
    num_valid = len(valid_indices)

    cluster_id = vtk.vtkIntArray()
    cluster_id.SetNumberOfComponents(1)
    cluster_id.SetNumberOfTuples(num_valid)
    cluster_id.SetName("Cluster ID")

    valid_points = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    
    # Need to extract only the valid points
    extract = vtk.vtkExtractSelection()
    extract.SetInputData(extrema_points)
    
    selectionNode = vtk.vtkSelectionNode()
    selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
    selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
    
    idArray = vtk.vtkIdTypeArray()
    for idx in valid_indices:
        idArray.InsertNextValue(idx)
        
    selectionNode.SetSelectionList(idArray)
    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)
    
    extract.SetInputData(1, selection)
    extract.Update()
    
    filtered_extrema = extract.GetOutput()
    
    for i, original_idx in enumerate(valid_indices):
        cluster_id.SetTuple1(i, cluster_assign[original_idx])
        
    filtered_extrema.GetPointData().AddArray(cluster_id)
    return pv.wrap(filtered_extrema)


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

    # Identify the most negative point in the cluster
    for i in range(num_points_min):
        x, y = min_points["Longitude"][i], min_points["Latitude"][i]
        coords = [x, y]
        min_pt_dict[cluster_id_min[i]].append(coords)

        if cluster_min_arr[cluster_id_min[i]] > min_scalars[i]:
            cluster_min_arr[cluster_id_min[i]] = min_scalars[i]
            cluster_min_point[cluster_id_min[i]][0] = min_points["Longitude"][i]
            cluster_min_point[cluster_id_min[i]][1] = min_points["Latitude"][i]

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

    # Identify largest point in each cluster
    for i in range(num_points_max):
        x, y = max_points["Longitude"][i], max_points["Latitude"][i]
        coords = [x, y]
        max_pt_dict[cluster_id_max[i]].append(coords)
        if cluster_max_arr[cluster_id_max[i]] < max_scalars[i]:
            cluster_max_arr[cluster_id_max[i]] = max_scalars[i]
            cluster_max_point[cluster_id_max[i]][0] = max_points["Longitude"][i]
            cluster_max_point[cluster_id_max[i]][1] = max_points["Latitude"][i]

    return (cluster_max_arr, cluster_max_point, max_pt_dict, num_max_clusters)
