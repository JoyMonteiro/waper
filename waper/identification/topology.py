import math
import numpy as np
import pyvista as pv
import vtk
from sklearn import cluster
from collections import defaultdict

CLUSTER_MAX_DISTANCE = 150


def cluster_max(base_field, connectivity_clipped_scalar_field, max_points, scalar_name):
    """Cluster all the maxima in the scalar field

    Args:
        base_field (object): vtk object containing the unclipped scalar field data
        connectivity_clipped_scalar_field (object): vtk object containing connectivity information of the clipped scalar field
        max_points (object): vtk object containing all the maxima available in the field
        scalar_name (string): name of the variable

    Returns:
        object: list of maxima points with cluster IDs
    """
    # import scalar field and critical point data objects
    maxima_points = max_points
    # base_field = scalar_field
    scalar_field = connectivity_clipped_scalar_field


    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(scalar_field)
    geometry_filter.Update()
    scalar_field = geometry_filter.GetOutput()

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(scalar_field)
    triangle_filter.Update()
    scalar_field = triangle_filter.GetOutput()

    maxima_point_id = maxima_points.GetPointData().GetArray("vtkOriginalPointIds")
    num_points = maxima_points.GetNumberOfPoints()

    maxima_regions = maxima_points.GetPointData().GetArray("RegionId")

    point_region_id = scalar_field.GetPointData().GetArray("RegionId")
    num_regions = int(np.max(point_region_id) + 1)

    dist_matrix = np.full((num_points, num_points), CLUSTER_MAX_DISTANCE)

    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(scalar_field)

    # region_distance_array=[[[0 for col in range(0)]for row in range(0)]for clusters in range(num_regions)]

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(base_field)
    locator.BuildLocator()
    cell_ids = vtk.vtkIdList()

    cell_v = base_field.GetCellData().GetArray("{} Cell Value".format(scalar_name))

    point_coords = np.empty((0, 3))
    for i in range(num_points):
        point_coords = np.append(point_coords, [maxima_points.GetPoint(i)], axis=0)

    for i in range(num_points):
        for j in range(i + 1, num_points):
            min_v = 1000
            p0 = [0, 0, 0]
            p1 = [0, 0, 0]
            dist = 0.0
            region_1 = maxima_regions.GetTuple1(i)
            region_2 = maxima_regions.GetTuple1(j)
            if region_1 != region_2:
                continue
            dijkstra.SetStartVertex(int(maxima_point_id.GetTuple1(i)))
            dijkstra.SetEndVertex(int(maxima_point_id.GetTuple1(j)))
            dijkstra.Update()
            pts = dijkstra.GetOutput().GetPoints()
            for ptId in range(pts.GetNumberOfPoints() - 1):
                pts.GetPoint(ptId, p0)
                pts.GetPoint(ptId + 1, p1)
                dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
            
            locator.FindCellsAlongLine(point_coords[i], point_coords[j], 0.001, cell_ids)
            for k in range(cell_ids.GetNumberOfIds()):
                if cell_v.GetTuple1(cell_ids.GetId(k)) < min_v:
                    min_v = cell_v.GetTuple1(cell_ids.GetId(k))
                    
            dist_matrix[i][j] = dist_matrix[i][j] - min_v
            dist_matrix[j][i] = dist_matrix[i][j]

    region_array = [[0 for col in range(0)] for row in range(num_regions)]
    cluster_assign = np.full(num_points, 0)

    median_dist = -np.median(dist_matrix)
    
    for i in range(num_points):
        region_array[int(point_region_id.GetTuple1(int(maxima_point_id.GetTuple1(i))))].append(
            i
        )
        
    prev_max = 0

    for k in range(num_regions):
        if len(region_array[k]) == 1:
            cluster_assign[region_array[k][0]] = prev_max
            prev_max += 1
            continue
        if len(region_array[k]) == 2:
            cluster_assign[region_array[k][0]] = prev_max
            cluster_assign[region_array[k][1]] = prev_max
            prev_max += 1
            continue

        num_cluster = int(len(region_array[k]))
        new_dist = np.full((num_cluster, num_cluster), 0.0)

        for i in range(num_cluster):
            for j in range(i + 1, num_cluster):
                new_dist[i][j] = dist_matrix[region_array[k][i]][region_array[k][j]]
                new_dist[j][i] = new_dist[i][j]

        if num_cluster == 0:
            continue

        sim_matrix = np.negative(new_dist)
                
        af_clustering = cluster.AffinityPropagation(
            preference=np.full(num_cluster, median_dist / 5.0), affinity="precomputed"
        )
        af_clustering.fit(sim_matrix)
        clusters = af_clustering.labels_ + prev_max
        prev_max = np.max(clusters) + 1

        for i in range(num_cluster):
            cluster_assign[region_array[k][i]] = clusters[i]

    cluster_id = vtk.vtkIntArray()
    cluster_id.SetNumberOfComponents(1)
    cluster_id.SetNumberOfTuples(num_points)
    cluster_id.SetName("Cluster ID")

    for i in range(num_points):
        cluster_id.SetTuple1(i, cluster_assign[i])

    maxima_points.GetPointData().AddArray(cluster_id)
    return maxima_points


def cluster_min(base_field, connectivity_clipped_scalar_field, min_points, scalar_name):
    """Cluster all the minima in the scalar field

    Args:
        scalar_field (object): vtk object containing the scalar field data
        connectivity_clipped_scalar_field (object): vtk object containing connectivity information of the clipped scalar field
        min_points (object): vtk object containing all the minima available in the field

    Returns:
        object: list of minima points with cluster IDs
    """

    scalar_field = connectivity_clipped_scalar_field
    minima_points = min_points
    # base_field = scalar_field

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(scalar_field)
    geometry_filter.Update()
    scalar_field = geometry_filter.GetOutput()

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(scalar_field)
    triangle_filter.Update()
    scalar_field = triangle_filter.GetOutput()

    minima_point_id = minima_points.GetPointData().GetArray("vtkOriginalPointIds")
    num_points = minima_points.GetNumberOfPoints()

    minima_regions = minima_points.GetPointData().GetArray("RegionId")
    point_region_id = scalar_field.GetPointData().GetArray("RegionId")
    num_regions = int(np.max(point_region_id) + 1)

    dist_matrix = np.full((num_points, num_points), CLUSTER_MAX_DISTANCE)

    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(scalar_field)

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(base_field)
    locator.BuildLocator()
    cell_ids = vtk.vtkIdList()

    cell_v = base_field.GetCellData().GetArray("{} Cell Value".format(scalar_name))

    co_ords = np.empty((0, 3))
    for i in range(num_points):
        co_ords = np.append(co_ords, [minima_points.GetPoint(i)], axis=0)

    for i in range(num_points):
        for j in range(i + 1, num_points):
            max_v = -1000
            p0 = [0, 0, 0]
            p1 = [0, 0, 0]
            dist = 0.0
            region_1 = minima_regions.GetTuple1(i)
            region_2 = minima_regions.GetTuple1(j)
            if region_1 != region_2:
                continue

            dijkstra.SetStartVertex(int(minima_point_id.GetTuple1(i)))
            dijkstra.SetEndVertex(int(minima_point_id.GetTuple1(j)))
            dijkstra.Update()
            shortest_path_points = dijkstra.GetOutput().GetPoints()

            for point_id in range(shortest_path_points.GetNumberOfPoints() - 1):
                shortest_path_points.GetPoint(point_id, p0)
                shortest_path_points.GetPoint(point_id + 1, p1)
                dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))

            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
            locator.FindCellsAlongLine(co_ords[i], co_ords[j], 0.001, cell_ids)

            for k in range(cell_ids.GetNumberOfIds()):
                if cell_v.GetTuple1(cell_ids.GetId(k)) > max_v:
                    max_v = cell_v.GetTuple1(cell_ids.GetId(k))

            dist_matrix[i][j] = dist_matrix[i][j] + max_v
            dist_matrix[j][i] = dist_matrix[i][j]

    region_array = [[0 for col in range(0)] for row in range(num_regions)]
    cluster_assign = np.full(num_points, 0)

    median_dist = -np.median(dist_matrix)

    for i in range(num_points):
        region_array[int(point_region_id.GetTuple1(int(minima_point_id.GetTuple1(i))))].append(
            i
        )

    prev_min = 0

    for k in range(num_regions):
        if len(region_array[k]) == 1:
            cluster_assign[region_array[k][0]] = prev_min
            prev_min += 1
            continue
        if len(region_array[k]) == 2:
            cluster_assign[region_array[k][0]] = prev_min
            cluster_assign[region_array[k][1]] = prev_min
            prev_min += 1
            continue

        num_cluster = int(len(region_array[k]))
        new_dist = np.full((num_cluster, num_cluster), 0.0)

        for i in range(num_cluster):
            for j in range(i + 1, num_cluster):
                new_dist[i][j] = dist_matrix[region_array[k][i]][region_array[k][j]]
                new_dist[j][i] = new_dist[i][j]

        if num_cluster == 0:
            continue

        sim_matrix = np.negative(new_dist)

        af_clustering = cluster.AffinityPropagation(
            preference=np.full(num_cluster, median_dist / 5.0), affinity="precomputed"
        )
        af_clustering.fit(sim_matrix)
        clusters = af_clustering.labels_ + prev_min
        prev_min = np.max(clusters) + 1

        for i in range(num_cluster):
            cluster_assign[region_array[k][i]] = clusters[i]

    cluster_id = vtk.vtkIntArray()
    cluster_id.SetNumberOfComponents(1)
    cluster_id.SetNumberOfTuples(num_points)
    cluster_id.SetName("Cluster ID")

    for i in range(num_points):
        cluster_id.SetTuple1(i, cluster_assign[i])

    minima_points.GetPointData().AddArray(cluster_id)
    return minima_points


def identify_connected_regions(dataset):
    """Identify connected regions in the data

    Args:
        dataset (pv.PolyData): scalar field

    Returns:
        pv.PolyData: scalar field labeled by connected regions
    """

    return dataset.connectivity(largest=False)
    # connectivity_filter = vtk.vtkConnectivityFilter()
    # connectivity_filter.SetInputData(dataset)
    # connectivity_filter.SetExtractionModeToAllRegions()
    # connectivity_filter.ColorRegionsOn()
    # connectivity_filter.Update()
    # return connectivity_filter.GetOutput()


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
    cluster_id_min = min_points["Cluster ID"]
    num_min_clusters = np.max(cluster_id_min) + 1

    min_pt_dict = defaultdict(list)
    cluster_min_arr = np.full(num_min_clusters, 0.0)
    cluster_min_point = np.full((num_min_clusters, 2), 0.0)
    min_scalars = min_points[scalar_name]

    for i in range(num_points_min):
        x, y = min_points['Longitude'][i], min_points['Latitude'][i]
        coords = [x, y]
        min_pt_dict[cluster_id_min[i]].append(coords)

        # Identify the most negative point in the cluster
        if cluster_min_arr[cluster_id_min[i]] > min_scalars[i]:
            cluster_min_arr[cluster_id_min[i]] = min_scalars[i]
            cluster_min_point[cluster_id_min[i]][0] = min_points['Longitude'][i]
            cluster_min_point[cluster_id_min[i]][1] = min_points['Latitude'][i]

    # most negative point in each cluster, its coordinates,
    # dictionary with key = cluster ID and values = all points in cluster, total number of min clusters
    return (cluster_min_arr, cluster_min_point, min_pt_dict, num_min_clusters)


def max_cluster_assign(max_points, scalar_name):
    """Get points in each maxima cluster

    Args:
        max_points (pv.PolyData): clustered maxima points in scalar field
        scalar_name (string): name of the variable
    """

    num_points_max = max_points.n_points
    cluster_id_max = max_points["Cluster ID"]
    num_max_clusters = np.max(cluster_id_max) + 1

    max_pt_dict = defaultdict(list)
    cluster_max_arr = np.full(num_max_clusters, 0.0)
    cluster_max_point = np.full((num_max_clusters, 2), 0.0)
    max_scalars = max_points[scalar_name]

    #Identify largest point in each cluster
    for i in range(num_points_max):
        x, y = max_points['Longitude'][i], max_points['Latitude'][i]
        coords = [x, y]
        max_pt_dict[cluster_id_max[i]].append(coords)
        if cluster_max_arr[cluster_id_max[i]] < max_scalars[i]:
            cluster_max_arr[cluster_id_max[i]] = max_scalars[i]
            cluster_max_point[cluster_id_max[i]][0] = max_points['Longitude'][i]
            cluster_max_point[cluster_id_max[i]][1] = max_points['Latitude'][i]

    # largest point in each cluster, its coordinates,
    # dictionary with key = cluster ID and values = all points in cluster, total number of max clusters
    return (cluster_max_arr, cluster_max_point, max_pt_dict, num_max_clusters)
