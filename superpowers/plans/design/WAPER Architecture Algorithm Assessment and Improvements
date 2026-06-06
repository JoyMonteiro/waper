## User

please take a look at [@architecture_and_algorithm.md](file:///Users/joymonteiro/github/waper/conductor/architecture_and_algorithm.md), [@pandey.md](file:///Users/joymonteiro/github/waper/docs/pandey.md) [@malavika.md](file:///Users/joymonteiro/github/waper/docs/malavika.md)  and the code. I would like your assessment of the current architecture, how well it implements the algorithm, your comments on the improvements (use of spherical instead of cartesian geometry, visualization, etc.,), issues with the current code, and how it could be improved.

## Assistant



I'll start by exploring the codebase structure to understand the current implementation, then provide a comprehensive assessment.

**Tool Call: List the `waper` directory's contents**
Status: Completed

# Folders:
waper/waper/identification
waper/waper/interface
waper/waper/tracking

# Files:
waper/waper/__init__.py
waper/waper/py.typed


**Tool Call: List the `waper/waper/identification` directory's contents**
Status: Completed


# Files:
waper/waper/identification/__init__.py
waper/waper/identification/max_min.py
waper/waper/identification/rwp_graph.py
waper/waper/identification/topology.py
waper/waper/identification/utils.py


**Tool Call: List the `waper/waper/tracking` directory's contents**
Status: Completed


# Files:
waper/waper/tracking/__init__.py
waper/waper/tracking/quadtree.py
waper/waper/tracking/rwp_polygon.py
waper/waper/tracking/tracking_graph.py


**Tool Call: List the `waper/waper/interface` directory's contents**
Status: Completed


# Files:
waper/waper/interface/__init__.py
waper/waper/interface/api.py
waper/waper/interface/visualization.py


**Tool Call: Read identification module**
Status: Completed

Here are the complete contents of all four files:

---

## 1. `waper/waper/identification/max_min.py`

```waper/waper/identification/max_min.py#L1-264
import numpy as np
import vtk
import pyvista as pv
from .utils import get_vtk_object_from_data_array


def add_maxima_data(scalar_values, scalar_name, longitudes, latitudes):
    """Identify maxima in scalar field

    Args:
        scalar_values (DataArray): the scalar field
        scalar_name (string): name of the scalar
        longitudes (np.array): longitude coordinates
        latitudes (np.array): latitude coordinates

    Returns:
        pv.PolyData: vtk object containing the scalar data and maxima
    """
    r, c = scalar_values.shape
    check = np.zeros((r, c))
    is_max = np.zeros((r, c))
    vertex_identifiers = np.zeros(r * c)
    
    lons = np.linspace(0, 360, len(longitudes))
    lats = latitudes

    grid_vtk = get_vtk_object_from_data_array(scalar_values, lons, lats, scalar_name)
    
    numpy_data = scalar_values.values

    count = 0
    k = 0

    for i in range(r):
        for j in range(c):

            vertex_identifiers[k] = k + 1
            k += 1
            max_flag = 1

            if check[i][j] == 1:
                continue

            else:
                if j == 0:
                    for x in [i - 1, i, i + 1]:
                        for y in [c - 1, j, j + 1]:
                            if (0 <= x < r) and (0 <= y < c):
                                if numpy_data[x][y] > numpy_data[i][j]:
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

                if j == c - 1:
                    for x in [i - 1, i, i + 1]:
                        for y in [j - 1, j, 0]:
                            if (0 <= x < r) and (0 <= y < c):
                                if numpy_data[x][y] > numpy_data[i][j]:
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

                else:
                    for x in [i - 1, i, i + 1]:
                        for y in [j - 1, j, j + 1]:
                            if (0 <= x < r) and (0 <= y < c):
                                if numpy_data[x][y] > numpy_data[i][j]:
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

            if max_flag == 1:
                is_max[i][j] = 1
                check[i][j] = 1
                count += 1

    cell_number = grid_vtk.GetNumberOfCells()
    cell_id = np.arange(cell_number)

    grid_vtk.point_data["is max"] = is_max.ravel()
    grid_vtk.point_data["Vertex_id"] = vertex_identifiers
    grid_vtk.cell_data["{} Cell ID".format(scalar_name)] = cell_id

    return grid_vtk

def add_minima_data(scalar_values, scalar_name, longitudes, latitudes):
    """Identify minima in scalar field

    Args:
        scalar_values (DataArray): the scalar field
        scalar_name (string): name of the scalar
        longitudes (np.array): longitude coordinates
        latitudes (np.array): latitude coordinates

    Returns:
        pv.PolyData: vtk object containing the scalar data and minima
    """

    r, c = scalar_values.shape
    check = np.zeros((r, c))
    is_min = np.zeros((r, c))
    vertex_identifiers = np.zeros(r * c)
    
    lons = np.linspace(0, 360, len(longitudes))
    lats = latitudes

    grid_vtk = get_vtk_object_from_data_array(scalar_values, lons, lats, scalar_name)

    numpy_data = scalar_values.values
    
    count = 0
    k = 0

    for i in range(r):
        for j in range(c):

            vertex_identifiers[k] = k + 1
            k += 1
            min_flag = 1

            if check[i][j] == 1:
                continue

            else:
                if j == 0:
                    for x in [i - 1, i, i + 1]:
                        for y in [c - 1, j, j + 1]:
                            if (0 <= x < r) and (0 <= y < c):
                                if numpy_data[x][y] < numpy_data[i][j]:
                                    min_flag = 0
                                else:
                                    check[x][y] = 1

                if j == c - 1:
                    for x in [i - 1, i, i + 1]:
                        for y in [j - 1, j, 0]:
                            if (0 <= x < r) and (0 <= y < c):
                                if numpy_data[x][y] < numpy_data[i][j]:
                                    min_flag = 0
                                else:
                                    check[x][y] = 1

                else:
                    for x in [i - 1, i, i + 1]:
                        for y in [j - 1, j, j + 1]:
                            if (0 <= x < r) and (0 <= y < c):
                                if numpy_data[x][y] < numpy_data[i][j]:
                                    min_flag = 0
                                else:
                                    check[x][y] = 1

                if min_flag == 1 and i != 0:
                    is_min[i][j] = 1
                    check[i][j] = 1
                    count += 1

    cell_number = grid_vtk.GetNumberOfCells()
    cell_id = np.arange(cell_number)


    grid_vtk.point_data["is min"] = is_min.ravel()
    grid_vtk.point_data["Vertex_id"] = vertex_identifiers
    grid_vtk.cell_data["{} Cell ID".format(scalar_name)] = cell_id
    # print("min points", count)
    return grid_vtk


def extract_maxima_points(scalar_field, threshold, scalar_name):
    """Get data corresponding to identified maxima

    Args:
        scalar_field (vtk.vtkUnstructuredGrid): vtk object containing clipped dataset
        threshold (float): discard maxima below threshold
        scalar_name (string): name of variable in scalar_field

    Returns:
        vtk.vtkUnstructuredGrid: array containing identified maxima
    """
        
    return scalar_field.extract_points(
        ((scalar_field.point_data['is max'] == 1)
         & (scalar_field.point_data[scalar_name] > threshold)),
        include_cells=False)
    
    
def extract_minima_points(scalar_field, threshold, scalar_name):
    """Get data corresponding to identified minima

    Args:
        scalar_field (vtk.vtkUnstructuredGrid): vtk object containing clipped dataset
        threshold (float): discard minima above threshold
        scalar_name (string): name of variable in scalar_field

    Returns:
        vtk.vtkUnstructuredGrid: array containing identified minima
    """
        
    return scalar_field.extract_points(
        ((scalar_field.point_data['is min'] == 1)
         & (scalar_field.point_data[scalar_name] < threshold)),
        include_cells=False)


def interpolate_cell_values(dataset, scalar_name):
    """Interpolate point data to cells

    Args:
        dataset (vtk.RectilinearGrid): vtk object containing point data
        scalar_name (string): name of variable being interpolated

    Returns:
        vtk.RectilinearGrid: input vtk object with cell data added
    """

    num_cells = dataset.GetNumberOfCells()
    scalar_v = dataset.GetPointData().GetArray(scalar_name)
    cell_scalars = vtk.vtkFloatArray()
    cell_scalars.SetNumberOfComponents(1)
    cell_scalars.SetNumberOfTuples(num_cells)
    cell_scalars.SetName("{} Cell Value".format(scalar_name))

    for i in range(num_cells):
        cell = dataset.GetCell(i)
        num_points = cell.GetNumberOfPoints()
        func_value = 0
        for j in range(num_points):
            pid = cell.GetPointId(j)
            func_value += scalar_v.GetTuple1(pid)
        func_value /= num_points
        cell_scalars.SetTuple1(i, func_value)

    dataset.GetCellData().AddArray(cell_scalars)
    return dataset


# CAN BE REMOVED?
def interpolate_cell_values_min(inputs, scalar_name):
    """Interpolate point data to cells

    Args:
        inputs (vtk.RectilinearGrid): vtk object containing point data
        scalar_name (string): name of variable being interpolated

    Returns:
        vtk.RectilinearGrid: input vtk object with cell data added
    """
    
    num_cells = inputs.GetNumberOfCells()
    scalar_v = inputs.GetPointData().GetArray(scalar_name)
    cell_scalars = vtk.vtkFloatArray()
    cell_scalars.SetNumberOfComponents(1)
    cell_scalars.SetNumberOfTuples(num_cells)
    cell_scalars.SetName("{} Cell Value".format(scalar_name))

    for i in range(num_cells):
        cell = inputs.GetCell(i)
        num_points = cell.GetNumberOfPoints()
        func_value = 0
        for j in range(num_points):
            pid = cell.GetPointId(j)
            func_value += scalar_v.GetTuple1(pid)
        func_value /= num_points
        cell_scalars.SetTuple1(i, func_value)

    inputs.GetCellData().AddArray(cell_scalars)
    return inputs

def clip_dataset(dataset, scalar_name, threshold, invert=False):
    """clip scalar field to eliminate values below threshold

    Args:
        dataset (pv.PolyData): pyvista object containing scalar field
        scalar_name (string): name of the scalar in the vtk object
        threshold (float): threshold to clip at
        invert (boolean): if False retain values above threshold, else below

    Returns:
        pv.PolyData: pv object containing the data
    """
    
    return dataset.clip_scalar(scalars=scalar_name, invert=invert, value=threshold)


# def clip_dataset_min(dataset, scalar_name, threshold):
#     """clip scalar field to eliminate values above threshold

#     Args:
#         dataset (vtk.RectilinearGrid): vtk object containing scalar field
#         scalar_name (string): name of the scalar in the vtk object
#         threshold (float): threshold to clip at

#     Returns:
#         vtk.UnstructuredGrid: vtk object containing the data
#     """

#     clip_dataset = vtk.vtkClipDataSet()
#     dataset.GetPointData().SetScalars(dataset.GetPointData().GetArray(scalar_name))
#     clip_dataset.SetValue(threshold)
#     clip_dataset.SetInputData(dataset)
#     clip_dataset.InsideOutOn()
#     clip_dataset.Update()
#     return clip_dataset.GetOutput()

# def extract_position_ids_minima(scalar_field, threshold, scalar_name):
#     """extract position IDs of identified minima

#     Args:
#         scalar_field (pv.PolyData): vtk object with clipped dataset
#         threshold (float): discard minima above threshold
#         scalar_name (string): name of variable in scalar_field

#     Returns:
#         vtk.vtkIdTypeArray: list of position IDs
#     """

#     pos_min_ids = vtk.vtkIdTypeArray()
#     num_pts = scalar_field.GetNumberOfPoints()
#     is_min_arr = scalar_field.GetPointData().GetArray("is min")
#     scalar_arr = scalar_field.GetPointData().GetArray(scalar_name)
    
#     for i in range(num_pts):
#         if is_min_arr.GetTuple1(i) == 1 and scalar_arr.GetTuple1(i) <= threshold:
#             pos_min_ids.InsertNextValue(i)
#     return pos_min_ids


# def extract_position_ids_maxima(scalar_field, threshold, scalar_name):
#     """extract position IDs of identified maxima

#     Args:
#         scalar_field (vtk): vtk object with clipped dataset
#         threshold (float): discard minima below threshold
#         scalar_name (string): name of variable in scalar_field

#     Returns:
#         vtk.vtkIdTypeArray: list of position IDs
#     """
    
#     pos_max_ids = vtk.vtkIdTypeArray()
#     num_pts = scalar_field.GetNumberOfPoints()
#     is_max_arr = scalar_field.GetPointData().GetArray("is max")
#     scalar_arr = scalar_field.GetPointData().GetArray(scalar_name)
    
#     for i in range(num_pts):
#         if is_max_arr.GetTuple1(i) == 1 and scalar_arr.GetTuple1(i) >= threshold:
#             pos_max_ids.InsertNextValue(i)
            
#     print(pos_max_ids)
#     return pos_max_ids

# def extract_selection_ids_maxima(scalar_field, id_list):
#     """Get data corresponding to identified maxima

#     Args:
#         scalar_field (vtk.vtkUnstructuredGrid): vtk object containing clipped dataset
#         id_list (list): list of ids selected

#     Returns:
#         vtk.vtkUnstructuredGrid: array containing identified maxima
#     """
    
#     print('here 4.11')
    
#     selection_node = vtk.vtkSelectionNode()
#     selection_node.SetFieldType(1)
#     selection_node.SetContentType(4)
#     selection_node.SetSelectionList(id_list)
#     selection = vtk.vtkSelection()
#     selection.AddNode(selection_node)
    
#     print('here 4.12')
    
#     extract_selection = vtk.vtkExtractSelection()
    

#     extract_selection.SetInputData(0, scalar_field)
#     print('here 4.13')
#     extract_selection.SetInputData(1, selection)
#     print('here 4.14')
#     extract_selection.Update()
#     print('here 4.15')
    

#     return pv.wrap(extract_selection.GetOutput())

# def extract_selection_ids_minima(scalar_field, id_list):
#     """Get data corresponding to identified minima

#     Args:
#         scalar_field (vtk.vtkUnstructuredGrid): vtk object containing clipped dataset
#         id_list (list): list of ids selected

#     Returns:
#         vtk.vtkUnstructuredGrid: array containing identified minima
#     """
    
#     selection_node=vtk.vtkSelectionNode()
#     selection_node.SetFieldType(1)
#     selection_node.SetContentType(4)
#     selection_node.SetSelectionList(id_list)
#     selection=vtk.vtkSelection()
#     selection.AddNode(selection_node)
    
#     extract_selection=vtk.vtkExtractSelection()
#     extract_selection.SetInputData(0,scalar_field)
#     extract_selection.SetInputData(1,selection)
#     extract_selection.Update()
#     return pv.wrap(extract_selection.GetOutput())
```

---

## 2. `waper/waper/identification/topology.py`

```waper/waper/identification/topology.py#L1-304
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
```

---

## 3. `waper/waper/identification/rwp_graph.py`

```waper/waper/identification/rwp_graph.py#L1-331
import networkx as nx
import numpy as np
from collections import defaultdict
from .utils import haversine_distance, is_to_the_east

WAPER_MAX_SCALAR_VALUE = 100
WAPER_MAX_NODE_DISTANCE = 1000

WAPER_MIN_LON_DELTA = 6


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
        
        lon_0 = assoc_graph.nodes[start_node]["coords"][0]
        lon_1 = assoc_graph.nodes[end_node]["coords"][0]
        
        if abs(lon_0 - lon_1) <= WAPER_MIN_LON_DELTA:
            continue
        
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
                # best_path = None
                # max_weight = 0
                for path in nx.all_simple_paths(assoc_graph, source=source, target=sink):
                    # consistent = True
                    # for node in path[:-1]:
                    #     if is_to_the_east(
                    #         assoc_graph.nodes[node]["coords"][0], assoc_graph.nodes[path[-1]]["coords"][0]
                    #     ):
                    #         consistent = False
                    
                #     path_weight = get_path_weight(assoc_graph, path)
                    
                #     if path_weight > max_weight:
                #         max_weight = path_weight
                #         best_path = path
                  
                # if isinstance(best_path, list):
                #     # print(source, sink, best_path)
                #     path_list.append(best_path)
                
                    path_list.append(path)
                
    # return path_list

    path_wt_dict = {}

    # # print(len(path_list), "number of paths found")

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
```

---

## 4. `waper/waper/identification/utils.py`

```waper/waper/identification/utils.py#L1-76
import geovista as gv
import vtk
import numpy as np
import math

RADIUS_SPHERE = 63.71
RADIUS_EARTH = 6.371e6

def get_point_data_label(scalar_name):
    return scalar_name

def get_cell_data_label(scalar_name):
    return "Cell Value {}".format(scalar_name)

def get_vtk_object_from_data_array(data_array, lons, lats, array_name="v"):
    """Get vtk object from xarray dataArray

    Args:
        longitude (array): coordinates along zonal direction
        latitude (array): coordinates along meridional direction
        scalar_values (array): scalar field to convert to vtk object
    """

    grid = gv.Transform.from_1d(
        lons, lats, 
        data=data_array.data, name=array_name, radius=RADIUS_SPHERE, clean=False)
    
    mesh_lons, mesh_lats = np.meshgrid(lons, lats, indexing='xy')
    
    grid.cell_data['{} Cell Value'.format(array_name)] = grid.point_data_to_cell_data()[array_name]
    
    grid.point_data['Longitude'] = mesh_lons.ravel()
    grid.point_data['Latitude'] = mesh_lats.ravel()

    return grid

def get_iso_contour(scalar_field, value, scalar_name):
    
    contour_filter = vtk.vtkContourFilter()
    scalar_field.GetPointData().SetScalars(scalar_field.GetPointData().GetArray(scalar_name))
    contour_filter.SetValue(0, value)
    contour_filter.SetInputData(scalar_field)
    contour_filter.Update()
    return contour_filter.GetOutput()


def compute_gradients(scalar_field, scalar_name):
    
    gradient_filter = vtk.vtkGradientFilter()
    scalar_field.GetPointData().SetScalars(scalar_field.GetPointData().GetArray(scalar_name))
    gradient_filter.SetInputData(scalar_field)
    gradient_filter.Update()
    return gradient_filter.GetOutput()

def haversine_distance(lat1, lon1, lat2, lon2):

    # circum = 2*np.pi*RADIUS_EARTH*np.cos(np.radians(30))

    dlat = math.radians(lat1 - lat2)

    dlon = math.radians(lon1 - lon2)

    a = (math.sin(dlat/2))**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * (math.sin(dlon/2))**2
    c = 2 * np.arctan2(math.sqrt(a), math.sqrt(1-a))
    distance = RADIUS_EARTH * c

    return distance

def is_to_the_east(lon1, lon2):
    
    delta_lat = lon1 - lon2
    
    if abs(delta_lat) > 180:
        delta_lat = -delta_lat
    
    if delta_lat > 0:
        return True
```

---

There you go — the complete contents of all four files. Here's a quick summary of each:

- **`max_min.py`** — Identifies maxima and minima in scalar fields on a lat/lon grid, extracts them as VTK point sets, interpolates cell values, and clips datasets by threshold. It handles periodic boundary conditions at the longitude edges (`j == 0` and `j == c-1`).

- **`topology.py`** — Clusters maxima and minima using Dijkstra shortest paths on VTK meshes combined with Affinity Propagation clustering (from scikit-learn). Also identifies connected regions and assigns cluster memberships.

- **`rwp_graph.py`** — Builds an association graph (using NetworkX) by linking maxima and minima clusters through isocontour proximity. Prunes the graph by scalar thresholds and edge weights (based on haversine distance), then ranks paths through the graph.

- **`utils.py`** — Utility functions including: converting xarray DataArrays to VTK objects via `geovista`, computing isocontours, computing gradients, haversine distance calculation, and an `is_to_the_east` directional check.

**Tool Call: Read tracking module**
Status: Completed

Here are the complete contents of all three files:

---

## 1. `waper/waper/tracking/quadtree.py`

```waper/waper/tracking/quadtree.py#L1-244
from operator import add
from functools import reduce
import networkx as nx
import math
import numpy as np
from collections import defaultdict

from .rwp_polygon import WAPER_NUM_PIXELS, WAPER_IMAGE_SIZE

# function to split input raster image into 4 equal images
# returns the 4 split images


def split4(raster):
    # half_split = np.array_split(raster, 2)
    # res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    # return np.array(reduce(add, res))
    r, c = raster.shape

    half_r = int(r / 2)
    half_c = int(c / 2)

    return (
        raster[:half_r, :half_c],
        raster[:half_r, half_c:],
        raster[half_r:, :half_c],
        raster[half_r:, half_c:],
    )


# function to compute mean pixel value of an image
# returns mean value for each image


def calculate_mean(raster):
    return np.mean(raster)


# function to return all the features (in terms of pixel value assigned to each feature) present in an image
def get_features(raster):
    features = set(raster.ravel())
    # features.add(0)
    return tuple(features)


# function to create a quadtree corresponding to a raster image. Initial node i=0 is created beforehand and passed on as an argument
# i represents the position of a node in the raster image. Between 2 quadtrees, nodes with the same i value represents the same position within their respective raster images
# level represents the height at which a particular node is located in a quadtree. It serves as a pointer to the dimensions of a node (how many pixels it contains)
# the function returns the constructed quadtree Q; nodes are given by their i values and each node consists of the mean of the pixel values, features it contains and the height/level at which the node is located


def create_quadtree(raster):
    quadtree = nx.DiGraph()

    quadtree.add_node(
        0, mean=np.mean(raster), features=get_features(raster), level=0, start_pixel=(0, 0)
    )

    return insert_node(quadtree, 0, raster, 0)


def insert_node(Q, parent_node_id, raster, level):

    r, c = np.array(raster).shape
    if r > 1 and c > 1:
        level = level + 1
        split_raster = split4(raster)

        parent_start_x, parent_start_y = Q.nodes[parent_node_id]["start_pixel"]

        m_1 = calculate_mean(split_raster[0])
        f_1 = get_features(split_raster[0])
        Q.add_node(
            (4 * parent_node_id) + 1,
            mean=m_1,
            features=f_1,
            level=level,
            start_pixel=(parent_start_x, parent_start_y),
        )
        Q.add_edge(parent_node_id, (4 * parent_node_id) + 1)

        m_2 = calculate_mean(split_raster[1])
        f_2 = get_features(split_raster[1])
        Q.add_node(
            (4 * parent_node_id) + 2,
            mean=m_2,
            features=f_2,
            level=level,
            start_pixel=(parent_start_x, parent_start_y + int(r / 2)),
        )
        Q.add_edge(parent_node_id, (4 * parent_node_id) + 2)

        m_3 = calculate_mean(split_raster[2])
        f_3 = get_features(split_raster[2])
        Q.add_node(
            (4 * parent_node_id) + 3,
            mean=m_3,
            features=f_3,
            level=level,
            start_pixel=(parent_start_x + int(r / 2), parent_start_y),
        )
        Q.add_edge(parent_node_id, (4 * parent_node_id) + 3)

        m_4 = calculate_mean(split_raster[3])
        f_4 = get_features(split_raster[3])
        Q.add_node(
            (4 * parent_node_id) + 4,
            mean=m_4,
            features=f_4,
            level=level,
            start_pixel=(parent_start_x + int(r / 2), parent_start_y + int(r / 2)),
        )
        Q.add_edge(parent_node_id, (4 * parent_node_id) + 4)

        if len(f_1) > 1:
            Q = insert_node(Q, (4 * parent_node_id) + 1, split_raster[0], level)

        if len(f_2) > 1:
            Q = insert_node(Q, (4 * parent_node_id) + 2, split_raster[1], level)

        if len(f_3) > 1:
            Q = insert_node(Q, (4 * parent_node_id) + 3, split_raster[2], level)

        if len(f_4) > 1:
            Q = insert_node(Q, (4 * parent_node_id) + 4, split_raster[3], level)

    return Q


# function to compute the number of pixels corresponding to each feature in a particular quadtree
# returns a dictionary "pixel_dict" whose - key:feature values, values:number of pixels


def compute_pixels(quadtree):

    pixel_dict = defaultdict(list)
    leaf_nodes = [
        node
        for node in quadtree.nodes()
        if quadtree.in_degree(node) != 0 and quadtree.out_degree(node) == 0
    ]
    for i in range(len(leaf_nodes)):
        f = quadtree.nodes[leaf_nodes[i]]["features"]
        f = tuple(f)
        if 0 in f:
            continue
        if f in pixel_dict:
            pixel_dict[f] += WAPER_NUM_PIXELS / (4 ** (quadtree.nodes[leaf_nodes[i]]["level"]))
        else:
            pixel_dict[f] = WAPER_NUM_PIXELS / (4 ** (quadtree.nodes[leaf_nodes[i]]["level"]))
    return pixel_dict


def contains_no_features(node):
    if len(node["features"]) == 1:
        if node["features"][0] == 0:
            return True

    return False


def contains_more_than_one_feature(node):
    return len(node["features"]) > 1


def contains_one_feature(node):
    if len(node["features"]) == 1:
        if node["features"][0] != 0:
            return True

    return False


# function to construct a certain branch of the merge quadtree
# Used when in a particular location, the feature in one quadtree is bigger in size compared to the feature in the second quadtree
# G represents quadtree who has the smaller feature; i is the leaf node(feature node) in the other quadtree
# Returns merged quadtree Q with the branch rooted at i same as that of the input quadtree G


def construct(merged_quadtree, test_quadtree, node_number, larger_feature):

    for j in range(1, 5):
        if contains_no_features(test_quadtree.nodes[(4 * node_number) + j]):
            merged_quadtree.add_node(
                (4 * node_number) + j,
                features=[0],
                level=test_quadtree.nodes[(4 * node_number) + j]["level"],
                start_pixel=test_quadtree.nodes[(4 * node_number) + j]["start_pixel"],
            )
        else:
            merged_quadtree.add_node(
                (4 * node_number) + j,
                features=np.sort(
                    np.concatenate(
                        [
                            test_quadtree.nodes[(4 * node_number) + j]["features"],
                            larger_feature,
                        ]
                    )
                ),
                level=test_quadtree.nodes[(4 * node_number) + j]["level"],
                start_pixel=test_quadtree.nodes[(4 * node_number) + j]["start_pixel"],
            )
        merged_quadtree.add_edge(node_number, (4 * node_number) + j)

    for j in range(1, 5):
        if len(test_quadtree.nodes[(4 * node_number) + j]["features"]) > 1:
            merged_quadtree = construct(
                merged_quadtree, test_quadtree, (4 * node_number) + j, larger_feature
            )
    return merged_quadtree


# function to merge two quadtrees G and H; Q represents the merged quadtree
# returns the merged quadtree Q
def merge(curr_time_quadtree, prev_time_quadtree):

    merged_quadtree = nx.DiGraph()
    common_nodes = set(curr_time_quadtree).intersection(prev_time_quadtree)
    for node_number in common_nodes:
        if node_number not in list(merged_quadtree):
            if contains_no_features(
                prev_time_quadtree.nodes[node_number]
            ) or contains_no_features(curr_time_quadtree.nodes[node_number]):
                merged_quadtree.add_node(
                    node_number,
                    features=[0],
                    level=prev_time_quadtree.nodes[node_number]["level"],
                    start_pixel=prev_time_quadtree.nodes[node_number]["start_pixel"],
                )
                if math.ceil((node_number / 4) - 1) >= 0:
                    merged_quadtree.add_edge(math.ceil((node_number / 4) - 1), node_number)

            elif contains_more_than_one_feature(
                prev_time_quadtree.nodes[node_number]
            ) and contains_one_feature(curr_time_quadtree.nodes[node_number]):
                merged_quadtree = construct(
                    merged_quadtree,
                    prev_time_quadtree,
                    node_number,
                    curr_time_quadtree.nodes[node_number]["features"],
                )

            elif contains_more_than_one_feature(
                curr_time_quadtree.nodes[node_number]
            ) and contains_one_feature(prev_time_quadtree.nodes[node_number]):
                merged_quadtree = construct(
                    merged_quadtree,
                    curr_time_quadtree,
                    node_number,
                    prev_time_quadtree.nodes[node_number]["features"],
                )

            else:
                features = np.concatenate(
                    [
                        prev_time_quadtree.nodes[node_number]["features"],
                        curr_time_quadtree.nodes[node_number]["features"],
                    ]
                )
                merged_quadtree.add_node(
                    node_number,
                    features=np.sort(features),
                    level=prev_time_quadtree.nodes[node_number]["level"],
                    start_pixel=prev_time_quadtree.nodes[node_number]["start_pixel"],
                )
                if math.ceil((node_number / 4) - 1) >= 0:
                    merged_quadtree.add_edge(math.ceil((node_number / 4) - 1), node_number)

    return merged_quadtree


# function to compute the number of pixels corresponding to each feature in a particular quadtree
# returns a dictionary "pixel_dict" whose - key:feature values, values:number of pixels


def compute_size_features(quadtree):

    pixel_dict = defaultdict(float)
    leaf_nodes = [
        node
        for node in quadtree.nodes()
        if quadtree.in_degree(node) != 0 and quadtree.out_degree(node) == 0
    ]
    for i in range(len(leaf_nodes)):
        f = quadtree.nodes[leaf_nodes[i]]["features"]
        f = tuple(f)
        if 0 in f:
            continue
        if f in pixel_dict:
            pixel_dict[f] += WAPER_NUM_PIXELS / (4 ** (quadtree.nodes[leaf_nodes[i]]["level"]))
        else:
            pixel_dict[f] = WAPER_NUM_PIXELS / (4 ** (quadtree.nodes[leaf_nodes[i]]["level"]))
    return pixel_dict


def reconstruct_image(quadtree):

    image = np.zeros((WAPER_IMAGE_SIZE, WAPER_IMAGE_SIZE))

    leaf_nodes = [
        node
        for node in quadtree.nodes()
        if quadtree.in_degree(node) != 0 and quadtree.out_degree(node) == 0
    ]

    for node in leaf_nodes:
        feature = quadtree.nodes[node]["features"][0]
        level = quadtree.nodes[node]["level"]
        x_pixel, y_pixel = quadtree.nodes[node]["start_pixel"]
        image[
            x_pixel : x_pixel + int(WAPER_IMAGE_SIZE / (2**level)),
            y_pixel : y_pixel + int(WAPER_IMAGE_SIZE / (2**level)),
        ] = feature

    return image
```

---

## 2. `waper/waper/tracking/rwp_polygon.py`

```waper/waper/tracking/rwp_polygon.py#L1-201
from ..identification import topology
import numpy as np
from pyproj.transformer import Transformer
import pyproj

from shapely.geometry import MultiPoint
from rasterio import features, Affine

WAPER_SUBSAMPLE = 5
WAPER_IMAGE_SIZE = 512
WAPER_CLUSTER_WIDTH = 60
WAPER_NUM_PIXELS = WAPER_IMAGE_SIZE*WAPER_IMAGE_SIZE

WAPER_X_BOUNDS = (12712833.087371958, -12712833.087371958)
WAPER_Y_BOUNDS = (12710532.145483922, -12713600.098850505)

WAPER_X_RES = (WAPER_X_BOUNDS[0] - WAPER_X_BOUNDS[1]) / WAPER_IMAGE_SIZE
WAPER_Y_RES = (WAPER_Y_BOUNDS[0] - WAPER_Y_BOUNDS[1]) / WAPER_IMAGE_SIZE

WAPER_RASTER_TRANSFORM = Affine.translation(
    WAPER_X_BOUNDS[1] - WAPER_X_RES / 2, WAPER_Y_BOUNDS[1] - WAPER_Y_RES / 2
) * Affine.scale(WAPER_X_RES, WAPER_Y_RES)

# TODO this must handle both north and south poles
def transform_to_stereographic(input_xs, input_ys, inverse=False):

    from_crs = pyproj.crs.CRS(4326)  # standard lat-lon
    to_crs = pyproj.crs.CRS("+proj=stere +lat_0=90 +lon_0=0")  # north pole stereographic
    if inverse:
        transformer = Transformer.from_crs(to_crs, from_crs, always_xy="True")
    else:
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy="True")

    try:
        return transformer.transform(input_xs, input_ys, errcheck=True)
    except:
        print(input_xs, input_ys)
        raise ValueError()


def get_consistent_longitudes(longitude_array, min_lon):
    """fix issue with wrap around of longitudes

    Args:
        longitude_array (list): list of longitudes
    """

    final_array = np.array(longitude_array)
    # print(np.max(longitude_array), np.min(longitude_array))
    if (np.max(longitude_array) - np.min(longitude_array)) > WAPER_CLUSTER_WIDTH:
        # print("Inconsistent, fixing")
        for i in range(len(final_array)):
            # print(final_array[i])
            if final_array[i] < min_lon:
                # print("*" * 10)
                final_array[i] += 360

        # final_array[np.where(final_array < min_lon)] += 360

    return list(final_array)


def get_region_points_and_values(
    assoc_graph, node, clipped_region, clip_threshold, scalar_name
):
    """Get all points in a region corresponding to a node in the association graph

    Args:
        assoc_graph (nx.Graph): Association Graph
        node (nx.Node): Node in the above graph
        clipped_region (pv.PolyData): scalar which includes connectivity information
        clip_threshold (float): Threshold at which scalar data is thresholded
        scalar_name (str): name of the scalar quantity

    Returns:
        tuple: coordinates of points close to node in graph
    """

    if abs(assoc_graph.nodes[node]["scalar"]) < clip_threshold:
        return None

    closest_point = clipped_region.find_closest_point(
        assoc_graph.nodes[node]["spherical_coords"]
    )
    region_id_node = clipped_region.point_data["RegionId"][closest_point]

    lons = clipped_region["Longitude"][clipped_region.point_data["RegionId"] == region_id_node]
    lats = clipped_region["Latitude"][clipped_region.point_data["RegionId"] == region_id_node]
    values = clipped_region.point_data[scalar_name][
        clipped_region.point_data["RegionId"] == region_id_node
    ]
    
    # node_latitude = assoc_graph.nodes[node]["coords"][1]
    
    # #TODO 3 should be a paramter
    # valid_region = np.logical_and(lats >= node_latitude-3, lats <= node_latitude+3)
    # lons = lons[valid_region]
    # lats = lats[valid_region]
    # values = values[valid_region]

    return lons, lats, values


def get_polygon_for_rwp_path(path, assoc_graph, scalar_data, scalar_name, min_latitude, max_latitude):
    """Get bounding polygon for an identified RWP

    Args:
        path (list): list of nodes in each path
        assoc_graph (nx.Graph): association graph
        scalar_data (pv.PolyData): scalar field

    Returns:
        tuple: convex hull of points and polygon ID
    """

    path_max = -100
    for node in path:
        max_value = abs(assoc_graph.nodes[node]["scalar"])

        if max_value > path_max:
            path_max = max_value

    clip_threshold = path_max / 3.0

    max_clipped_region = topology.identify_connected_regions(
        scalar_data.clip_scalar(
            scalars=scalar_name, value=clip_threshold, invert=False
        ).clean()
    )

    min_clipped_region = topology.identify_connected_regions(
        scalar_data.clip_scalar(
            scalars=scalar_name, value=-clip_threshold, invert=True
        ).clean()
    )

    list_rwp_points = []
    list_lons = []
    list_lats = []
    list_values = []

    # min_lon = 360
    for node in path:
        if node > 0:
            out = get_region_points_and_values(
                assoc_graph, node, max_clipped_region, clip_threshold, scalar_name
            )
            if out:
                lons, lats, values = out
                
                valid_region = np.logical_and(lats >= min_latitude, lats <= max_latitude)
                lons = lons[valid_region]
                lats = lats[valid_region]
                values = values[valid_region]

                # if min_lon > np.min(lons):  # store location of most westward cluster.
                #     min_lon = np.min(lons)

                # lons = get_consistent_longitudes(lons, min_lon)
                list_lons.extend(lons)
                list_lats.extend(lats)
                list_values.extend(values)

                # lons = lons[::WAPER_SUBSAMPLE]
                # lats = lats[::WAPER_SUBSAMPLE]
                # list_rwp_points.extend(list(zip(lons, lats)))
            else:
                pass
        else:
            out = get_region_points_and_values(
                assoc_graph, node, min_clipped_region, clip_threshold, scalar_name
            )
            if out:
                lons, lats, values = out
                
                valid_region = np.logical_and(lats >= min_latitude, lats <= max_latitude)
                lons = lons[valid_region]
                lats = lats[valid_region]
                values = values[valid_region]

                # if min_lon > np.min(lons):  # store location of most westward cluster.
                #     min_lon = np.min(lons)

                # lons = get_consistent_longitudes(lons, min_lon)
                list_lons.extend(lons)
                list_lats.extend(lats)
                list_values.extend(values)

                # lons = lons[::WAPER_SUBSAMPLE]
                # lats = lats[::WAPER_SUBSAMPLE]
                # list_rwp_points.extend(list(zip(lons, lats)))

            else:
                pass

    polygon_id = round(path_max, 2)

    xs, ys = transform_to_stereographic(list_lons, list_lats)

    weighted_ys = np.average(ys, weights=np.abs(np.array(list_values)))
    weighted_xs = np.average(xs, weights=np.abs(np.array(list_values)))

    weighted_longitude, weighted_latitude = transform_to_stereographic(
        weighted_xs, weighted_ys, inverse=True
    )

    rwp_poly = MultiPoint(list(zip(xs, ys))).convex_hull

    list_rwp_points = list(zip(xs[::WAPER_SUBSAMPLE], ys[::WAPER_SUBSAMPLE]))

    return (
        rwp_poly,
        polygon_id,
        list_rwp_points,
        weighted_longitude,
        weighted_latitude,
    )


def rasterize_all_rwps(polygon_list):
    """Get a rasterized image containing all rwp polygons

    Args:
        polygon_list (list): list of tuples of rwp polygons and rwp id

    Returns:
        np.ndarray: raster image of all polygons
    """
    if len(polygon_list) == 0:
        return None

    return features.rasterize(
        ((g, i) for g, i in polygon_list),
        out_shape=(WAPER_IMAGE_SIZE, WAPER_IMAGE_SIZE),
        all_touched=True,
        transform=WAPER_RASTER_TRANSFORM,
    )
```

---

## 3. `waper/waper/tracking/tracking_graph.py`

```waper/waper/tracking/tracking_graph.py#L1-157
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

def get_path_weight(track_graph, path):

    curr_wt = 0
    # print(path)
    for i in range(len(path) - 1):
        # print(assoc_graph.nodes[path[i]]["coords"][0], assoc_graph.nodes[path[i+1]]["coords"][0])
        curr_wt += track_graph[path[i]][path[i + 1]]["weight"]
    
    return curr_wt

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
            best_path = []
            max_weight = 0
            for path in nx.all_simple_paths(
                tracking_graph, source=start_end[0], target=start_end[1]
            ):
                
                curr_weight = get_path_weight(tracking_graph, path)
                if curr_weight > max_weight:
                    max_weight = curr_weight
                    best_path = path
                
            if len(best_path) > 0:
                track_paths.append(best_path)
                
    path_wt_dict = {}

    for path in track_paths:
        path_wt_dict[tuple(path)] = get_path_weight(tracking_graph, path)

    top_paths = list(
        filter(
            lambda f: not any(
                [
                    (  # Condition reduces to "True if path weight is less than reference and both are part of the same path"
                        path_wt_dict[tuple(f)] < path_wt_dict[tuple(g)]
                        and len(set(f) & set(g)) != 0
                    )
                    for g in track_paths
                ]
            ),
            track_paths,
        )
    )

    return top_paths

    # return track_paths
```

Those are the complete contents of all three files.

**Tool Call: Read interface module**
Status: Completed

Here are the complete contents of all three files:

---

### 1. `waper/waper/interface/api.py`

```waper/waper/interface/api.py#L1-290
from dataclasses import dataclass

from pyvista import PolyData
from networkx import Graph
from xarray import DataArray
from tqdm import tqdm
from numpy import ndarray
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from ..identification import max_min, topology, rwp_graph, utils
from ..tracking import rwp_polygon
from .visualization import (
    _plot_clusters,
    _plot_graph,
    _plot_polygons,
    _plot_rwp_paths,
    _plot_raster,
)
from ..tracking import quadtree, tracking_graph
from waper import tracking
@dataclass(eq=False, frozen=True)
class WaperConfig:

    debug: bool
    scalar_name: str
    latitude_label: str
    longitude_label: str
    time_label: str

    clip_value: float
    extrema_threshold: float

    max_latitude: float
    min_latitude: float

    node_pruning_threshold: float
    edge_pruning_threshold: float
    max_edge_weight: float

    track_pruning_threshold: float

    vtk_latitude_label: str = "Latitude"
    vtk_longitude_label: str = "Longitude"
    vtk_region_label: str = "RegionId"


@dataclass(eq=False)
class WaperSingleTimestepData:

    input_data: DataArray

    vtk_data: PolyData

    number_max_clusters: int
    number_min_clusters: int

    max_cluster_info: dict
    min_cluster_info: dict

    all_minima: PolyData
    all_maxima: PolyData

    association_graph: Graph
    pruned_graph: Graph

    identified_rwp_paths: list

    rwp_info: dict

    raster_data: ndarray
    raster_features: list
    quadtree: Graph

    def __init__(self, input_data: DataArray, config: WaperConfig) -> None:
        self.input_data = input_data
        self.vtk_data = utils.get_vtk_object_from_data_array(
            input_data,
            input_data[config.longitude_label],
            input_data[config.latitude_label],
            array_name=config.scalar_name,
        )
        self.rwp_info = {}
        return

def logging(log_info, config: WaperConfig):
    if config.debug:
        print(log_info)


def _identify_rwps(scalar_data: DataArray, config: WaperConfig) -> WaperSingleTimestepData:

    input_data = scalar_data
    latitude = input_data[config.latitude_label].values
    longitude = input_data[config.longitude_label].values

    time_step_data = WaperSingleTimestepData(input_data=input_data, config=config)
    # Identify and cluster maxima

    data_with_maxima = max_min.add_maxima_data(
        input_data, config.scalar_name, longitude, latitude
    )

    if config.min_latitude:
        data_with_maxima = data_with_maxima.clip_scalar(
            scalars=config.vtk_latitude_label, invert=False, value=config.min_latitude
        )

    if config.max_latitude:
        data_with_maxima = data_with_maxima.clip_scalar(
            scalars=config.vtk_latitude_label, invert=True, value=config.max_latitude
        )

    clipped_data_with_maxima = data_with_maxima.clip_scalar(
        scalars=config.scalar_name, invert=False, value=config.clip_value
    )
    
    connectivity = topology.identify_connected_regions(clipped_data_with_maxima)

    maxima_points = max_min.extract_maxima_points(connectivity, config.extrema_threshold, config.scalar_name)
    
    clustered_points = topology.cluster_max(
        data_with_maxima, connectivity, maxima_points, config.scalar_name
    )

    (
        cluster_max_arr,
        cluster_max_point,
        max_pt_dict,
        num_max_clusters,
    ) = topology.max_cluster_assign(clustered_points, config.scalar_name)

    time_step_data.all_maxima = maxima_points
    time_step_data.number_max_clusters = num_max_clusters
    time_step_data.max_cluster_info = max_pt_dict

    # Identify and cluster minima

    data_with_minima = max_min.add_minima_data(
        input_data, config.scalar_name, longitude, latitude
    )
    

    if config.max_latitude:
        data_with_minima = data_with_minima.clip_scalar(
            scalars=config.vtk_latitude_label, invert=True, value=config.max_latitude
        )

    if config.min_latitude:
        data_with_minima = data_with_minima.clip_scalar(
            scalars=config.vtk_latitude_label, invert=False, value=config.min_latitude
        )

    clipped_data_with_minima = data_with_minima.clip_scalar(
        scalars=config.scalar_name, value=-config.clip_value, invert=True
    )

    connectivity = topology.identify_connected_regions(clipped_data_with_minima)

    minima_points = max_min.extract_minima_points(
        connectivity, -config.extrema_threshold, config.scalar_name
    )

    clustered_points = topology.cluster_min(
        data_with_minima, connectivity, minima_points, config.scalar_name
    )

    (
        cluster_min_arr,
        cluster_min_point,
        min_pt_dict,
        num_min_clusters,
    ) = topology.min_cluster_assign(clustered_points, config.scalar_name)
    
    time_step_data.all_minima = minima_points
    time_step_data.number_min_clusters = num_min_clusters
    time_step_data.min_cluster_info = min_pt_dict

    # Compute and Prune Association Graph
    
    zero_isocontour = time_step_data.vtk_data.contour([0], scalars=config.scalar_name)
    time_step_data.association_graph = rwp_graph.compute_association_graph(
        maxima_points, minima_points, zero_isocontour, config.scalar_name
    )

    node_pruned_graph = rwp_graph.prune_association_graph_nodes(
        time_step_data.association_graph, scalar_threshold=config.node_pruning_threshold
    )

    time_step_data.pruned_graph = rwp_graph.prune_association_graph_edges(
        node_pruned_graph, config.edge_pruning_threshold, config.max_edge_weight
    )

    time_step_data.identified_rwp_paths = rwp_graph.get_ranked_paths(
        time_step_data.pruned_graph, config.max_edge_weight
    )

    for index, path in enumerate(time_step_data.identified_rwp_paths):
        (
            polygon,
            rwp_id,
            sample_points,
            weighted_lon,
            weighted_lat,
        ) = rwp_polygon.get_polygon_for_rwp_path(
            path, time_step_data.pruned_graph, time_step_data.vtk_data, config.scalar_name,
            config.min_latitude, config.max_latitude
        )
        time_step_data.rwp_info[tuple(path)] = {
            "polygon": polygon,
            "rwp_id": rwp_id,
            "sample_points": sample_points,
            "weighted_longitude": weighted_lon,
            "weighted_latitude": weighted_lat,
        }

    list_polygons = []
    for path in time_step_data.identified_rwp_paths:
        list_polygons.append(
            (
                time_step_data.rwp_info[tuple(path)]["polygon"],
                time_step_data.rwp_info[tuple(path)]["rwp_id"],
            )
        )
        
    if len(list_polygons) == 0:
        print('No RWPs found, change thresholds')
        
    time_step_data.raster_data = rwp_polygon.rasterize_all_rwps(list_polygons)

    features = set(np.unique(time_step_data.raster_data))
    features.add(0)

    time_step_data.raster_features = features

    if time_step_data.raster_data is None:
        time_step_data.quadtree = None
    else:
        time_step_data.quadtree = quadtree.create_quadtree(time_step_data.raster_data)

    return time_step_data


def _track_rwps(time_step_data, num_time_steps):

    return tracking_graph.build_tracking_graph(time_step_data, num_time_steps)


class Waper:
    def __init__(
        self,
        data_array,
        scalar_name,
        latitude_label,
        longitude_label,
        time_label,
        clip_value=2,
        extrema_threshold=10,
        max_latitude=None,
        min_latitude=None,
        node_pruning_threshold=20,
        edge_pruning_threshold=3e-5,
        track_pruning_threshold=0.3,
        max_edge_weight=1,
        debug=False
    ) -> None:

        self._config = WaperConfig(
            scalar_name=scalar_name,
            latitude_label=latitude_label,
            longitude_label=longitude_label,
            time_label=time_label,
            clip_value=clip_value,
            extrema_threshold=extrema_threshold,
            max_latitude=max_latitude,
            min_latitude=min_latitude,
            node_pruning_threshold=node_pruning_threshold,
            edge_pruning_threshold=edge_pruning_threshold,
            track_pruning_threshold=track_pruning_threshold,
            max_edge_weight=max_edge_weight,
            debug=debug
        )

        self.data_array = data_array
        self._num_time_steps = len(data_array[time_label])
        self._time_step_data = []

    def identify_rwps(self):

        for i in tqdm(range(self._num_time_steps)):
            self._time_step_data.append(
                _identify_rwps(self.data_array[self._config.scalar_name][i], self._config)
            )

    def track_rwps(self, num_time_steps=None):

        self._tracking_graph = _track_rwps(self._time_step_data, num_time_steps)
        self._pruned_tracking_graph = tracking_graph.prune_tracking_graph(
            self._tracking_graph, self._config.track_pruning_threshold
        )

    def plot_clusters(self, time_index):

        time_step_data = self._time_step_data[time_index]
        return _plot_clusters(
            time_step_data.input_data,
            time_step_data.all_maxima,
            time_step_data.all_minima,
            time_step_data.max_cluster_info,
            time_step_data.min_cluster_info,
            self._config.vtk_longitude_label,
            self._config.vtk_latitude_label,
            self._config.vtk_region_label,
            self._config.clip_value,
        )

    def plot_association_graph(self, time_index, ax=None):
        time_step_data = self._time_step_data[time_index]

        return _plot_graph(time_step_data.association_graph, time_step_data.input_data, ax=ax)

    def plot_pruned_graph(self, time_index, ax=None):
        time_step_data = self._time_step_data[time_index]

        return _plot_graph(time_step_data.pruned_graph, time_step_data.input_data, ax=ax)

    def plot_rwp_graphs(self, time_index, ax=None, plot_scalar_data=True):
        time_step_data = self._time_step_data[time_index]
        
        field = None
        if plot_scalar_data:
            field = time_step_data.input_data

        return _plot_rwp_paths(
            time_step_data.pruned_graph,
            time_step_data.identified_rwp_paths,
            field,
            ax=ax
        )

    def plot_rwp_polygons(self, time_index, plot_samples=False, ax=None):
        time_step_data = self._time_step_data[time_index]

        poly_list = [rwp_info["polygon"] for rwp_info in time_step_data.rwp_info.values()]
        sample_points_list = [
            rwp_info["sample_points"] for rwp_info in time_step_data.rwp_info.values()
        ]

        weighted_lon_list = [
            rwp_info["weighted_longitude"] for rwp_info in time_step_data.rwp_info.values()
        ]

        weighted_lat_list = [
            rwp_info["weighted_latitude"] for rwp_info in time_step_data.rwp_info.values()
        ]

        return _plot_polygons(
            poly_list,
            time_step_data.input_data,
            sample_points_list,
            weighted_lon_list,
            weighted_lat_list,
            plot_samples=plot_samples,
            ax=ax
        )

    def plot_raster(self, time_index):
        time_step_data = self._time_step_data[time_index]

        return _plot_raster(time_step_data.raster_data)

    def plot_tracks(self, threshold=None):
        pruned = tracking_graph.prune_tracking_graph(
            self._tracking_graph, threshold=threshold
        )
        paths = tracking_graph.get_track_paths(pruned)
        return _plot_rwp_paths(
            pruned,
            paths,
            None,
            path_transform=ccrs.Geodetic(),
            map_projection=ccrs.PlateCarree(),
        )

    def plot_track_polygons(self, path, plot_samples=False, ax=None):

        poly_list = []
        sample_points_list = []
        weighted_lon_list = []
        weighted_lat_list = []
        for node in path:
            time_step_data = self._time_step_data[node[0]]

            for rwp in time_step_data.rwp_info.values():
                if abs(rwp["rwp_id"] - node[1]) < 1e-2:
                    poly_list.append(rwp["polygon"])
                    sample_points_list.append(rwp["sample_points"])
                    weighted_lon_list.append(rwp["weighted_longitude"])
                    weighted_lat_list.append(rwp["weighted_latitude"])

        return _plot_polygons(
            poly_list,
            None,
            sample_points_list,
            weighted_lon_list,
            weighted_lat_list,
            plot_samples=plot_samples,
            ax=ax
        )

    def plot_track_rwps(self, path, ax=None):

        rwp_list = []
        
        if ax is None:
            ax = plt.subplot(
                projection=ccrs.Orthographic(central_longitude=180, central_latitude=90)
            )
        
        for node in path:
            time_step_data = self._time_step_data[node[0]]

            for path, rwp_info in time_step_data.rwp_info.items():
                if abs(rwp_info["rwp_id"] - node[1]) < 1e-2:
                    rwp_list.append(([path], time_step_data.pruned_graph))

        for path, pruned_graph in rwp_list:
            _plot_rwp_paths(paths=path, rwp_graph=pruned_graph, ax=ax)
            
        return ax
```

---

### 2. `waper/waper/interface/visualization.py`

```waper/waper/interface/visualization.py#L1-296
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pyproj import transform
import pyvista as pv
import numpy as np
from xarray import DataArray
from matplotlib.colors import LinearSegmentedColormap

from ..tracking.rwp_polygon import WAPER_X_BOUNDS, WAPER_Y_BOUNDS

cdictDivergeNL = {'red' : (
                  (0.,0.455,0.455),
                  (0.25,0.670,0.670),
                  (0.4,0.878,0.878),
                  (0.5,1.000,1.000),
                  (0.6,0.996,0.996),
                  (0.75,0.992,0.992),
                  (1.,0.957,0.957),
                  ),

        'green' : (
                  (0.,0.678,0.678),
                  (0.25,0.851,0.851),
                  (0.4,0.953,0.953),
                  (0.5,1.000,1.000),
                  (0.6,0.878,0.878),
                  (0.75,0.682,0.682),
                  (1.,0.427,0.427),
                  ),

        'blue' : (
                  (0.,0.820,0.820),
                  (0.25,0.914,0.914),
                  (0.4,0.973,0.973),
                  (0.5,1.,1.),
                  (0.6,0.565,0.565),
                  (0.75,0.380,0.380),
                  (1.,0.263,0.263),
                  ),

    }

NLDivCmap = LinearSegmentedColormap('NLDCmap',cdictDivergeNL)

def _plot_clusters(
    input_data,
    maxima_points,
    minima_points,
    max_pt_dict,
    min_pt_dict,
    vtk_lon_label,
    vtk_lat_label,
    vtk_region_label,
    clip_value,
):

    ax = plt.subplot(211, projection=ccrs.PlateCarree(central_longitude=180))

    input_data.plot.contour(
        ax=ax,
        levels=12,
        transform=ccrs.PlateCarree(central_longitude=0),
        labels=True,
        colors="k",
        linewidth=1,
        zorder=1,
    )

    input_data.plot.contour(
        ax=ax,
        levels=[-clip_value, clip_value],
        transform=ccrs.PlateCarree(central_longitude=0),
        labels=True,
        colors="r",
        linewidth=3,
        zorder=1,
    )

    out = pv.wrap(maxima_points)
    for lon, lat, region_id in zip(
        out[vtk_lon_label], out[vtk_lat_label], out[vtk_region_label]
    ):

        ax.annotate(
            str(region_id + 1),
            (lon, lat),
            bbox=dict(boxstyle="round", fc="white", ec="b"),
            transform=ccrs.PlateCarree(central_longitude=0),
        )

    out = pv.wrap(minima_points)
    for lon, lat, region_id in zip(
        out[vtk_lon_label], out[vtk_lat_label], out[vtk_region_label]
    ):

        ax.annotate(
            str(-region_id - 1),
            (lon, lat),
            bbox=dict(boxstyle="round", fc="white", ec="b"),
            transform=ccrs.PlateCarree(central_longitude=0),
        )

    ax = plt.subplot(212, projection=ccrs.PlateCarree(central_longitude=180))

    input_data.plot.contour(
        ax=ax,
        levels=12,
        colors="k",
        transform=ccrs.PlateCarree(central_longitude=0),
        labels=True,
    )

    for cluster_id, points in max_pt_dict.items():

        for point in points:
            ax.annotate(
                str(cluster_id),
                (point[0], point[1]),
                bbox=dict(boxstyle="round", fc="white", ec="b"),
                transform=ccrs.PlateCarree(central_longitude=0),
            )

    for cluster_id, points in min_pt_dict.items():

        for point in points:
            if cluster_id == 0:
                cluster_id = 100
            ax.annotate(
                str(-cluster_id),
                (point[0], point[1]),
                bbox=dict(boxstyle="round", fc="white", ec="b"),
                transform=ccrs.PlateCarree(central_longitude=0),
            )

    plt.tight_layout()
    return ax


def _plot_graph(rwp_graph, scalar_data=None, ax=None):

    if ax is None:
        ax = plt.subplot(
            projection=ccrs.Orthographic(central_longitude=180, central_latitude=90)
        )

    if isinstance(scalar_data, DataArray):
        scalar_data.plot.contour(
            ax=ax,
            levels=12,
            transform=ccrs.PlateCarree(central_longitude=0),
            labels=True,
            colors="k",
            linewidths=1,
            zorder=1,
        )
        
        scalar_data.plot.contourf(
            ax=ax,
            levels=12,
            transform=ccrs.PlateCarree(central_longitude=0),
            zorder=1,
            cmap=NLDivCmap,
            add_colorbar=True,
            cbar_kwargs=dict(
                orientation='horizontal',
                shrink=0.6,
                aspect=30
            )
        )
        
    for node in rwp_graph.nodes:
        coords = rwp_graph.nodes[node]["coords"]
        ax.scatter(
            coords[0], coords[1], color="r", transform=ccrs.PlateCarree(central_longitude=0)
        )

    for edge in rwp_graph.edges:
        node1_coords = rwp_graph.nodes[edge[0]]["coords"]
        node2_coords = rwp_graph.nodes[edge[1]]["coords"]

        ax.plot(
            [node1_coords[0], node2_coords[0]],
            [node1_coords[1], node2_coords[1]],
            color="b",
            transform=ccrs.PlateCarree(central_longitude=0),
        )

    plt.tight_layout()
    return ax


def _plot_rwp_paths(rwp_graph, paths, scalar_data=None, ax=None):

    if ax is None:
        ax = plt.subplot(
            projection=ccrs.Orthographic(central_longitude=180, central_latitude=90)
        )

    colors = plt.cm.tab20.colors

    if isinstance(scalar_data, DataArray):
        scalar_data.plot.contour(
            ax=ax,
            levels=12,
            transform=ccrs.PlateCarree(central_longitude=0),
            labels=True,
            colors="k",
            linewidths=2,
            zorder=1,
        )
        
        scalar_data.plot.contourf(
            ax=ax,
            levels=11,
            transform=ccrs.PlateCarree(central_longitude=0),
            zorder=1,
            cmap=NLDivCmap,
            add_colorbar=True,
            cbar_kwargs=dict(
                orientation='horizontal',
                shrink=0.6,
                aspect=30
            )
        )

    for index, path in enumerate(paths):
        for node in path:
            coords = rwp_graph.nodes[node]["coords"]
            color = 'r'
            if node < 0:
                color = 'b'
            
            ax.scatter(coords[0], coords[1], color=color, transform=ccrs.PlateCarree())

        for edge in [(path[i], path[i + 1]) for i in range(len(path) - 1)]:
            node1_coords = rwp_graph.nodes[edge[0]]["coords"]
            node2_coords = rwp_graph.nodes[edge[1]]["coords"]

            delta_coord = 0
            if node1_coords[0] - node2_coords[0] > 180:
                delta_coord = 360

            ax.plot(
                [node1_coords[0], node2_coords[0] + delta_coord],
                [node1_coords[1], node2_coords[1]],
                color=colors[index % 20],
                transform=ccrs.PlateCarree(),
            )

    plt.tight_layout()
    return ax


def _plot_polygons(
    poly_list,
    scalar_data,
    sample_points_list,
    weighted_lon_list=None,
    weighted_lat_list=None,
    plot_samples=False,
    ax=None,
):

    if ax is None:
        ax = plt.subplot(
            projection=ccrs.Stereographic(central_longitude=0, central_latitude=90)
        )

    if not (scalar_data is None):
        scalar_data.plot.contour(
            ax=ax,
            levels=12,
            transform=ccrs.PlateCarree(central_longitude=0),
            labels=True,
            colors="k",
            linewidth=1,
            zorder=1,
        )

    for poly in poly_list:

        lons, lats = poly.exterior.coords.xy

        ax.plot(
            lons, lats, transform=ccrs.Stereographic(central_longitude=0, central_latitude=90)
        )

        for lon, lat in zip(lons, lats):
            ax.scatter(
                lon,
                lat,
                color="r",
                s=30,
                zorder=100,
                transform=ccrs.Stereographic(central_longitude=0, central_latitude=90),
            )

    if not (weighted_lat_list is None):
        for index, coords in enumerate(list(zip(weighted_lon_list, weighted_lat_list))):
            lon, lat = coords
            ax.scatter(
                lon,
                lat,
                transform=ccrs.PlateCarree(central_longitude=0),
                s=50,
                color="green",
                zorder=100,
            )

            ax.annotate(
                str(index),
                (lon, lat),
                bbox=dict(boxstyle="round", fc="white", ec="b"),
                transform=ccrs.PlateCarree(central_longitude=0),
                zorder=1000,
            )

    if plot_samples:
        for sample_points in sample_points_list:
            for lon, lat in sample_points:
                ax.scatter(
                    lon,
                    lat,
                    color="b",
                    s=5,
                    transform=ccrs.Stereographic(central_longitude=0, central_latitude=90),
                )

    plt.tight_layout()
    return ax


def _plot_raster(raster_data):
    ax = plt.subplot(projection=ccrs.Stereographic(central_longitude=0, central_latitude=90))

    ax.imshow(
        np.ma.array(raster_data, mask=(raster_data == 0)),
        origin="lower",
        cmap="tab20b",
        extent=(WAPER_X_BOUNDS[1], WAPER_X_BOUNDS[0], WAPER_Y_BOUNDS[1], WAPER_Y_BOUNDS[0]),
        alpha=0.7,
    )

    plt.tight_layout()
    return ax
```

---

### 3. `waper/waper/__init__.py`

```waper/waper/__init__.py#L1-1
from .interface import (Waper, WaperSingleTimestepData, max_min, topology, WaperConfig)
```

---

Those are the complete contents of all three files. The `__init__.py` is just a single line that re-exports key symbols from the `interface` subpackage. The `api.py` file contains the core `Waper` class and the RWP identification/tracking pipeline logic. The `visualization.py` file contains all the matplotlib/cartopy-based plotting functions used by the `Waper` class.

## Assistant

Now let me also check for any additional configuration files, tests, or examples:

**Tool Call: 8 matches**
Status: Completed

file:///Users/joymonteiro/github/waper/.bettercodehub.yml

file:///Users/joymonteiro/github/waper/.github/workflows/test.yaml

file:///Users/joymonteiro/github/waper/.prospector.yml

file:///Users/joymonteiro/github/waper/.readthedocs.yml

file:///Users/joymonteiro/github/waper/docs/spelling_wordlist.txt

file:///Users/joymonteiro/github/waper/environment.yml

file:///Users/joymonteiro/github/waper/pyproject.toml

file:///Users/joymonteiro/github/waper/tox.ini

**Tool Call: 3 matches**
Status: Completed

file:///Users/joymonteiro/github/waper/.github/workflows/test.yaml

file:///Users/joymonteiro/github/waper/tests

file:///Users/joymonteiro/github/waper/tests/smoke_test.py

**Tool Call: 2 matches**
Status: Completed

file:///Users/joymonteiro/github/waper/scripts/Tracking_final.ipynb

file:///Users/joymonteiro/github/waper/scripts/Tracking_final_Malavika.ipynb

**Tool Call: Read file `pyproject.toml`**
Status: Completed

```waper/pyproject.toml
# BUILD

[build-system]
requires = ["setuptools>=61.2", "setuptools-scm"]
build-backend = "setuptools.build_meta"


# Poetry

# Information required for building (sdist/wheel)
## Also renders on pypi as 'subtitle'
[project]
name = "waper"
version = "0.1.0"
description = "Rossby wave packet tracker"
authors = [{name="Malavika Biju"}, 
            {name="Joy Monteiro"},
            {name="Karran Pandey"}]
maintainers = [{name="Joy Monteiro", email="joy.merwin@gmail.com"}]
license = {file= "LICENSE"}
readme = "README.rst"
requires-python = ">= 3.9"

keywords = [
    "python package",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: End Users/Desktop",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU Affero General Public License v3",
    "Natural Language :: English",
    "Operating System :: Unix",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: System :: Software Distribution",
    "Typing :: Typed"
]

[tool.setuptools]
packages=["waper", "waper.identification", "waper.tracking", "waper.interface"]

#  [tool.setuptools.packages.find]
# {include = ["tests", 
#     "docs/**/*.rst",
#     "docs/conf.py", 
#     "src/**/*.typed", 
#     "src/stubs/*.pyi",
#     "pyproject.toml",
#     "LICENSE",
#     "README.rst",
#     "CONTRIBUTING.md",
#     "CHANGELOG.rst",
# ]},
# {exclude = [
#     ".github/*",
#     "docs/*",
#     "requirements/*",
#     "scripts/*.py",
#     "tox.ini",
#     ".bettercodehub.yml",
#     ".circleci/config.yml",
#     ".coveragerc",
#     ".DS_Store",
#     ".gitignore",
#     ".prospector.yml",
#     ".pylintrc",
#     ".readthedocs.yml",
#     ".scrutinizer.yml",
#     ".travis.yml"
# ]}]

# A list of all of the optional dependencies, some of which are included in the
# below `extras`. They can be opted into by apps.


# PyPi url links, that appear in 'Project Links' section
[project.urls]
"Bug Tracker" = "https://github.com/waper/issues"
"CI: Github Actions" = "https://github.com/JoyMonteiro/waper/actions"
"Documentation" = "https://waper.readthedocs.io/"
"Source Code" = "https://github.com/JoyMonteiro/waper"
"Changelog" = "https://github.com/JoyMonteiro/waper/blob/master/CHANGELOG.rst"
"Code of Conduct" = "https://github.com/JoyMonteiro/waper/blob/master/CONTRIBUTING.rst"
```


**Tool Call: Read file `tests/smoke_test.py`**
Status: Completed

```waper/tests/smoke_test.py
def test_smoke_is_not_risingfrom_module():
    import my_new_project

    assert my_new_project is not None

```


**Tool Call: Read file `environment.yml`**
Status: Completed

```waper/environment.yml
name: base
channels:
  - conda-forge
  - defaults
dependencies:
  - _libgcc_mutex=0.1=conda_forge
  - _openmp_mutex=4.5=2_kmp_llvm
  - aom=3.4.0=h27087fc_1
  - appdirs=1.4.4=pyh9f0ad1d_0
  - brotli=1.0.9=h166bdaf_7
  - brotli-bin=1.0.9=h166bdaf_7
  - brotlipy=0.7.0=py39hb9d737c_1004
  - bzip2=1.0.8=h7f98852_4
  - c-ares=1.18.1=h7f98852_0
  - ca-certificates=2022.9.14=ha878542_0
  - cartopy=0.21.0=py39hf5d525c_0
  - cffi=1.15.1=py39h74dc2b5_0
  - cftime=1.6.2=py39h2ae25f5_0
  - charset-normalizer=2.1.1=pyhd8ed1ab_0
  - conda=4.14.0=py39hf3d152e_0
  - conda-package-handling=1.9.0=py39hb9d737c_0
  - contourpy=1.0.5=py39hf939315_0
  - curl=7.83.1=h7bff187_0
  - cycler=0.11.0=pyhd8ed1ab_0
  - double-conversion=3.2.0=h9c3ff4c_0
  - eigen=3.4.0=h4bd325d_0
  - elfutils=0.186=he364ef2_0
  - expat=2.4.9=h27087fc_0
  - ffmpeg=4.4.2=gpl_h45a1190_107
  - font-ttf-dejavu-sans-mono=2.37=hab24e00_0
  - font-ttf-inconsolata=3.000=h77eed37_0
  - font-ttf-source-code-pro=2.038=h77eed37_0
  - font-ttf-ubuntu=0.83=hab24e00_0
  - fontconfig=2.14.0=hc2a2eb6_1
  - fonts-conda-ecosystem=1=0
  - fonts-conda-forge=1=0
  - fonttools=4.37.3=py39hb9d737c_0
  - freetype=2.12.1=hca18f0e_0
  - geos=3.11.0=h27087fc_0
  - gettext=0.19.8.1=h0b5b191_1005
  - gl2ps=1.4.2=h0708190_0
  - gmp=6.2.1=h58526e2_0
  - gnutls=3.7.6=hbf5b4be_4
  - hdf4=4.2.15=h9772cbc_4
  - hdf5=1.12.2=nompi_h2386368_100
  - icu=70.1=h27087fc_0
  - idna=3.4=pyhd8ed1ab_0
  - imageio=2.22.0=pyhfa7a67d_0
  - jpeg=9e=h166bdaf_2
  - jsoncpp=1.9.5=h4bd325d_1
  - keyutils=1.6.1=h166bdaf_0
  - kiwisolver=1.4.4=py39hf939315_0
  - krb5=1.19.3=h3790be6_0
  - lame=3.100=h7f98852_1001
  - lcms2=2.12=hddcbb42_0
  - ld_impl_linux-64=2.38=h1181459_1
  - lerc=4.0.0=h27087fc_0
  - libarchive=3.5.2=hb890918_3
  - libblas=3.9.0=16_linux64_openblas
  - libbrotlicommon=1.0.9=h166bdaf_7
  - libbrotlidec=1.0.9=h166bdaf_7
  - libbrotlienc=1.0.9=h166bdaf_7
  - libcblas=3.9.0=16_linux64_openblas
  - libcurl=7.83.1=h7bff187_0
  - libdeflate=1.13=h166bdaf_0
  - libdrm=2.4.113=h166bdaf_0
  - libedit=3.1.20191231=he28a2e2_2
  - libev=4.33=h516909a_1
  - libffi=3.3=he6710b0_2
  - libgcc-ng=12.1.0=h8d9b700_16
  - libgfortran-ng=12.1.0=h69a702a_16
  - libgfortran5=12.1.0=hdcd56e2_16
  - libhwloc=2.8.0=h32351e8_1
  - libiconv=1.16=h516909a_0
  - libidn2=2.3.3=h166bdaf_0
  - liblapack=3.9.0=16_linux64_openblas
  - libmamba=0.25.0=hd8a31e3_2
  - libmambapy=0.25.0=py39hd55135b_2
  - libmicrohttpd=0.9.75=h2603550_1
  - libnetcdf=4.8.1=nompi_h21705cb_104
  - libnghttp2=1.47.0=hdcd2b5c_1
  - libogg=1.3.4=h7f98852_1
  - libopenblas=0.3.21=pthreads_h78a6416_3
  - libpciaccess=0.16=h516909a_0
  - libpng=1.6.38=h753d276_0
  - libsolv=0.7.22=h6239696_0
  - libsqlite=3.39.3=h753d276_0
  - libssh2=1.10.0=haa6b8db_3
  - libstdcxx-ng=12.1.0=ha89aaad_16
  - libtasn1=4.19.0=h166bdaf_0
  - libtheora=1.1.1=h7f98852_1005
  - libtiff=4.4.0=h0e0dad5_3
  - libunistring=0.9.10=h7f98852_0
  - libuuid=2.32.1=h7f98852_1000
  - libva=2.15.0=h166bdaf_0
  - libvorbis=1.3.7=h9c3ff4c_0
  - libvpx=1.11.0=h9c3ff4c_3
  - libwebp-base=1.2.4=h166bdaf_0
  - libxcb=1.13=h7f98852_1004
  - libxml2=2.9.14=h22db469_4
  - libzip=1.9.2=hc869a4a_1
  - libzlib=1.2.12=h166bdaf_3
  - llvm-openmp=14.0.4=he0ac6c6_0
  - loguru=0.6.0=py39hf3d152e_1
  - lz4-c=1.9.3=h9c3ff4c_1
  - lzo=2.10=h516909a_1000
  - mamba=0.25.0=py39hfa8f2c8_2
  - matplotlib-base=3.6.0=py39hf9fd14e_0
  - mesalib=21.2.5=h0e4506f_3
  - munkres=1.1.4=pyh9f0ad1d_0
  - ncurses=6.3=h5eee18b_3
  - netcdf4=1.6.1=nompi_py39hfaa66c4_100
  - nettle=3.8.1=hc379101_1
  - numpy=1.23.3=py39hba7629e_0
  - openh264=2.3.0=h27087fc_0
  - openjpeg=2.5.0=h7d73246_1
  - openssl=1.1.1q=h166bdaf_0
  - p11-kit=0.23.21=hb875675_1
  - pandas=1.5.0=py39h4661b88_0
  - pillow=9.2.0=py39hd5dbb17_2
  - pip=22.1.2=py39h06a4308_0
  - proj=9.0.1=h93bde94_1
  - pthread-stubs=0.4=h36c2ea0_1001
  - pugixml=1.11.4=h9c3ff4c_0
  - pybind11-abi=4=hd8ed1ab_3
  - pycosat=0.6.3=py39hb9d737c_1010
  - pycparser=2.21=pyhd8ed1ab_0
  - pyopenssl=22.0.0=pyhd8ed1ab_0
  - pyproj=3.4.0=py39hdcf6798_0
  - pyshp=2.3.1=pyhd8ed1ab_0
  - pysocks=1.7.1=pyha2e5f31_6
  - python=3.9.13=haa1d7c7_1
  - python-dateutil=2.8.2=pyhd8ed1ab_0
  - python_abi=3.9=2_cp39
  - pyvista=0.36.1=pyhd8ed1ab_0
  - readline=8.1.2=h7f8727e_1
  - reproc=14.2.3=h7f98852_0
  - reproc-cpp=14.2.3=h9c3ff4c_0
  - requests=2.28.1=pyhd8ed1ab_1
  - ruamel_yaml=0.15.80=py39hb9d737c_1007
  - scipy=1.9.1=py39h8ba3f38_0
  - scooby=0.6.0=pyhd8ed1ab_0
  - setuptools=63.4.1=py39h06a4308_0
  - shapely=1.8.4=py39h68ae834_0
  - six=1.16.0=pyh6c4a22f_0
  - sqlite=3.39.2=h5082296_0
  - svt-av1=1.2.0=h27087fc_0
  - tbb=2021.5.0=h924138e_3
  - tbb-devel=2021.5.0=h924138e_3
  - tk=8.6.12=h1ccaba5_0
  - toolz=0.12.0=pyhd8ed1ab_0
  - tqdm=4.64.1=pyhd8ed1ab_0
  - typing_extensions=4.3.0=pyha770c72_0
  - tzdata=2022c=h04d1e81_0
  - unicodedata2=14.0.0=py39hb9d737c_1
  - utfcpp=3.2.1=ha770c72_0
  - vtk=9.1.0=osmesa_py39h4086a4b_114
  - wheel=0.37.1=pyhd3eb1b0_0
  - x264=1!164.3095=h166bdaf_2
  - x265=3.5=h924138e_3
  - xarray=2022.6.0=pyhd8ed1ab_1
  - xorg-damageproto=1.2.1=h7f98852_1002
  - xorg-fixesproto=5.0=h7f98852_1002
  - xorg-glproto=1.4.17=h7f98852_1002
  - xorg-kbproto=1.0.7=h7f98852_1002
  - xorg-libice=1.0.10=h7f98852_0
  - xorg-libsm=1.2.3=hd9c2040_1000
  - xorg-libx11=1.7.2=h7f98852_0
  - xorg-libxau=1.0.9=h7f98852_0
  - xorg-libxdamage=1.1.5=h7f98852_1
  - xorg-libxdmcp=1.1.3=h7f98852_0
  - xorg-libxext=1.3.4=h7f98852_1
  - xorg-libxfixes=5.0.3=h7f98852_1004
  - xorg-libxrandr=1.5.2=h7f98852_1
  - xorg-libxrender=0.9.10=h7f98852_1003
  - xorg-libxt=1.2.1=h7f98852_2
  - xorg-randrproto=1.5.0=h7f98852_1001
  - xorg-renderproto=0.11.1=h7f98852_1002
  - xorg-util-macros=1.19.3=h7f98852_0
  - xorg-xextproto=7.3.0=h7f98852_1002
  - xorg-xf86vidmodeproto=2.3.1=h7f98852_1002
  - xorg-xproto=7.0.31=h7f98852_1007
  - xz=5.2.5=h7f8727e_1
  - yaml=0.2.5=h7f98852_2
  - yaml-cpp=0.7.0=h27087fc_1
  - zlib=1.2.12=h5eee18b_3
  - zstd=1.5.2=h6239696_4
  - pip:
    - cachecontrol==0.12.11
    - cachy==0.3.0
    - certifi==2022.9.14
    - cleo==1.0.0a5
    - crashtest==0.3.1
    - cryptography==38.0.1
    - distlib==0.3.6
    - dulwich==0.20.46
    - filelock==3.8.0
    - html5lib==1.1
    - jaraco-classes==3.2.2
    - jeepney==0.8.0
    - joblib==1.2.0
    - jsonschema==4.16.0
    - keyring==23.9.3
    - lockfile==0.12.2
    - more-itertools==8.14.0
    - msgpack==1.0.4
    - networkx==2.8.6
    - pexpect==4.8.0
    - pkginfo==1.8.3
    - platformdirs==2.5.2
    - poetry==1.2.1
    - poetry-core==1.2.0
    - poetry-plugin-export==1.0.7
    - ptyprocess==0.7.0
    - pylev==1.4.0
    - pyrsistent==0.18.1
    - requests-toolbelt==0.9.1
    - scikit-learn==1.1.2
    - secretstorage==3.3.3
    - shellingham==1.5.0
    - sklearn==0.0
    - threadpoolctl==3.1.0
    - tomlkit==0.11.4
    - urllib3==1.26.12
    - virtualenv==20.16.5
    - waper==0.0.1
    - webencodings==0.5.1
prefix: /data/miniconda3/envs/waper

```


**Tool Call: Read file `waper/interface/__init__.py`**
Status: Completed

```waper/waper/interface/__init__.py
from .api import Waper, WaperSingleTimestepData, WaperConfig, max_min, topology
__all__ = (Waper, WaperSingleTimestepData, max_min, topology, WaperConfig)
```


**Tool Call: Read file `waper/identification/__init__.py`**
Status: Completed

```waper/waper/identification/__init__.py

```


**Tool Call: Read file `waper/tracking/__init__.py`**
Status: Completed

```waper/waper/tracking/__init__.py

```


## Assistant

Now let me check the conductor directory structure for completeness:

**Tool Call: List the `waper` directory's contents**
Status: Completed

# Folders:
waper/waper/identification
waper/waper/interface
waper/waper/tracking

# Files:
waper/waper/__init__.py
waper/waper/py.typed


**Tool Call: List the `waper/conductor` directory's contents**
Status: Completed


# Files:
waper/conductor/architecture_and_algorithm.md


## Assistant

I now have a thorough understanding of the entire codebase. Here is my comprehensive assessment.

---

# WAPER: Comprehensive Code Assessment

## 1. Architecture Overview

WAPER is cleanly structured into three logical modules that mirror the scientific workflow:

| Module | Purpose | Key Files |
|---|---|---|
| `identification` | Per-timestep RWP topology extraction | `max_min.py`, `topology.py`, `rwp_graph.py`, `utils.py` |
| `tracking` | Inter-timestep feature matching | `quadtree.py`, `rwp_polygon.py`, `tracking_graph.py` |
| `interface` | User-facing API & visualization | `api.py`, `visualization.py` |

This separation of concerns is well thought out. The `Waper` class in `api.py` acts as an orchestrator, and the `WaperConfig` frozen dataclass cleanly groups hyperparameters. The `WaperSingleTimestepData` dataclass carries all intermediate artifacts for a timestep, which is excellent for debugging and visualization.

---

## 2. How Well the Code Implements the Algorithm

### 2.1 Faithful Implementation of Pandey et al. (2020)

The core pipeline faithfully follows the Pandey paper's geometric-topological approach:

1. **Critical Point Extraction** → `max_min.py` identifies local maxima/minima via neighborhood comparisons
2. **Level Set Construction** → `clip_scalar()` calls separate the superlevel and sublevel sets
3. **Connected Component Labeling** → `topology.identify_connected_regions()` via VTK's connectivity filter
4. **Affinity Propagation Clustering** → `topology.cluster_max/cluster_min` compute pairwise geodesic distances, build similarity matrices, and cluster via `sklearn.cluster.AffinityPropagation`
5. **Association Graph** → `rwp_graph.compute_association_graph()` finds nearest max/min clusters to each zero-isocontour point
6. **Pruning** → Node pruning by scalar threshold, edge pruning by gradient weight (curvature vorticity proxy)
7. **Representative Path Extraction** → `rwp_graph.get_ranked_paths()` finds dominant paths

### 2.2 Faithful Implementation of Malavika's Tracking Extension

The tracking pipeline from Malavika's thesis is also implemented:

1. **Polygon footprints** → `rwp_polygon.get_polygon_for_rwp_path()` creates convex hulls in stereographic projection
2. **Rasterization** → `rwp_polygon.rasterize_all_rwps()` converts polygons to a 512×512 grid
3. **Quadtree construction** → `quadtree.create_quadtree()` builds the hierarchical spatial index
4. **Quadtree merging** → `quadtree.merge()` computes spatial overlap between timesteps
5. **Tracking graph** → `tracking_graph.build_tracking_graph()` links features across time via overlap weights
6. **Path extraction** → `tracking_graph.get_track_paths()` finds dominant temporal tracks

### 2.3 Upgrade to Spherical Geometry

The code has been partially upgraded to use spherical geometry via `geovista`:

```waper/waper/identification/utils.py#L17-L36
def get_vtk_object_from_data_array(data_array, lons, lats, array_name="v"):
    """Get vtk object from xarray dataArray

    Args:
        longitude (array): coordinates along zonal direction
        latitude (array): coordinates along meridional direction
        scalar_values (array): scalar field to convert to vtk object
    """

    grid = gv.Transform.from_1d(
        lons, lats, 
        data=data_array.data, name=array_name, radius=RADIUS_SPHERE, clean=False)
    
    mesh_lons, mesh_lats = np.meshgrid(lons, lats, indexing='xy')
    
    grid.cell_data['{} Cell Value'.format(array_name)] = grid.point_data_to_cell_data()[array_name]
    
    grid.point_data['Longitude'] = mesh_lons.ravel()
    grid.point_data['Latitude'] = mesh_lats.ravel()

    return grid
```

This maps lat/lon data onto a sphere of radius `RADIUS_SPHERE = 63.71` (a scaled-down Earth). The Dijkstra geodesic paths in `topology.py` then run on this spherical mesh, which is a genuine improvement over the flat-grid approach in the original Pandey code. The `haversine_distance` function in `utils.py` is also used for edge weight computation in `rwp_graph.py`.

**However**, the spherical geometry upgrade is *inconsistent* — several parts of the code still use Euclidean distance (see §3 below).

---

## 3. Issues with the Current Code

### 3.1 Critical Bugs

**A. Euclidean distance used in the association graph where spherical distance is needed**

In `compute_association_graph()`, the nearest max/min to each isocontour point is found using Cartesian distance on the 3D sphere coordinates:

```waper/waper/identification/rwp_graph.py#L100-L106
        for j in range(num_max_pts):
            max_point = max_points_array[j]
            curr_max_id = max_cluster_ids[j]
            max_dir_vector = [max_point[0] - contour_point[0], max_point[1] - contour_point[1]]
            curr_max_dist = (max_dir_vector[0] ** 2 + max_dir_vector[1] ** 2) ** 0.5
            if curr_max_dist < max_dist:
                max_dist = curr_max_dist
```

This computes the Euclidean distance of the **3D Cartesian** coordinates (x, y) on the sphere. While this is *monotonic* with geodesic distance for nearby points (and thus won't change nearest-neighbor results much), it's inconsistent with the spherical philosophy and breaks near the pole or date line. It should use the full 3D Euclidean distance `(dx² + dy² + dz²)^0.5` (which is a valid proxy for geodesic on the sphere), or haversine on lon/lat.

**B. The `min_id == 0` hack**

```waper/waper/identification/rwp_graph.py#L137-L139
        if min_id == 0:
            min_id = 100
```

This is a brittle workaround because node IDs for minima are negated (so cluster 0 would map to node `0`, colliding with max cluster 0). The code remaps `min_id = 0` to `min_id = 100` to avoid this collision. This will **break** if there are ≥100 max clusters (node ID collision) or if there happen to be exactly 100 min clusters. A much safer approach is to use string-typed node IDs like `("max", id)` / `("min", id)` or use a dedicated bipartite graph structure.

**C. The `check` array in `max_min.py` causes incorrect extrema detection**

```waper/waper/identification/max_min.py#L44-L50
            else:
                if j == 0:
                    for x in [i - 1, i, i + 1]:
                        for y in [c - 1, j, j + 1]:
                            if (0 <= x < r) and (0 <= y < c):
                                if numpy_data[x][y] > numpy_data[i][j]:
                                    max_flag = 0
                                else:
                                    check[x][y] = 1
```

The `check` array marks neighbors as "already visited" when they're found to be ≤ the current point. But this means a point that is a valid local maximum might be skipped if it was already marked `check == 1` by an earlier, smaller neighbor. This is a **correctness issue** — it can miss valid maxima and produce false negatives. The optimization is broken.

**D. Boundary handling in `max_min.py` — the `j == 0` and `j == c-1` cases are not mutually exclusive**

```waper/waper/identification/max_min.py#L43-L68
                if j == 0:
                    for x in [i - 1, i, i + 1]:
                        for y in [c - 1, j, j + 1]:
                            ...

                if j == c - 1:
                    for x in [i - 1, i, i + 1]:
                        for y in [j - 1, j, 0]:
                            ...

                else:
                    for x in [i - 1, i, i + 1]:
                        for y in [j - 1, j, j + 1]:
                            ...
```

When `j == 0`, the code enters the first `if` block, but then falls through to the `if j == c-1: ... else: ...` block as well (the second `if` is not `elif`). This means for `j == 0`, the point is checked **twice** — once with periodic wrap and once with the generic case `[j-1, j, j+1] = [-1, 0, 1]`. Since Python uses negative indexing, `numpy_data[x][-1]` wraps to the last column. This doesn't crash but introduces subtle double-checking and the `check` array gets set incorrectly.

**E. `is_to_the_east` returns `None` for westward**

```waper/waper/identification/utils.py#L71-L76
def is_to_the_east(lon1, lon2):
    
    delta_lat = lon1 - lon2
    
    if abs(delta_lat) > 180:
        delta_lat = -delta_lat
    
    if delta_lat > 0:
        return True
```

There's no `return False` — when `lon1` is to the west, the function implicitly returns `None`, which is falsy. This works accidentally but is a latent bug. Also, the variable is called `delta_lat` when it should be `delta_lon`.

### 3.2 Performance Bottlenecks

**A. O(N²) Python loops for extrema detection** — `max_min.py` iterates over every grid cell with nested Python `for` loops. On a typical ERA5 grid (721 × 1440 ≈ 1M points), this is extremely slow. The architecture document correctly identifies that `scipy.ndimage.maximum_filter`/`minimum_filter` would provide orders-of-magnitude speedup.

**B. VTK Dijkstra overhead** — `topology.py` builds heavy VTK mesh objects (`vtkGeometryFilter` → `vtkTriangleFilter`) and runs `vtkDijkstraGraphGeodesicPath` for every pair of extrema points. Each call involves full VTK pipeline updates. Using `scipy.sparse.csgraph.dijkstra` on a sparse adjacency matrix would be far faster.

**C. O(N²) loops in association graph** — `compute_association_graph()` iterates over every contour point × every max point × every min point. This is O(C × M) + O(C × N) where C can be very large (thousands of isocontour points). A KD-tree or ball-tree spatial index would reduce this dramatically.

**D. Exponential path enumeration** — Both `get_ranked_paths()` and `get_track_paths()` use `nx.all_simple_paths()`, which has factorial worst-case complexity. The architecture document correctly notes this should be replaced with dynamic programming on the DAG.

**E. Redundant merge computation** — In `build_tracking_graph()`, the quadtree `merge` is called once per timestep, but the edge list iterates over the Cartesian product of all features. The merge + feature-size computation could be done more efficiently.

### 3.3 Code Quality Issues

**A. Massive code duplication** — `cluster_max()` and `cluster_min()` in `topology.py` are nearly identical (~120 lines each) with trivially inverted comparisons. These should be unified into a single `cluster_extrema(sign=+1/-1)` function.

Similarly, `add_maxima_data()` and `add_minima_data()` are nearly identical. `interpolate_cell_values()` and `interpolate_cell_values_min()` are *literally identical*.

**B. Large blocks of commented-out code** — `max_min.py` has ~80 lines of commented-out code (old VTK-based extraction functions). `rwp_graph.py` has ~30 lines of commented-out gradient-based logic. These should be removed and tracked via version control.

**C. Magic numbers scattered throughout**

```waper/waper/identification/topology.py#L8
CLUSTER_MAX_DISTANCE = 150
```

```waper/waper/identification/rwp_graph.py#L6-L9
WAPER_MAX_SCALAR_VALUE = 100
WAPER_MAX_NODE_DISTANCE = 1000
WAPER_MIN_LON_DELTA = 6
```

```waper/waper/tracking/rwp_polygon.py#L9-L11
WAPER_SUBSAMPLE = 5
WAPER_IMAGE_SIZE = 512
WAPER_CLUSTER_WIDTH = 60
```

These should be part of `WaperConfig` and exposed to the user.

**D. No tests** — The only test file is:

```waper/tests/smoke_test.py#L1-2
def test_smoke_is_not_risingfrom_module():
    import my_new_project
```

This imports a module called `my_new_project` (not `waper`) — it's a template placeholder that was never updated. There are **zero** meaningful tests.

**E. Bare `except` clause**

```waper/waper/tracking/rwp_polygon.py#L38-L40
    try:
        return transformer.transform(input_xs, input_ys, errcheck=True)
    except:
        print(input_xs, input_ys)
        raise ValueError()
```

This catches all exceptions (including `KeyboardInterrupt`, `SystemExit`) and re-raises a generic `ValueError` with no message, destroying the original traceback.

**F. Type annotations are incomplete** — `WaperSingleTimestepData` declares type hints in its fields but many function signatures lack them entirely. The `py.typed` marker file is present, suggesting this package wants to support type checking, but very little is actually type-annotated.

---

## 4. Assessment of the Spherical Geometry Upgrade

The move to spherical geometry (via `geovista` for mesh construction and haversine for distances) is a significant and correct improvement over the original flat-grid approach. It addresses:

- **Polar distortion**: Near the poles, equirectangular grids have severely compressed meridians; spherical meshes handle this naturally.
- **Date-line wrapping**: The spherical mesh is inherently continuous, avoiding the 0°/360° boundary artifacts.
- **Physical distance metrics**: Haversine gives true great-circle distances.

**However**, the upgrade is incomplete:

1. The association graph's nearest-neighbor search still uses partial Euclidean distance (only x,y components, ignoring z).
2. The tracking module's polygon footprints use **North Pole stereographic projection** with hardcoded bounds:

```waper/waper/tracking/rwp_polygon.py#L15-L16
WAPER_X_BOUNDS = (12712833.087371958, -12712833.087371958)
WAPER_Y_BOUNDS = (12710532.145483922, -12713600.098850505)
```

This means the tracking only works for the Northern Hemisphere. The code has a `# TODO this must handle both north and south poles` comment but no implementation for the Southern Hemisphere.

3. The `RADIUS_SPHERE = 63.71` in `utils.py` is a scaled-down radius (Earth's radius in units of 100 km?), but `RADIUS_EARTH = 6.371e6` is also defined and used in haversine. This inconsistency between the VTK mesh radius and the haversine radius could introduce subtle distance-scale mismatches.

---

## 5. Assessment of the Visualization

The visualization module is functional and uses appropriate tools (Matplotlib + Cartopy for geographic projections, PyVista for 3D). It provides good diagnostic coverage:

- Cluster annotations on contour plots (`_plot_clusters`)
- Association/pruned graph overlays (`_plot_graph`)
- RWP path visualization with color coding (`_plot_rwp_paths`)
- Polygon footprints in stereographic projection (`_plot_polygons`)
- Raster images (`_plot_raster`)

**Issues:**

1. **Projection inconsistency**: Some plots use `PlateCarree`, others `Orthographic`, others `Stereographic`. The user can't control the projection.
2. **No coastlines or gridlines**: None of the plots call `ax.coastlines()` or `ax.gridlines()`, making it hard to orient geographically.
3. **Hardcoded figure layout**: `_plot_clusters` creates `plt.subplot(211, ...)` and `plt.subplot(212, ...)` internally, preventing integration into user figure layouts.
4. **Polygons are in stereographic coordinates**: The polygon plot uses stereographic coordinates natively, meaning the polygon boundaries are meaningful only in that projection — they can't be easily overlaid on a lat/lon map.
5. **The track visualization is minimal**: There's no Hovmöller diagram or time-longitude plot, which is standard for RWP analysis.

---

## 6. Recommended Improvements

### High Priority (Correctness)
1. **Fix extrema detection**: Replace the `max_min.py` nested loops with `scipy.ndimage.maximum_filter` / `minimum_filter`. This fixes the `check` array bug and the boundary handling bug simultaneously, while giving a 100–1000× speedup.
2. **Fix the node ID collision**: Replace the `min_id == 0 → 100` hack with typed node IDs like `("max", id)` and `("min", id)`.
3. **Fix `is_to_the_east`**: Add explicit `return False` and rename the variable to `delta_lon`.
4. **Use consistent distance metrics**: Use haversine or 3D Euclidean everywhere in the association graph.

### High Priority (Performance)
5. **Replace VTK Dijkstra with scipy sparse graph**: Build a sparse adjacency matrix from the grid connectivity and use `scipy.sparse.csgraph.shortest_path`.
6. **Use spatial indexing for nearest-neighbor queries**: Replace O(N²) loops in `compute_association_graph` with `scipy.spatial.cKDTree` or `sklearn.neighbors.BallTree`.
7. **Replace `all_simple_paths` with DAG longest path**: Both in `get_ranked_paths()` and `get_track_paths()`, use topological sort + DP.

### Medium Priority (Code Quality)
8. **Unify `cluster_max`/`cluster_min`**: Single function with a `sign` parameter.
9. **Unify `add_maxima_data`/`add_minima_data`**: Same approach.
10. **Remove all commented-out code** and dead functions like `interpolate_cell_values_min`.
11. **Move magic numbers to `WaperConfig`**: `CLUSTER_MAX_DISTANCE`, `WAPER_IMAGE_SIZE`, `WAPER_MIN_LON_DELTA`, etc.
12. **Add real tests**: Unit tests for extrema detection, clustering, graph construction, and integration tests for the full pipeline.
13. **Fix the smoke test**: It imports `my_new_project` instead of `waper`.

### Medium Priority (Features)
14. **Southern Hemisphere support**: Implement the `# TODO` for south-pole stereographic projection in `rwp_polygon.py`.
15. **Configurable projections in visualization**: Allow users to pass projection and figure/axes.
16. **Add coastlines and gridlines** to all map plots.
17. **Add Hovmöller diagram** for tracking visualization.

### Lower Priority (Architecture)
18. **Replace NetworkX quadtree with R-tree**: Use `shapely.strtree.STRtree` for spatial indexing, eliminating the rasterization step entirely.
19. **Consider xarray/dask integration**: For processing multiple timesteps in parallel.
20. **Add proper logging**: Replace `print()` statements with Python's `logging` module.
21. **Add `pyproject.toml` dependencies**: The `pyproject.toml` currently has no `[project.dependencies]` section — all dependencies are implicit via the conda `environment.yml`.

---

## Summary

WAPER faithfully implements the Pandey et al. (2020) identification algorithm and Malavika's tracking extension. The conceptual architecture is sound, and the spherical geometry upgrade is a genuine scientific improvement. However, the implementation suffers from several correctness bugs (extrema detection, node ID collisions, inconsistent distance metrics), severe performance bottlenecks (Python loops where vectorized/indexed operations should be used), significant code duplication, and a complete absence of tests. The visualization is functional but inflexible. Addressing the high-priority items above would transform this from a research prototype into a reliable, performant tool.

## User

Thanks. I'm also concerned with the behaviour of affinity propagation. because it requires all points to be part of a cluster, it ends up creating clusters where isolated points far away from the primary extrema are included, making the wave packet crest or trough look artificially large. Could you also see if you can identify similar high-level algorithmic issues, and add them to your assessment?

## Assistant



Good point about Affinity Propagation. Let me re-examine the clustering logic, the association graph construction, the path ranking, and the tracking overlap computation more carefully for algorithmic-level issues.

**Tool Call: Read file `waper/identification/topology.py` (lines 85-130)**
Status: Completed

```waper/waper/identification/topology.py
            
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

```


**Tool Call: Read file `waper/identification/topology.py` (lines 130-145)**
Status: Completed

```waper/waper/identification/topology.py
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

```


**Tool Call: Read file `waper/identification/rwp_graph.py` (lines 248-331)**
Status: Completed

```waper/waper/identification/rwp_graph.py

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
        
        lon_0 = assoc_graph.nodes[start_node]["coords"][0]
        lon_1 = assoc_graph.nodes[end_node]["coords"][0]
        
        if abs(lon_0 - lon_1) <= WAPER_MIN_LON_DELTA:
            continue
        

```


**Tool Call: Read file `waper/identification/rwp_graph.py` (lines 155-200)**
Status: Completed

```waper/waper/identification/rwp_graph.py
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

```


## Assistant

Now let me look more carefully at the path ranking logic, the polygon construction, and the tracking overlap:

**Tool Call: Read file `waper/identification/rwp_graph.py` (lines 340-410)**
Status: Completed

```waper/waper/identification/rwp_graph.py
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
                # best_path = None
                # max_weight = 0
                for path in nx.all_simple_paths(assoc_graph, source=source, target=sink):
                    # consistent = True
                    # for node in path[:-1]:
                    #     if is_to_the_east(
                    #         assoc_graph.nodes[node]["coords"][0], assoc_graph.nodes[path[-1]]["coords"][0]
                    #     ):
                    #         consistent = False
                    
                #     path_weight = get_path_weight(assoc_graph, path)
                    
                #     if path_weight > max_weight:
                #         max_weight = path_weight
                #         best_path = path
                  
                # if isinstance(best_path, list):
                #     # print(source, sink, best_path)
                #     path_list.append(best_path)
                
                    path_list.append(path)
                
    # return path_list

    path_wt_dict = {}

    # # print(len(path_list), "number of paths found")

    for path in path_list:
        curr_wt = 0
        # print(path)
        for i in range(len(path) - 1):

```


**Tool Call: Read file `waper/identification/rwp_graph.py` (lines 410-440)**
Status: Completed

```waper/waper/identification/rwp_graph.py
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

```


**Tool Call: Read file `waper/tracking/rwp_polygon.py` (lines 100-201)**
Status: Completed

```waper/waper/tracking/rwp_polygon.py

    return lons, lats, values


def get_polygon_for_rwp_path(path, assoc_graph, scalar_data, scalar_name, min_latitude, max_latitude):
    """Get bounding polygon for an identified RWP

    Args:
        path (list): list of nodes in each path
        assoc_graph (nx.Graph): association graph
        scalar_data (pv.PolyData): scalar field

    Returns:
        tuple: convex hull of points and polygon ID
    """

    path_max = -100
    for node in path:
        max_value = abs(assoc_graph.nodes[node]["scalar"])

        if max_value > path_max:
            path_max = max_value

    clip_threshold = path_max / 3.0

    max_clipped_region = topology.identify_connected_regions(
        scalar_data.clip_scalar(
            scalars=scalar_name, value=clip_threshold, invert=False
        ).clean()
    )

    min_clipped_region = topology.identify_connected_regions(
        scalar_data.clip_scalar(
            scalars=scalar_name, value=-clip_threshold, invert=True
        ).clean()
    )

    list_rwp_points = []
    list_lons = []
    list_lats = []
    list_values = []

    # min_lon = 360
    for node in path:
        if node > 0:
            out = get_region_points_and_values(
                assoc_graph, node, max_clipped_region, clip_threshold, scalar_name
            )
            if out:
                lons, lats, values = out
                
                valid_region = np.logical_and(lats >= min_latitude, lats <= max_latitude)
                lons = lons[valid_region]
                lats = lats[valid_region]
                values = values[valid_region]

                # if min_lon > np.min(lons):  # store location of most westward cluster.
                #     min_lon = np.min(lons)

                # lons = get_consistent_longitudes(lons, min_lon)
                list_lons.extend(lons)
                list_lats.extend(lats)
                list_values.extend(values)

                # lons = lons[::WAPER_SUBSAMPLE]
                # lats = lats[::WAPER_SUBSAMPLE]
                # list_rwp_points.extend(list(zip(lons, lats)))
            else:
                pass
        else:
            out = get_region_points_and_values(
                assoc_graph, node, min_clipped_region, clip_threshold, scalar_name
            )
            if out:
                lons, lats, values = out
                
                valid_region = np.logical_and(lats >= min_latitude, lats <= max_latitude)
                lons = lons[valid_region]
                lats = lats[valid_region]
                values = values[valid_region]

                # if min_lon > np.min(lons):  # store location of most westward cluster.
                #     min_lon = np.min(lons)

                # lons = get_consistent_longitudes(lons, min_lon)
                list_lons.extend(lons)
                list_lats.extend(lats)
                list_values.extend(values)

                # lons = lons[::WAPER_SUBSAMPLE]
                # lats = lats[::WAPER_SUBSAMPLE]
                # list_rwp_points.extend(list(zip(lons, lats)))

            else:
                pass

    polygon_id = round(path_max, 2)

    xs, ys = transform_to_stereographic(list_lons, list_lats)

    weighted_ys = np.average(ys, weights=np.abs(np.array(list_values)))
    weighted_xs = np.average(xs, weights=np.abs(np.array(list_values)))

```


## Assistant

Now let me look at the tracking overlap and the `preference` parameter more closely:

**Tool Call: Read file `waper/tracking/tracking_graph.py` (lines 30-80)**
Status: Completed

```waper/waper/tracking/tracking_graph.py
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

```


## Assistant

Now let me look at how the polygon footprint uses connectivity-based regions (the source of the "inflated footprint" problem you're hinting at):

**Tool Call: Read file `waper/tracking/rwp_polygon.py` (lines 67-100)**
Status: Completed

```waper/waper/tracking/rwp_polygon.py

    Args:
        assoc_graph (nx.Graph): Association Graph
        node (nx.Node): Node in the above graph
        clipped_region (pv.PolyData): scalar which includes connectivity information
        clip_threshold (float): Threshold at which scalar data is thresholded
        scalar_name (str): name of the scalar quantity

    Returns:
        tuple: coordinates of points close to node in graph
    """

    if abs(assoc_graph.nodes[node]["scalar"]) < clip_threshold:
        return None

    closest_point = clipped_region.find_closest_point(
        assoc_graph.nodes[node]["spherical_coords"]
    )
    region_id_node = clipped_region.point_data["RegionId"][closest_point]

    lons = clipped_region["Longitude"][clipped_region.point_data["RegionId"] == region_id_node]
    lats = clipped_region["Latitude"][clipped_region.point_data["RegionId"] == region_id_node]
    values = clipped_region.point_data[scalar_name][
        clipped_region.point_data["RegionId"] == region_id_node
    ]
    
    # node_latitude = assoc_graph.nodes[node]["coords"][1]
    
    # #TODO 3 should be a paramter
    # valid_region = np.logical_and(lats >= node_latitude-3, lats <= node_latitude+3)
    # lons = lons[valid_region]
    # lats = lats[valid_region]
    # values = values[valid_region]


```


## Assistant

Now let me look at how the two-point-per-region special case works and the preference parameter:

**Tool Call: Read file `waper/identification/topology.py` (lines 56-100)**
Status: Completed

```waper/waper/identification/topology.py
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

```


## Assistant

Now let me check how `polygon_id` is used as a feature identifier in the tracking — this looks like a potential issue:

**Tool Call: Read file `waper/tracking/rwp_polygon.py` (lines 186-201)**
Status: Completed

```waper/waper/tracking/rwp_polygon.py
                list_lats.extend(lats)
                list_values.extend(values)

                # lons = lons[::WAPER_SUBSAMPLE]
                # lats = lats[::WAPER_SUBSAMPLE]
                # list_rwp_points.extend(list(zip(lons, lats)))

            else:
                pass

    polygon_id = round(path_max, 2)

    xs, ys = transform_to_stereographic(list_lons, list_lats)

    weighted_ys = np.average(ys, weights=np.abs(np.array(list_values)))
    weighted_xs = np.average(xs, weights=np.abs(np.array(list_values)))

```


**Tool Call: Read file `waper/tracking/tracking_graph.py` (lines 35-42)**
Status: Completed

```waper/waper/tracking/tracking_graph.py
            for rwp_info in time_step_data[time].rwp_info.values():
                if abs(feature - rwp_info["rwp_id"]) < 1e-2:
                    lon = rwp_info["weighted_longitude"]
                    lat = rwp_info["weighted_latitude"]

            if lon == 0:
                print(feature)


```


## Assistant

Good — I've now traced through all the algorithmic details I needed. Let me also verify the edge pruning's longitude delta check:

**Tool Call: Read file `waper/identification/rwp_graph.py` (lines 316-340)**
Status: Completed

```waper/waper/identification/rwp_graph.py
        nx.Graph: association graph with low weight edges pruned
    """

    pruned_graph = nx.Graph()
    edges = [e for e in assoc_graph.edges()]

    for e in edges:
        start_node = e[0]
        end_node = e[1]
        
        lon_0 = assoc_graph.nodes[start_node]["coords"][0]
        lon_1 = assoc_graph.nodes[end_node]["coords"][0]
        
        if abs(lon_0 - lon_1) <= WAPER_MIN_LON_DELTA:
            continue
        
        if start_node >= 0:
            weight = edge_weight(assoc_graph, start_node, end_node)
        else:
            weight = edge_weight(assoc_graph, end_node, start_node)
        assoc_graph[start_node][end_node]["weight"] = weight

        if weight >= threshold and weight <= max_weight:
            pruned_graph.add_node(
                start_node,

```


## Assistant

Now I have a complete picture. Here's the expanded assessment with the high-level algorithmic issues.

---

## Addendum: High-Level Algorithmic Issues

Building on the previous assessment, here are the algorithmic-design-level problems — issues not with the code per se, but with how the algorithm's logic leads to physically incorrect or scientifically misleading results.

---

### Issue 1: Affinity Propagation Forces All Extrema Into Clusters (The Problem You Identified)

This is the most consequential algorithmic issue. In `topology.py`, within each connected component of the superlevel/sublevel set, **every** extremum must be assigned to some cluster:

```waper/waper/identification/topology.py#L130-L132
        af_clustering = cluster.AffinityPropagation(
            preference=np.full(num_cluster, median_dist / 5.0), affinity="precomputed"
        )
```

Affinity Propagation has no concept of an "outlier" or "noise" point. Every point is either an exemplar or is assigned to the nearest exemplar. Consider a connected region that spans a large latitude band — there may be a tight group of 5 strong maxima near 50°N forming a coherent crest, plus a single weak maximum at 35°N that happens to fall in the same connected component because the superlevel set is contiguous. AP will assign that outlier to the nearest cluster, pulling the cluster centroid southward and inflating the spatial footprint of the crest.

This propagates downstream in two critical ways:

1. **The cluster representative point** (`cluster_max_point`) is the point with the largest scalar value, but the cluster's membership (and hence its polygon footprint) now extends far from that point. The edge weight computation uses the representative's coordinates, but the polygon drawn for tracking uses *all* points in the connected region associated with the cluster — yielding a convex hull that is much larger than the actual meteorological feature.

2. **The preference parameter** `median_dist / 5.0` is a global constant applied uniformly to all regions. This means the clustering granularity doesn't adapt to the actual spatial structure of each region — a compact region with 3 close extrema and a sprawling region with 10 scattered extrema both get the same preference for how many clusters to form.

**Recommendation:** Switch to a density-based clustering algorithm like **DBSCAN** or **HDBSCAN**, which naturally identify noise points and don't force every point into a cluster. Extrema that don't have enough nearby neighbors would be classified as noise and excluded. The `eps` parameter in DBSCAN has a clear physical interpretation (maximum distance between points in the same crest/trough), which makes tuning more intuitive than AP's preference.

---

### Issue 2: Two-Point Regions Are Unconditionally Merged

```waper/waper/identification/topology.py#L112-L115
        if len(region_array[k]) == 2:
            cluster_assign[region_array[k][0]] = prev_max
            cluster_assign[region_array[k][1]] = prev_max
            prev_max += 1
            continue
```

When a connected region contains exactly two extrema, they are **always** placed in the same cluster regardless of how far apart they are. Two maxima 3000 km apart in a large superlevel set will be merged into a single "crest." This bypasses the distance-based similarity logic entirely and can create physically nonsensical clusters.

---

### Issue 3: Polygon Footprint Uses Entire Connected Region, Not Cluster Extent

In `rwp_polygon.get_region_points_and_values()`, the footprint for each node in the path is defined by finding the **entire connected region** in the clipped field that contains the node's representative point:

```waper/waper/tracking/rwp_polygon.py#L82-L90
    closest_point = clipped_region.find_closest_point(
        assoc_graph.nodes[node]["spherical_coords"]
    )
    region_id_node = clipped_region.point_data["RegionId"][closest_point]

    lons = clipped_region["Longitude"][clipped_region.point_data["RegionId"] == region_id_node]
    lats = clipped_region["Latitude"][clipped_region.point_data["RegionId"] == region_id_node]
```

This means the footprint captures the **full connected component** of the thresholded field, not just the area around the cluster's extrema. If the superlevel set at `path_max / 3.0` forms one large contiguous region spanning multiple crests, *every* node in the path that falls within it gets the same enormous region as its footprint. The resulting convex hull (used for tracking overlap) then covers far more area than the actual wave packet. There's even a commented-out latitude restriction that was apparently an attempt to fix this:

```waper/waper/tracking/rwp_polygon.py#L92-L96
    # node_latitude = assoc_graph.nodes[node]["coords"][1]
    
    # #TODO 3 should be a paramter
    # valid_region = np.logical_and(lats >= node_latitude-3, lats <= node_latitude+3)
```

This suggests the authors were aware of the problem but didn't settle on a solution.

---

### Issue 4: RWP Identity is Based on Peak Scalar Value — Not Unique

```waper/waper/tracking/rwp_polygon.py#L195
    polygon_id = round(path_max, 2)
```

The unique identifier for each RWP is `round(path_max, 2)` — the peak absolute scalar value rounded to 2 decimal places. This is used as the raster label and the key for tracking graph nodes. The problem is that **two different RWPs at the same timestep can have the same peak value** (e.g., two wave packets both peaking at 25.37 m/s). When this happens:

- The rasterizer overwrites one polygon's label with the other's (last-write-wins in `rasterio.features.rasterize`).
- The tracking graph loses a feature entirely.
- The feature lookup in `build_tracking_graph` uses `abs(feature - rwp_info["rwp_id"]) < 1e-2`, which will match the **first** one it finds, silently dropping the second.

This is a fundamental design flaw. Feature IDs should be based on a monotonic index, not a physical quantity.

---

### Issue 5: The Path Ranking Filter Has O(N²) Logic and Can Drop Valid Paths

The "top paths" filter in `get_ranked_paths()`:

```waper/waper/identification/rwp_graph.py#L415-L428
    top_paths = list(
        filter(
            lambda f: not any(
                [
                    (
                        path_wt_dict[tuple(f)] < path_wt_dict[tuple(g)]
                        and len(set(f) & set(g)) != 0
                    )
                    for g in path_list
                ]
            ),
            path_list,
        )
    )
```

This keeps a path only if no other path that shares any node with it has a higher weight. The intent is to select non-overlapping dominant paths. But the logic is not transitive: path A might dominate path B (sharing a node, higher weight), and path C might dominate path A (sharing a different node), but path B and C share no nodes. The result is that only C survives, even though B represented a valid, independent RWP that happened to share a single node with A. In essence, the algorithm greedily kills paths through an intermediate competitor rather than solving the proper **maximum-weight independent set** problem on the conflict graph.

The identical logic in `get_track_paths()` has the same flaw.

---

### Issue 6: The Association Graph Longitude Delta Pruning Is Not Wraparound-Aware

```waper/waper/identification/rwp_graph.py#L328-L330
        if abs(lon_0 - lon_1) <= WAPER_MIN_LON_DELTA:
            continue
```

This discards edges between clusters that are within 6° of longitude of each other, which is a physical filter to ensure alternating crests and troughs are sufficiently separated. However, it uses a naive `abs()` difference that doesn't handle the 0°/360° wraparound. Two clusters at 358° and 2° have an actual separation of 4° but `abs(358 - 2) = 356`, so the filter would *not* prune them (when it should). Conversely, the check is correct for most longitudes, but wrong at the date line — exactly the kind of boundary issue the spherical geometry upgrade was supposed to eliminate.

---

### Issue 7: The Cluster Representative Is the Peak-Value Point, Not the Centroid

In `topology.max_cluster_assign()`:

```waper/waper/identification/topology.py#L289-L294
    for i in range(num_points_max):
        x, y = max_points['Longitude'][i], max_points['Latitude'][i]
        coords = [x, y]
        max_pt_dict[cluster_id_max[i]].append(coords)
        if cluster_max_arr[cluster_id_max[i]] < max_scalars[i]:
            cluster_max_arr[cluster_id_max[i]] = max_scalars[i]
            cluster_max_point[cluster_id_max[i]][0] = max_points['Longitude'][i]
            cluster_max_point[cluster_id_max[i]][1] = max_points['Latitude'][i]
```

The cluster's representative location is the position of the **most extreme** point, not the weighted centroid of all extrema in the cluster. When a cluster contains many extrema scattered around a region (compounded by Issue 1), the representative point can be at the edge of the cluster rather than its center. This distorts:

- The edge weight calculation (gradient proxy uses the distance between representatives)
- The association graph edges (which represent adjacency between crests and troughs)
- The physical interpretation of the RWP graph

For a wave crest with 5 maxima, the "center of action" should be the value-weighted centroid, not the single grid point that happened to have the largest v.

---

### Issue 8: The Similarity Penalty for Maxima Clustering Uses a Questionable Metric

```waper/waper/identification/topology.py#L85-L91
            locator.FindCellsAlongLine(point_coords[i], point_coords[j], 0.001, cell_ids)
            for k in range(cell_ids.GetNumberOfIds()):
                if cell_v.GetTuple1(cell_ids.GetId(k)) < min_v:
                    min_v = cell_v.GetTuple1(cell_ids.GetId(k))
                    
            dist_matrix[i][j] = dist_matrix[i][j] - min_v
```

For maxima clustering, the pairwise distance is `geodesic_distance - min_cell_value_along_line`. Since `min_v` is initialized to 1000 and only updated downward, and since we're in the superlevel set (values > 0), `min_v` will be a positive number. So the penalty **decreases** the distance, making points seem closer. The intent is that if the scalar field stays high between two maxima, they should be more likely to cluster together.

But `FindCellsAlongLine` shoots a **straight line in 3D Cartesian space** through the sphere, not along the surface. Near the poles or for widely separated points, this line passes through the interior of the sphere and may not intersect the surface mesh cells at all. The result is `min_v` staying at its initialized value of 1000, which then massively deflates the distance. This can cause distant maxima to appear artificially close in the similarity matrix.

For minima, the analogous code adds `max_v` (the most positive value along the line), with `max_v` initialized to -1000 — the same problem in reverse.

---

### Issue 9: Convex Hull Is an Inappropriate Shape for RWP Footprints

```waper/waper/tracking/rwp_polygon.py#L200
    rwp_poly = MultiPoint(list(zip(xs, ys))).convex_hull
```

RWPs are elongated, zonally-extended features. The convex hull of all points from all crests and troughs in the path creates a single blob that fills in the concavities between alternating positive and negative regions. Two adjacent RWPs whose convex hulls overlap (especially at their zonal extents) will be matched during tracking even if they're physically distinct features. A concave hull (alpha shape) or separate per-node polygons with a union would better represent the actual footprint.

---

### Summary Table of Algorithmic Issues

| # | Issue | Impact | Severity |
|---|---|---|---|
| 1 | AP forces all extrema into clusters | Inflated clusters, wrong footprints | **High** |
| 2 | 2-point regions always merged | Clusters span arbitrary distances | **Medium** |
| 3 | Footprint = entire connected region | Tracking polygons far too large | **High** |
| 4 | Feature ID = rounded scalar value | Non-unique IDs, silent feature loss | **High** |
| 5 | Path ranking filter is non-transitive | Valid RWPs dropped via indirect competition | **Medium** |
| 6 | Lon-delta pruning ignores wraparound | Wrong pruning at date line | **Medium** |
| 7 | Cluster rep = peak point, not centroid | Distorted edge weights and graph structure | **Medium** |
| 8 | `FindCellsAlongLine` shoots through sphere interior | Bogus similarity penalties near poles | **High** |
| 9 | Convex hull merges crest/trough extents | Overlapping footprints, false tracking matches | **Medium** |

Issues 1, 3, and 4 are the ones most likely to cause visibly wrong scientific results. Issue 8 is insidious because it silently corrupts the clustering similarity matrix without any obvious symptom unless you inspect the raw distance values. The combination of Issues 1 + 3 is especially damaging: AP bloats the clusters, then the polygon footprint takes the entire connected region of each cluster, then the convex hull wraps the whole thing into an enormous blob — at every stage the feature gets larger than the physical wave packet.

## User

Thanks! could you create a detailed spec which captures all the issues and improvements that could be made, including testing, documentation, refactoring (vtk to pyvista), edge cases, algorithmic issues. Please provide a step by step way to tackle each of these so that other less capable LLMs can also follow through with the plan

## Assistant



Let me re-read the full codebase one more time to make sure I capture every edge

## User (checkpoint)

Thanks! could you create a detailed spec which captures all the issues and improvements that could be made, including testing, documentation, refactoring (vtk to pyvista), edge cases, algorithmic issues. Please provide a step by step way to tackle each of these so that other less capable LLMs can also follow through with the plan

## Assistant



Let me re-read key files one more time to make sure I capture every detail for the spec, particularly around VTK vs PyVista usage patterns, edge cases, and the full interface surface.

**Tool Call: Read file `waper/identification/max_min.py` (lines 1-90)**
Status: Completed

```waper/waper/identification/max_min.py
import numpy as np
import vtk
import pyvista as pv
from .utils import get_vtk_object_from_data_array


def add_maxima_data(scalar_values, scalar_name, longitudes, latitudes):
    """Identify maxima in scalar field

    Args:
        scalar_values (DataArray): the scalar field
        scalar_name (string): name of the scalar
        longitudes (np.array): longitude coordinates
        latitudes (np.array): latitude coordinates

    Returns:
        pv.PolyData: vtk object containing the scalar data and maxima
    """
    r, c = scalar_values.shape
    check = np.zeros((r, c))
    is_max = np.zeros((r, c))
    vertex_identifiers = np.zeros(r * c)
    
    lons = np.linspace(0, 360, len(longitudes))
    lats = latitudes

    grid_vtk = get_vtk_object_from_data_array(scalar_values, lons, lats, scalar_name)
    
    numpy_data = scalar_values.values

    count = 0
    k = 0

    for i in range(r):
        for j in range(c):

            vertex_identifiers[k] = k + 1
            k += 1
            max_flag = 1

            if check[i][j] == 1:
                continue

            else:
                if j == 0:
                    for x in [i - 1, i, i + 1]:
                        for y in [c - 1, j, j + 1]:
                            if (0 <= x < r) and (0 <= y < c):
                                if numpy_data[x][y] > numpy_data[i][j]:
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

                if j == c - 1:
                    for x in [i - 1, i, i + 1]:
                        for y in [j - 1, j, 0]:
                            if (0 <= x < r) and (0 <= y < c):
                                if numpy_data[x][y] > numpy_data[i][j]:
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

                else:
                    for x in [i - 1, i, i + 1]:
                        for y in [j - 1, j, j + 1]:
                            if (0 <= x < r) and (0 <= y < c):
                                if numpy_data[x][y] > numpy_data[i][j]:
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

            if max_flag == 1:
                is_max[i][j] = 1
                check[i][j] = 1
                count += 1

    cell_number = grid_vtk.GetNumberOfCells()
    cell_id = np.arange(cell_number)

    grid_vtk.point_data["is max"] = is_max.ravel()
    grid_vtk.point_data["Vertex_id"] = vertex_identifiers
    grid_vtk.cell_data["{} Cell ID".format(scalar_name)] = cell_id

    return grid_vtk

def add_minima_data(scalar_values, scalar_name, longitudes, latitudes):
    """Identify minima in scalar field

    Args:
        scalar_values (DataArray): the scalar field

```


**Tool Call: Read file `waper/identification/topology.py` (lines 1-10)**
Status: Completed

```waper/waper/identification/topology.py
import math
import numpy as np
import pyvista as pv
import vtk
from sklearn import cluster
from collections import defaultdict

CLUSTER_MAX_DISTANCE = 150



```


**Tool Call: Read file `waper/identification/utils.py`**
Status: Completed

```waper/waper/identification/utils.py
import geovista as gv
import vtk
import numpy as np
import math

RADIUS_SPHERE = 63.71
RADIUS_EARTH = 6.371e6

def get_point_data_label(scalar_name):
    return scalar_name

def get_cell_data_label(scalar_name):
    return "Cell Value {}".format(scalar_name)

def get_vtk_object_from_data_array(data_array, lons, lats, array_name="v"):
    """Get vtk object from xarray dataArray

    Args:
        longitude (array): coordinates along zonal direction
        latitude (array): coordinates along meridional direction
        scalar_values (array): scalar field to convert to vtk object
    """

    grid = gv.Transform.from_1d(
        lons, lats, 
        data=data_array.data, name=array_name, radius=RADIUS_SPHERE, clean=False)
    
    mesh_lons, mesh_lats = np.meshgrid(lons, lats, indexing='xy')
    
    grid.cell_data['{} Cell Value'.format(array_name)] = grid.point_data_to_cell_data()[array_name]
    
    grid.point_data['Longitude'] = mesh_lons.ravel()
    grid.point_data['Latitude'] = mesh_lats.ravel()

    return grid

def get_iso_contour(scalar_field, value, scalar_name):
    
    contour_filter = vtk.vtkContourFilter()
    scalar_field.GetPointData().SetScalars(scalar_field.GetPointData().GetArray(scalar_name))
    contour_filter.SetValue(0, value)
    contour_filter.SetInputData(scalar_field)
    contour_filter.Update()
    return contour_filter.GetOutput()


def compute_gradients(scalar_field, scalar_name):
    
    gradient_filter = vtk.vtkGradientFilter()
    scalar_field.GetPointData().SetScalars(scalar_field.GetPointData().GetArray(scalar_name))
    gradient_filter.SetInputData(scalar_field)
    gradient_filter.Update()
    return gradient_filter.GetOutput()

def haversine_distance(lat1, lon1, lat2, lon2):

    # circum = 2*np.pi*RADIUS_EARTH*np.cos(np.radians(30))

    dlat = math.radians(lat1 - lat2)

    dlon = math.radians(lon1 - lon2)

    a = (math.sin(dlat/2))**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * (math.sin(dlon/2))**2
    c = 2 * np.arctan2(math.sqrt(a), math.sqrt(1-a))
    distance = RADIUS_EARTH * c

    return distance

def is_to_the_east(lon1, lon2):
    
    delta_lat = lon1 - lon2
    
    if abs(delta_lat) > 180:
        delta_lat = -delta_lat
    
    if delta_lat > 0:
        return True
```


**Tool Call: Read file `waper/tracking/rwp_polygon.py` (lines 1-45)**
Status: Completed

```waper/waper/tracking/rwp_polygon.py
from ..identification import topology
import numpy as np
from pyproj.transformer import Transformer
import pyproj

from shapely.geometry import MultiPoint
from rasterio import features, Affine

WAPER_SUBSAMPLE = 5
WAPER_IMAGE_SIZE = 512
WAPER_CLUSTER_WIDTH = 60
WAPER_NUM_PIXELS = WAPER_IMAGE_SIZE*WAPER_IMAGE_SIZE

WAPER_X_BOUNDS = (12712833.087371958, -12712833.087371958)
WAPER_Y_BOUNDS = (12710532.145483922, -12713600.098850505)

WAPER_X_RES = (WAPER_X_BOUNDS[0] - WAPER_X_BOUNDS[1]) / WAPER_IMAGE_SIZE
WAPER_Y_RES = (WAPER_Y_BOUNDS[0] - WAPER_Y_BOUNDS[1]) / WAPER_IMAGE_SIZE

WAPER_RASTER_TRANSFORM = Affine.translation(
    WAPER_X_BOUNDS[1] - WAPER_X_RES / 2, WAPER_Y_BOUNDS[1] - WAPER_Y_RES / 2
) * Affine.scale(WAPER_X_RES, WAPER_Y_RES)

# TODO this must handle both north and south poles
def transform_to_stereographic(input_xs, input_ys, inverse=False):

    from_crs = pyproj.crs.CRS(4326)  # standard lat-lon
    to_crs = pyproj.crs.CRS("+proj=stere +lat_0=90 +lon_0=0")  # north pole stereographic
    if inverse:
        transformer = Transformer.from_crs(to_crs, from_crs, always_xy="True")
    else:
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy="True")

    try:
        return transformer.transform(input_xs, input_ys, errcheck=True)
    except:
        print(input_xs, input_ys)
        raise ValueError()


def get_consistent_longitudes(longitude_array, min_lon):
    """fix issue with wrap around of longitudes

    Args:
        longitude_array (list): list of longitudes

```


**Tool Call: Read file `waper/interface/api.py` (lines 1-50)**
Status: Completed

```waper/waper/interface/api.py
from dataclasses import dataclass

from pyvista import PolyData
from networkx import Graph
from xarray import DataArray
from tqdm import tqdm
from numpy import ndarray
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from ..identification import max_min, topology, rwp_graph, utils
from ..tracking import rwp_polygon
from .visualization import (
    _plot_clusters,
    _plot_graph,
    _plot_polygons,
    _plot_rwp_paths,
    _plot_raster,
)
from ..tracking import quadtree, tracking_graph
from waper import tracking
@dataclass(eq=False, frozen=True)
class WaperConfig:

    debug: bool
    scalar_name: str
    latitude_label: str
    longitude_label: str
    time_label: str

    clip_value: float
    extrema_threshold: float

    max_latitude: float
    min_latitude: float

    node_pruning_threshold: float
    edge_pruning_threshold: float
    max_edge_weight: float

    track_pruning_threshold: float

    vtk_latitude_label: str = "Latitude"
    vtk_longitude_label: str = "Longitude"
    vtk_region_label: str = "RegionId"


@dataclass(eq=False)
class WaperSingleTimestepData:

```


**Tool Call: Read file `waper/interface/visualization.py` (lines 1-10)**
Status: Completed

```waper/waper/interface/visualization.py
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pyproj import transform
import pyvista as pv
import numpy as np
from xarray import DataArray
from matplotlib.colors import LinearSegmentedColormap

from ..tracking.rwp_polygon import WAPER_X_BOUNDS, WAPER_Y_BOUNDS


```


**Tool Call: Read file `pyproject.toml`**
Status: Completed

```waper/pyproject.toml
# BUILD

[build-system]
requires = ["setuptools>=61.2", "setuptools-scm"]
build-backend = "setuptools.build_meta"


# Poetry

# Information required for building (sdist/wheel)
## Also renders on pypi as 'subtitle'
[project]
name = "waper"
version = "0.1.0"
description = "Rossby wave packet tracker"
authors = [{name="Malavika Biju"}, 
            {name="Joy Monteiro"},
            {name="Karran Pandey"}]
maintainers = [{name="Joy Monteiro", email="joy.merwin@gmail.com"}]
license = {file= "LICENSE"}
readme = "README.rst"
requires-python = ">= 3.9"

keywords = [
    "python package",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: End Users/Desktop",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU Affero General Public License v3",
    "Natural Language :: English",
    "Operating System :: Unix",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: System :: Software Distribution",
    "Typing :: Typed"
]

[tool.setuptools]
packages=["waper", "waper.identification", "waper.tracking", "waper.interface"]

#  [tool.setuptools.packages.find]
# {include = ["tests", 
#     "docs/**/*.rst",
#     "docs/conf.py", 
#     "src/**/*.typed", 
#     "src/stubs/*.pyi",
#     "pyproject.toml",
#     "LICENSE",
#     "README.rst",
#     "CONTRIBUTING.md",
#     "CHANGELOG.rst",
# ]},
# {exclude = [
#     ".github/*",
#     "docs/*",
#     "requirements/*",
#     "scripts/*.py",
#     "tox.ini",
#     ".bettercodehub.yml",
#     ".circleci/config.yml",
#     ".coveragerc",
#     ".DS_Store",
#     ".gitignore",
#     ".prospector.yml",
#     ".pylintrc",
#     ".readthedocs.yml",
#     ".scrutinizer.yml",
#     ".travis.yml"
# ]}]

# A list of all of the optional dependencies, some of which are included in the
# below `extras`. They can be opted into by apps.


# PyPi url links, that appear in 'Project Links' section
[project.urls]
"Bug Tracker" = "https://github.com/waper/issues"
"CI: Github Actions" = "https://github.com/JoyMonteiro/waper/actions"
"Documentation" = "https://waper.readthedocs.io/"
"Source Code" = "https://github.com/JoyMonteiro/waper"
"Changelog" = "https://github.com/JoyMonteiro/waper/blob/master/CHANGELOG.rst"
"Code of Conduct" = "https://github.com/JoyMonteiro/waper/blob/master/CONTRIBUTING.rst"
```


## Assistant

Now let me check what other files exist at the repo root for documentation and CI context:

**Tool Call: List the `waper` directory's contents**
Status: Completed

# Folders:
waper/waper/identification
waper/waper/interface
waper/waper/tracking

# Files:
waper/waper/__init__.py
waper/waper/py.typed


**Tool Call: 8 matches**
Status: Completed

file:///Users/joymonteiro/github/waper/CHANGELOG.rst

file:///Users/joymonteiro/github/waper/README.rst

file:///Users/joymonteiro/github/waper/docs/contents/10_introduction.rst

file:///Users/joymonteiro/github/waper/docs/contents/20_why_this_package.rst

file:///Users/joymonteiro/github/waper/docs/contents/30_usage.rst

file:///Users/joymonteiro/github/waper/docs/contents/40_modules.rst

file:///Users/joymonteiro/github/waper/docs/contents/my_new_project.rst

file:///Users/joymonteiro/github/waper/docs/index.rst

**Tool Call: 4 matches**
Status: Completed

file:///Users/joymonteiro/github/waper/CONTRIBUTING.md

file:///Users/joymonteiro/github/waper/conductor/architecture_and_algorithm.md

file:///Users/joymonteiro/github/waper/docs/malavika.md

file:///Users/joymonteiro/github/waper/docs/pandey.md

**Tool Call: 16 matches**
Status: Completed

file:///Users/joymonteiro/github/waper/docs

file:///Users/joymonteiro/github/waper/docs/An_algorithm_for_Identification_and_Tracking_of_Rossby_Wave_Packets.pdf

file:///Users/joymonteiro/github/waper/docs/Makefile

file:///Users/joymonteiro/github/waper/docs/Pandey et al. - 2020 - An Integrated Geometric and Topological Approach for the Identification and Visual Analysis of Rossb.pdf

file:///Users/joymonteiro/github/waper/docs/conf.py

file:///Users/joymonteiro/github/waper/docs/contents

file:///Users/joymonteiro/github/waper/docs/contents/10_introduction.rst

file:///Users/joymonteiro/github/waper/docs/contents/20_why_this_package.rst

file:///Users/joymonteiro/github/waper/docs/contents/30_usage.rst

file:///Users/joymonteiro/github/waper/docs/contents/40_modules.rst

file:///Users/joymonteiro/github/waper/docs/contents/my_new_project.rst

file:///Users/joymonteiro/github/waper/docs/index.rst

file:///Users/joymonteiro/github/waper/docs/make.bat

file:///Users/joymonteiro/github/waper/docs/malavika.md

file:///Users/joymonteiro/github/waper/docs/pandey.md

file:///Users/joymonteiro/github/waper/docs/spelling_wordlist.txt

**Tool Call: Read file `.github/workflows/test.yaml`**
Status: Completed

```waper/.github/workflows/test.yaml
name: Test Python Package
# Stress Test against different platforms and python versions

on:
  push:
    branches:
      - master
      - dev
      - ci
  pull_request:
    branches:
      - master
      - dev
    tags:
      - v*


jobs:
  test_suite:
    runs-on: ${{ matrix.platform }}
    strategy:
      matrix:
        platform: [ubuntu-latest, macos-latest]
        python-version: ["3.6", "3.7", "3.8", "3.9", "3.10", "3.11"]
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install tox tox-gh-actions

    - name: Do Lint Checking
      run: tox -e lint -vv -s false

    - name: Do Type Checking
      run: tox -e type -vv -s false

    - name: Specify current package version to assist build tool
      run: echo "PKG_VERSION=$(python scripts/parse_version.py)" >> $GITHUB_ENV

    - name: Run Test Suite
      run: tox -vv -s false
      env:
        PLATFORM: ${{ matrix.platform }}
    - name: Check for compliance with Python Best Practices
      run: |
        DIST_DIR=dist
        echo "DIST_DIR=dist" >> $GITHUB_ENV

        mkdir "$DIST_DIR"
        mv ".tox/${DIST_DIR}/cookiecutter_python-${PKG_VERSION}.tar.gz" "${DIST_DIR}"
        mv ".tox/${DIST_DIR}/cookiecutter_python-${PKG_VERSION}-py3-none-any.whl" "${DIST_DIR}"
        tox -e check -vv -s false

    - name: Upload Source & Wheel distributions as Artefacts
      uses: actions/upload-artifact@v2
      with:
        name: dist-${{ matrix.platform }}-${{ matrix.python-version }}
        path: ${{ env.DIST_DIR }}
        if-no-files-found: error

    - name: Aggregate Test Suite Coverage Data
      run: |
        python -m pip install coverage
        tox -e coverage --sitepackages -vv -s false

    - name: Upload Coverage Data as Artefacts
      uses: actions/upload-artifact@v2
      with:
        name: coverage-xml-data
        path: ./.tox/coverage.xml
        if-no-files-found: error

  host_coverage_results:
    needs: test_suite
    runs-on: ubuntu-latest
    steps:
    - name: Download Coverage Data
      uses: actions/download-artifact@v2
      with:
        name: coverage-xml-data
    - name: Send Coverage Data to Codecov
      uses: codecov/codecov-action@v2
      with:
        directory: .
        env_vars: OS,PYTHON
        fail_ci_if_error: true
        files: coverage.xml
        flags: unittests
        name: codecov-umbrella
        verbose: true

```


**Tool Call: Read file `docs/contents/30_usage.rst`**
Status: Completed

```waper/docs/contents/30_usage.rst
=====
Usage
=====

------------
Installation
------------

| **my_new_project** is available on PyPI hence you can use `pip` to install it.

It is recommended to perform the installation in an isolated `python virtual environment` (env).
You can create and activate an `env` using any tool of your preference (ie `virtualenv`, `venv`, `pyenv`).

Assuming you have 'activated' a `python virtual environment`:

.. code-block:: shell

  python -m pip install my-new-project


---------------
Simple Use Case
---------------

| Common Use Case for the my_new_project is to TODO Document

TODO Document

```


**Tool Call: Read file `docs/contents/40_modules.rst`**
Status: Completed

```waper/docs/contents/40_modules.rst
my_new_project
===================================================

.. toctree::
   :maxdepth: 4

   my_new_project

```


**Tool Call: Read file `README.rst`**
Status: Completed

```waper/README.rst
Waper

A rossby Wave Packet trackER
.. start-badges

| |build| |docs| |coverage| |maintainability| |better_code_hub| |tech-debt|
| |release_version| |wheel| |supported_versions| |gh-lic| |commits_since_specific_tag_on_master| |commits_since_latest_github_release|

|
| **Code:** https://github.com/JoyMonteiro/waper
| **Docs:** https://waper.readthedocs.io/en/master/
| **PyPI:** https://pypi.org/project/waper/
| **CI:** https://github.com/JoyMonteiro/waper/actions/


Features
========

1. **waper** `python package`

   a. TODO Document a **Great Feature**
   b. TODO Document another **Nice Feature**
2. Tested against multiple `platforms` and `python` versions


Development
-----------
Here are some useful notes related to doing development on this project.

1. **Test Suite**, using `pytest`_, located in `tests` dir
2. **Parallel Execution** of Unit Tests, on multiple cpu's
3. **Documentation Pages**, hosted on `readthedocs` server, located in `docs` dir
4. **Automation**, using `tox`_, driven by single `tox.ini` file

   a. **Code Coverage** measuring
   b. **Build Command**, using the `build`_ python package
   c. **Pypi Deploy Command**, supporting upload to both `pypi.org`_ and `test.pypi.org`_ servers
   d. **Type Check Command**, using `mypy`_
   e. **Lint** *Check* and `Apply` commands, using `isort`_ and `black`_
5. **CI Pipeline**, running on `Github Actions`_, defined in `.github/`

   a. **Job Matrix**, spanning different `platform`'s and `python version`'s

      1. Platforms: `ubuntu-latest`, `macos-latest`
      2. Python Interpreters: `3.6`, `3.7`, `3.8`, `3.9`, `3.10`
   b. **Parallel Job** execution, generated from the `matrix`, that runs the `Test Suite`


Prerequisites
=============

You need to have `Python` installed. The code has been tested only with `Python=3.10`

You will also need the following packages, all of which must be installed from the `conda-forge channel` (in a fresh environment preferably)

* `geovista` (also installs `pyvista`)
* `vtk`
* `xarray`
* `networkx`
* `rasterio`
* `scikit-learn`
* `tqdm`

Preferably install them all in a single command so that `mamba/conda` can figure out the optimal way to resolve dependencies.

Quickstart
==========

Using `pip` is the approved way for installing `waper`.

.. code-block:: sh

    python3 -m pip install waper


TODO Document a use case


License
=======

|gh-lic|

* `BSD 3-Clause License`_


License
=======

* Free software: BSD 3-Clause License



.. LINKS

.. _tox: https://tox.wiki/en/latest/

.. _pytest: https://docs.pytest.org/en/7.1.x/

.. _build: https://github.com/pypa/build

.. _pypi.org: https://pypi.org/

.. _test.pypi.org: https://test.pypi.org/

.. _mypy: https://mypy.readthedocs.io/en/stable/

.. _isort: https://pycqa.github.io/isort/

.. _black: https://black.readthedocs.io/en/stable/

.. _Github Actions: https://github.com/JoyMonteiro/waper/actions

.. _BSD 3-Clause License: https://github.com/JoyMonteiro/waper/blob/master/LICENSE


.. BADGE ALIASES

.. Build Status
.. Github Actions: Test Workflow Status for specific branch <branch>

.. |build| image:: https://img.shields.io/github/workflow/status/JoyMonteiro/waper/Test%20Python%20Package/master?label=build&logo=github-actions&logoColor=%233392FF
    :alt: GitHub Workflow Status (branch)
    :target: https://github.com/JoyMonteiro/waper/actions/workflows/test.yaml?query=branch%3Amaster


.. Documentation

.. |docs| image:: https://img.shields.io/readthedocs/waper/master?logo=readthedocs&logoColor=lightblue
    :alt: Read the Docs (version)
    :target: https://waper.readthedocs.io/en/master/

.. Code Coverage

.. |coverage| image:: https://img.shields.io/codecov/c/github/JoyMonteiro/waper/master?logo=codecov
    :alt: Codecov
    :target: https://app.codecov.io/gh/JoyMonteiro/waper

.. PyPI

.. |release_version| image:: https://img.shields.io/pypi/v/waper
    :alt: Production Version
    :target: https://pypi.org/project/waper/

.. |wheel| image:: https://img.shields.io/pypi/wheel/waper?color=green&label=wheel
    :alt: PyPI - Wheel
    :target: https://pypi.org/project/waper

.. |supported_versions| image:: https://img.shields.io/pypi/pyversions/waper?color=blue&label=python&logo=python&logoColor=%23ccccff
    :alt: Supported Python versions
    :target: https://pypi.org/project/waper

.. Github Releases & Tags

.. |commits_since_specific_tag_on_master| image:: https://img.shields.io/github/commits-since/JoyMonteiro/waper/v0.0.1/master?color=blue&logo=github
    :alt: GitHub commits since tagged version (branch)
    :target: https://github.com/JoyMonteiro/waper/compare/v0.0.1..master

.. |commits_since_latest_github_release| image:: https://img.shields.io/github/commits-since/JoyMonteiro/waper/latest?color=blue&logo=semver&sort=semver
    :alt: GitHub commits since latest release (by SemVer)

.. LICENSE (eg AGPL, MIT)
.. Github License

.. |gh-lic| image:: https://img.shields.io/github/license/JoyMonteiro/waper
    :alt: GitHub
    :target: https://github.com/JoyMonteiro/waper/blob/master/LICENSE


.. CODE QUALITY

.. Better Code Hub
.. Software Design Patterns

.. |better_code_hub| image:: https://bettercodehub.com/edge/badge/JoyMonteiro/waper?branch=master
    :alt: Better Code Hub
    :target: https://bettercodehub.com/


.. Code Climate CI
.. Code maintainability & Technical Debt

.. |maintainability| image:: https://img.shields.io/codeclimate/maintainability/JoyMonteiro/waper
    :alt: Code Climate Maintainability
    :target: https://codeclimate.com/github/JoyMonteiro/waper/maintainability

.. |tech-debt| image:: https://img.shields.io/codeclimate/tech-debt/JoyMonteiro/waper
    :alt: Technical Debt
    :target: https://codeclimate.com/github/JoyMonteiro/waper/maintainability

```


## Assistant

Now I have everything. Let me create the comprehensive spec document.

**Tool Call: conductor/waper_refactoring_spec.md**
Status: Completed

Diff: conductor/waper_refactoring_spec.md
```
# WAPER Refactoring Specification

## Document Purpose

This specification captures **every** known issue, improvement, and enhancement for the WAPER (Rossby Wave Packet Extraction and Representation) package. Each item is written as a self-contained task with explicit step-by-step instructions so that any developer — or a code-generation LLM — can pick up a single task and execute it without needing to understand the full history of the project.

**How to use this document:** Work through the phases in order. Within each phase, tasks can often be parallelised (noted where possible). Every task ends with a "Definition of Done" checklist. Do not skip the testing tasks — they are load-bearing.

---

## Table of Contents

- [Phase 0: Scaffolding, CI, and Project Hygiene](#phase-0-scaffolding-ci-and-project-hygiene)
- [Phase 1: Testing Infrastructure](#phase-1-testing-infrastructure)
- [Phase 2: Critical Bug Fixes](#phase-2-critical-bug-fixes)
- [Phase 3: Algorithmic Improvements — Identification](#phase-3-algorithmic-improvements--identification)
- [Phase 4: Algorithmic Improvements — Tracking](#phase-4-algorithmic-improvements--tracking)
- [Phase 5: VTK-to-PyVista / SciPy Refactor](#phase-5-vtk-to-pyvista--scipy-refactor)
- [Phase 6: Performance Optimisation](#phase-6-performance-optimisation)
- [Phase 7: Visualisation Overhaul](#phase-7-visualisation-overhaul)
- [Phase 8: Documentation](#phase-8-documentation)
- [Phase 9: Stretch Goals](#phase-9-stretch-goals)
- [Appendix A: File Inventory](#appendix-a-file-inventory)
- [Appendix B: Magic Numbers Registry](#appendix-b-magic-numbers-registry)
- [Appendix C: Dependency Map](#appendix-c-dependency-map)

---

## Phase 0: Scaffolding, CI, and Project Hygiene

These tasks have **zero** algorithmic risk and can be done first to establish a clean working environment.

### Task 0.1 — Fix `pyproject.toml` Metadata and Dependencies

**Problem:** `pyproject.toml` has no `[project.dependencies]` section. All dependencies are implicit via the conda `environment.yml`. The CI workflow references a package called `cookiecutter_python`, not `waper`. Python 3.6/3.7 are listed in the CI matrix but the project requires `>= 3.9`.

**Files to edit:**
- `pyproject.toml`
- `.github/workflows/test.yaml`

**Steps:**

1. Open `pyproject.toml`.
2. Add a `[project.dependencies]` section listing all runtime dependencies with minimum versions:
   ```
   [project.dependencies]
   numpy >= 1.22
   scipy >= 1.9
   xarray >= 2022.6
   networkx >= 2.8
   pyvista >= 0.36
   geovista >= 0.4
   vtk >= 9.1
   scikit-learn >= 1.1
   shapely >= 1.8
   rasterio >= 1.3
   pyproj >= 3.4
   cartopy >= 0.21
   matplotlib >= 3.6
   tqdm >= 4.64
   ```
3. Add `[project.optional-dependencies]` for dev/test:
   ```
   [project.optional-dependencies]
   dev = ["pytest >= 7.0", "pytest-cov", "mypy", "ruff"]
   ```
4. Remove Python 3.6, 3.7, 3.8 from the `classifiers` list and the CI matrix. The minimum should be `3.9`.
5. In `.github/workflows/test.yaml`, replace all references to `cookiecutter_python` with `waper`. Update the matrix to `["3.9", "3.10", "3.11", "3.12"]`. Replace the install step with:
   ```yaml
   - name: Install package
     run: python -m pip install -e ".[dev]"
   ```
6. Remove the `environment.yml` file (or keep it as a convenience but document that `pyproject.toml` is the source of truth).

**Definition of Done:**
- [ ] `pip install -e ".[dev]"` succeeds in a fresh venv.
- [ ] CI workflow references `waper` everywhere, not `cookiecutter_python` or `my_new_project`.
- [ ] Python version matrix is 3.9–3.12.

---

### Task 0.2 — Remove Dead Code and Commented-Out Blocks

**Problem:** Multiple files contain large blocks of commented-out code and dead functions. This obscures the actual logic and confuses any reader or LLM.

**Files to edit:**
- `waper/identification/max_min.py`
- `waper/identification/rwp_graph.py`
- `waper/identification/topology.py`
- `waper/tracking/tracking_graph.py`

**Steps:**

1. In `max_min.py`:
   - Delete the entire commented-out function `clip_dataset_min` (approx lines 230–248).
   - Delete the commented-out functions `extract_position_ids_minima`, `extract_position_ids_maxima`, `extract_selection_ids_maxima`, `extract_selection_ids_minima` (approx lines 177–264, everything that is commented out).
   - Delete the function `interpolate_cell_values_min` — it is an exact duplicate of `interpolate_cell_values`. Search the codebase for any calls to `interpolate_cell_values_min` and replace them with `interpolate_cell_values`. (At time of writing there are zero calls.)

2. In `rwp_graph.py`:
   - In `compute_association_graph`, remove all commented-out lines referencing `grad_vector`, `curr_max_dir_deriv`, `curr_min_dir_deriv`, `curr_max_scalar`, `curr_min_scalar`, `curr_max_x`, `curr_min_x`, `point_cords_max`, `point_cords_min`, `point_tuple_max`, `point_tuple_min`, `assoc_index_array`, `line_dir_array`.
   - In `edge_weight`, remove all commented-out lines referencing `high_value_threshold`, `scalar_threshold`, `scalar_tolerance`, `high_value_flag`, `cluster_max_pts`, `cluster_min_pts`, and the commented-out nested loop over cluster points.
   - In `get_ranked_paths`, remove the commented-out `best_path` / `max_weight` / `consistent` logic block and the commented-out `return path_list`.

3. In `topology.py`:
   - In `identify_connected_regions`, remove the commented-out VTK connectivity filter block.

4. In `tracking_graph.py`:
   - Remove all commented-out `print` statements.
   - Remove the commented-out `return track_paths` at the end of `get_track_paths`.

5. Run `git diff` to review all removals are comments/dead code only.

**Definition of Done:**
- [ ] No commented-out function bodies remain in any `.py` file.
- [ ] No `# print(...)` lines remain.
- [ ] `interpolate_cell_values_min` no longer exists.
- [ ] Package still imports correctly (`python -c "from waper import Waper"`).

---

### Task 0.3 — Fix `__init__.py` and Smoke Test

**Problem:** `waper/__init__.py` exposes internal submodules (`max_min`, `topology`) at the top level, which is unusual. The smoke test imports `my_new_project`, not `waper`.

**Files to edit:**
- `waper/__init__.py`
- `waper/interface/__init__.py`
- `tests/smoke_test.py`

**Steps:**

1. Edit `waper/__init__.py` to:
   ```python
   from .interface.api import Waper, WaperConfig, WaperSingleTimestepData

   __all__ = ["Waper", "WaperConfig", "WaperSingleTimestepData"]
   ```

2. Edit `waper/interface/__init__.py` to:
   ```python
   from .api import Waper, WaperConfig, WaperSingleTimestepData

   __all__ = ["Waper", "WaperConfig", "WaperSingleTimestepData"]
   ```

3. In `api.py`, remove the line `from waper import tracking` (absolute import that is fragile and unused directly). The existing relative import `from ..tracking import quadtree, tracking_graph` already covers it.

4. Edit `tests/smoke_test.py` to:
   ```python
   def test_smoke_import():
       import waper
       assert waper is not None

   def test_smoke_classes_exist():
       from waper import Waper, WaperConfig
       assert Waper is not None
       assert WaperConfig is not None
   ```

5. Run the smoke test: `pytest tests/smoke_test.py -v`.

**Definition of Done:**
- [ ] `from waper import Waper, WaperConfig` works.
- [ ] `pytest tests/smoke_test.py` passes.
- [ ] No absolute `from waper import ...` inside the package source (only relative imports).

---

### Task 0.4 — Add Logging Framework

**Problem:** Debugging output uses bare `print()` statements scattered throughout. There is a `logging()` function in `api.py` that just calls `print()` and is never used.

**Files to edit:**
- `waper/interface/api.py`
- All files that currently contain `print()` calls.

**Steps:**

1. Delete the standalone `logging` function in `api.py` (the one that takes `log_info, config`).

2. At the top of every module that needs logging, add:
   ```python
   import logging
   logger = logging.getLogger(__name__)
   ```

3. Replace every bare `print(...)` call with an appropriate log level:
   - `print(feature)` in `tracking_graph.py` → `logger.warning("Feature %s has no matching rwp_info", feature)`
   - `print(input_xs, input_ys)` in `rwp_polygon.py` → `logger.error("Stereographic transform failed for xs=%s, ys=%s", input_xs, input_ys)`
   - `print('No RWPs found, change thresholds')` in `api.py` → `logger.warning("No RWPs found at this timestep. Consider adjusting thresholds.")`

4. In the `Waper.__init__` method, add:
   ```python
   if debug:
       logging.basicConfig(level=logging.DEBUG)
   ```

**Definition of Done:**
- [ ] Zero bare `print()` calls remain in `waper/` source.
- [ ] `Waper(debug=True, ...)` produces log output.
- [ ] `Waper(debug=False, ...)` is silent by default.

---

## Phase 1: Testing Infrastructure

These tasks create the test harness that all subsequent phases depend on. **Do this before any algorithmic changes.**

### Task 1.1 — Create Synthetic Test Data Fixtures

**Problem:** There are no tests. Before we can safely refactor, we need deterministic test inputs with known answers.

**Files to create:**
- `tests/conftest.py`
- `tests/fixtures/` directory

**Steps:**

1. Create `tests/conftest.py`.

2. Write a pytest fixture `simple_wave_field` that generates a synthetic 2D meridional wind field as an `xarray.DataArray`:
   ```python
   import numpy as np
   import xarray as xr
   import pytest

   @pytest.fixture
   def simple_wave_field():
       """A synthetic v-wind with 3 clear crests and 2 troughs.

       The field is:  v(lon, lat) = A(lon) * sin(k * lon)
       where A(lon) is a Gaussian envelope centered at lon=180
       and k gives ~4 full wavelengths across 360 degrees.

       Latitudes span 20N to 80N. Longitudes span 0 to 359.
       """
       lons = np.arange(0, 360, 2.5)       # 144 points
       lats = np.arange(20, 80.1, 2.5)     # 25 points
       lon2d, lat2d = np.meshgrid(lons, lats)

       k = 2 * np.pi * 4 / 360  # wavenumber 4
       envelope = 30 * np.exp(-((lon2d - 180) ** 2) / (2 * 40 ** 2))
       v = envelope * np.sin(k * np.radians(lon2d) * 360)

       da = xr.DataArray(
           v, dims=["latitude", "longitude"],
           coords={"latitude": lats, "longitude": lons},
       )
       return da
   ```

3. Write a fixture `two_timestep_field` that returns a dataset with 2 timesteps where the wave packet has shifted ~5° east between them. This is for tracking tests.

4. Write a fixture `single_maximum_field` that is a single isolated Gaussian bump (one clear maximum, no minima above threshold). This is for edge-case tests.

5. Write a fixture `flat_field` that is identically zero everywhere. This is for testing graceful failure.

6. Write a fixture `date_line_wave_field` where the wave packet straddles the 0°/360° boundary. This tests wraparound handling.

**Definition of Done:**
- [ ] `tests/conftest.py` exists with at least 5 fixtures.
- [ ] Each fixture returns an `xarray.DataArray` with latitude/longitude coordinates.
- [ ] `pytest tests/conftest.py --collect-only` shows all fixtures.

---

### Task 1.2 — Unit Tests for Extrema Detection

**File to create:** `tests/test_max_min.py`

**Steps:**

1. Write `test_finds_known_maxima`: Using `simple_wave_field`, call `add_maxima_data` and `extract_maxima_points`. Assert that the number of maxima found equals the expected count (manually compute from the synthetic formula). Assert that all maxima have scalar values > 0.

2. Write `test_finds_known_minima`: Same as above but for minima. Assert values < 0.

3. Write `test_threshold_filters_weak_extrema`: Create a field with one strong max (v=30) and one weak max (v=3). Set `extrema_threshold=5`. Assert only the strong maximum survives extraction.

4. Write `test_periodic_boundary_maxima`: Using `date_line_wave_field`, place a maximum at lon=359. Assert it is correctly detected and that the neighbor comparison wraps to lon=0.

5. Write `test_flat_field_no_extrema`: Using `flat_field`, assert zero maxima and zero minima are found.

6. Write `test_maxima_and_minima_do_not_overlap`: For any test field, assert that no grid point is flagged as both a maximum and a minimum.

**Definition of Done:**
- [ ] `pytest tests/test_max_min.py -v` passes.
- [ ] At least 6 test functions exist.

---

### Task 1.3 — Unit Tests for Clustering

**File to create:** `tests/test_clustering.py`

**Steps:**

1. Write `test_single_extremum_per_region_is_own_cluster`: Create a clipped field with one connected component containing one maximum. Assert it gets cluster ID 0.

2. Write `test_two_close_extrema_same_cluster`: Create a clipped field with two maxima 5° apart in the same connected component. Assert they receive the same cluster ID.

3. Write `test_two_distant_extrema_different_clusters`: Two maxima 60° apart in the same connected component. Assert they receive different cluster IDs.

4. Write `test_isolated_outlier_far_from_group`: Place 5 maxima in a tight group plus 1 maximum 40° away. Assert the outlier is in a different cluster from the group (this will currently FAIL — it documents the AP forcing issue and will be the regression test for Phase 3).

**Definition of Done:**
- [ ] `pytest tests/test_clustering.py -v` runs (some tests may be marked `xfail` until Phase 3).
- [ ] At least 4 test functions exist.

---

### Task 1.4 — Unit Tests for Association Graph

**File to create:** `tests/test_association_graph.py`

**Steps:**

1. Write `test_alternating_crests_troughs_connected`: Using `simple_wave_field`, run the full identification pipeline up to the association graph. Assert that the graph is bipartite (all edges connect a positive-ID node to a negative-ID node).

2. Write `test_isolated_max_no_adjacent_min`: Create a field with one crest but no trough above threshold. Assert the association graph is empty.

3. Write `test_node_pruning_removes_weak_nodes`: Build an association graph, then prune. Assert nodes with scalar below threshold are gone.

4. Write `test_edge_pruning_removes_low_gradient`: Build and prune. Assert only edges above the gradient threshold survive.

5. Write `test_date_line_association`: Using `date_line_wave_field`, assert that crests/troughs near 0°/360° are correctly associated.

**Definition of Done:**
- [ ] `pytest tests/test_association_graph.py -v` passes (some may be `xfail`).
- [ ] At least 5 test functions exist.

---

### Task 1.5 — Unit Tests for Tracking

**File to create:** `tests/test_tracking.py`

**Steps:**

1. Write `test_identical_timesteps_full_overlap`: Run identification on the same field twice. Build tracking graph. Assert overlap weight is 1.0 for matching features.

2. Write `test_shifted_field_partial_overlap`: Using `two_timestep_field`, assert overlap weight is between 0 and 1.

3. Write `test_no_overlap_no_edge`: Two timesteps with features in completely different hemispheres. Assert no edges in tracking graph.

4. Write `test_tracking_path_extraction`: Build a tracking graph with 3 timesteps. Assert the extracted path spans all 3.

5. Write `test_quadtree_pixel_counts`: Create a known raster, build quadtree, call `compute_pixels`. Assert pixel counts match expected values.

**Definition of Done:**
- [ ] `pytest tests/test_tracking.py -v` passes (some may be `xfail`).
- [ ] At least 5 test functions exist.

---

### Task 1.6 — Integration Test

**File to create:** `tests/test_integration.py`

**Steps:**

1. Write `test_full_pipeline_synthetic`: Instantiate `Waper` with `simple_wave_field` (duplicated to 2 timesteps). Call `identify_rwps()`. Assert at least 1 RWP path is found. Call `track_rwps()`. Assert no crash. This is the canary test that the whole pipeline runs end-to-end.

2. Write `test_full_pipeline_flat_field_graceful`: Instantiate with `flat_field`. Assert `identify_rwps()` completes without crash and finds 0 RWPs.

**Definition of Done:**
- [ ] `pytest tests/test_integration.py -v` passes.
- [ ] Tests run in < 60 seconds on a modern laptop.

---

## Phase 2: Critical Bug Fixes

These are correctness issues that should be fixed before any refactoring. Each fix should be accompanied by a test that would have caught the bug.

### Task 2.1 — Fix Extrema Detection (`max_min.py`)

**Problem 1 — `check` array causes missed extrema:** When point (i,j) is compared to its neighbors, any neighbor that is ≤ the current value gets `check[x][y] = 1`, marking it as "visited." But that neighbor might itself be a valid local maximum relative to *its own* neighbors. The `check` optimisation is incorrect and causes false negatives.

**Problem 2 — `if/if/else` instead of `if/elif/else` for boundary handling:** The `j == 0` block is an `if`, the `j == c-1` block is also an `if` (not `elif`), and the generic case is `else` attached to the second `if`. This means when `j == 0`, the code enters the first block AND then falls through to the `else` block (since `j == 0` is not `c-1`). The point is checked twice with inconsistent neighbor sets.

**Files to edit:**
- `waper/identification/max_min.py`

**Steps:**

1. **Replace the entire body** of `add_maxima_data` with a vectorised implementation:
   ```python
   from scipy.ndimage import maximum_filter

   def add_maxima_data(scalar_values, scalar_name, longitudes, latitudes):
       lons = np.linspace(0, 360, len(longitudes))
       lats = latitudes
       grid_vtk = get_vtk_object_from_data_array(scalar_values, lons, lats, scalar_name)

       numpy_data = scalar_values.values
       r, c = numpy_data.shape

       # maximum_filter with wrap mode on the longitude axis handles periodicity
       local_max = maximum_filter(
           numpy_data, size=3, mode=['constant', 'wrap']
       )
       is_max = (numpy_data == local_max).astype(float)

       # The above finds plateaux too; keep only strict local maxima
       # (equal to the filter output AND not a flat region)
       # For a flat region, all neighbors equal the center, so local_max == center everywhere.
       # We accept this as a maximum only if the value is nonzero.
       # The threshold filtering downstream will remove insignificant ones.

       vertex_identifiers = np.arange(1, r * c + 1, dtype=float)

       cell_number = grid_vtk.GetNumberOfCells()
       cell_id = np.arange(cell_number)

       grid_vtk.point_data["is max"] = is_max.ravel()
       grid_vtk.point_data["Vertex_id"] = vertex_identifiers
       grid_vtk.cell_data["{} Cell ID".format(scalar_name)] = cell_id

       return grid_vtk
   ```

2. **Do the same for `add_minima_data`** using `minimum_filter`:
   ```python
   from scipy.ndimage import minimum_filter

   def add_minima_data(scalar_values, scalar_name, longitudes, latitudes):
       lons = np.linspace(0, 360, len(longitudes))
       lats = latitudes
       grid_vtk = get_vtk_object_from_data_array(scalar_values, lons, lats, scalar_name)

       numpy_data = scalar_values.values
       r, c = numpy_data.shape

       local_min = minimum_filter(
           numpy_data, size=3, mode=['constant', 'wrap']
       )
       is_min = (numpy_data == local_min).astype(float)

       # Exclude the top row (i == 0), matching original behavior
       is_min[0, :] = 0

       vertex_identifiers = np.arange(1, r * c + 1, dtype=float)

       cell_number = grid_vtk.GetNumberOfCells()
       cell_id = np.arange(cell_number)

       grid_vtk.point_data["is min"] = is_min.ravel()
       grid_vtk.point_data["Vertex_id"] = vertex_identifiers
       grid_vtk.cell_data["{} Cell ID".format(scalar_name)] = cell_id

       return grid_vtk
   ```

3. Note on `mode` parameter: `['constant', 'wrap']` means axis 0 (latitude) uses constant padding (no wrap at poles) and axis 1 (longitude) uses wrap (periodic boundary). This exactly matches the intended behavior.

4. Run `pytest tests/test_max_min.py -v` — all tests from Task 1.2 should pass.

5. Run `pytest tests/test_integration.py -v` — the full pipeline should still work.

**Definition of Done:**
- [ ] No Python `for` loops remain in `add_maxima_data` or `add_minima_data`.
- [ ] `test_periodic_boundary_maxima` passes.
- [ ] `test_flat_field_no_extrema` passes.
- [ ] `test_maxima_and_minima_do_not_overlap` passes.
- [ ] Integration test passes.

---

### Task 2.2 — Fix Node ID Collision (`min_id == 0` Hack)

**Problem:** In `compute_association_graph`, minima cluster IDs are negated to distinguish them from maxima cluster IDs. Cluster ID 0 maps to node 0, which collides with max cluster 0. The code hacks around this by remapping `min_id = 0` to `min_id = 100`, which breaks if there are ≥100 max clusters.

**Files to edit:**
- `waper/identification/rwp_graph.py`

**Steps:**

1. Change the node ID scheme from plain integers to tuples. A max cluster with ID `k` becomes node `("max", k)`. A min cluster with ID `k` becomes node `("min", k)`.

2. In `compute_association_graph`:
   - Remove the `if min_id == 0: min_id = 100` block and the `if min_id == 100:` special case.
   - When adding nodes:
     ```python
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
     ```

3. In `prune_association_graph_nodes`:
   - Replace `if start_node >= 0:` with `if assoc_graph.nodes[start_node]["node_type"] == "max":`.

4. In `edge_weight`:
   - The function currently receives `max_id` and `min_id` — these are now tuple node IDs. The body doesn't depend on the sign of the ID, only on `assoc_graph.nodes[max_id]["scalar"]`, so no change needed to the body.

5. In `prune_association_graph_edges`:
   - Replace `if start_node >= 0:` with `if assoc_graph.nodes[start_node]["node_type"] == "max":`.

6. In `get_ranked_paths`:
   - The `is_to_the_east` call uses `assoc_graph.nodes[source]["coords"][0]` which doesn't depend on node ID format. No change needed.

7. In `rwp_polygon.get_polygon_for_rwp_path`:
   - Replace `if node > 0:` with `if node[0] == "max":`.

8. In `visualization.py` `_plot_rwp_paths`:
   - Replace `if node < 0:` with:
     ```python
     if isinstance(node, tuple):
         color = 'r' if node[0] == 'max' else 'b'
     else:
         color = 'r' if node >= 0 else 'b'
     ```
     (The `else` branch handles the tracking graph, which uses `(time, feature)` tuples with a different structure.)

9. In `visualization.py` `_plot_clusters`:
   - Replace the `if cluster_id == 0: cluster_id = 100` block with just a plain negative sign for display: `str(-cluster_id)`.

10. Search the entire codebase for any remaining `if ... >= 0` or `if ... < 0` or `if ... > 0` checks on node IDs and update them.

**Definition of Done:**
- [ ] The string `min_id = 100` no longer appears anywhere.
- [ ] No node ID is a plain integer in the association graph.
- [ ] `test_alternating_crests_troughs_connected` passes.
- [ ] Integration test passes.

---

### Task 2.3 — Fix `is_to_the_east` Missing Return and Wrong Variable Name

**Problem:** The function returns `True` when `lon1` is east of `lon2`, but implicitly returns `None` (falsy) otherwise. Variable named `delta_lat` should be `delta_lon`.

**File to edit:**
- `waper/identification/utils.py`

**Steps:**

1. Replace the function with:
   ```python
   def is_to_the_east(lon1, lon2):
       """Return True if lon1 is to the east of lon2, handling wraparound."""
       delta_lon = lon1 - lon2

       if abs(delta_lon) > 180:
           delta_lon = -delta_lon

       return delta_lon > 0
   ```

2. Write a test in `tests/test_utils.py`:
   ```python
   from waper.identification.utils import is_to_the_east

   def test_east_simple():
       assert is_to_the_east(10, 5) is True

   def test_west_simple():
       assert is_to_the_east(5, 10) is False

   def test_same_longitude():
       assert is_to_the_east(10, 10) is False

   def test_wraparound_east():
       assert is_to_the_east(5, 355) is True   # 5° is 10° east of 355°

   def test_wraparound_west():
       assert is_to_the_east(355, 5) is False
   ```

**Definition of Done:**
- [ ] `is_to_the_east` always returns a `bool`.
- [ ] Variable is named `delta_lon`.
- [ ] All 5 tests pass.

---

### Task 2.4 — Fix Euclidean Distance in Association Graph

**Problem:** In `compute_association_graph`, the nearest max/min to each isocontour point is found using 2D Euclidean distance on (x, y) components of the 3D spherical mesh coordinates, ignoring z. This is inconsistent with the spherical geometry intent and can give wrong nearest-neighbor results near poles.

**File to edit:**
- `waper/identification/rwp_graph.py`

**Steps:**

1. Import at the top of the file:
   ```python
   from scipy.spatial import cKDTree
   ```

2. In `compute_association_graph`, before the contour-point loop, build KD-trees on the full 3D Cartesian coordinates:
   ```python
   max_tree = cKDTree(max_points_array)  # shape (num_max_pts, 3)
   min_tree = cKDTree(min_points_array)  # shape (num_min_pts, 3)
   ```

3. Replace the inner loops over max/min points with:
   ```python
   for i in range(num_contour_pts):
       contour_point = contour_points[i]

       max_dist, max_idx = max_tree.query(contour_point)
       max_id = int(max_cluster_ids[max_idx])

       min_dist, min_idx = min_tree.query(contour_point)
       min_id = int(min_cluster_ids[min_idx])

       if max_id != -1 and min_id != -1:
           assoc_set.add((max_id, min_id))
   ```

4. This eliminates the O(C × M) + O(C × N) nested loops entirely. Each `query` call is O(log N).

5. Optionally, vectorise fully:
   ```python
   _, max_indices = max_tree.query(contour_points)
   _, min_indices = min_tree.query(contour_points)

   for i in range(num_contour_pts):
       max_id = int(max_cluster_ids[max_indices[i]])
       min_id = int(min_cluster_ids[min_indices[i]])
       assoc_set.add((max_id, min_id))
   ```

**Definition of Done:**
- [ ] No 2-component Euclidean distance calculation remains in `compute_association_graph`.
- [ ] KD-tree uses all 3 Cartesian components.
- [ ] `test_date_line_association` passes.
- [ ] Integration test passes.

---

### Task 2.5 — Fix Feature ID Collision (Rounded Scalar as ID)

**Problem:** `polygon_id = round(path_max, 2)` means two RWPs with the same peak scalar value get the same raster label. One overwrites the other in the raster, and the tracking graph silently loses a feature.

**Files to edit:**
- `waper/tracking/rwp_polygon.py`
- `waper/interface/api.py`
- `waper/tracking/tracking_graph.py`

**Steps:**

1. In `api.py`, in the `_identify_rwps` function, change the loop that creates `rwp_info`:
   ```python
   for index, path in enumerate(time_step_data.identified_rwp_paths):
       (
           polygon,
           _unused_id,
           sample_points,
           weighted_lon,
           weighted_lat,
       ) = rwp_polygon.get_polygon_for_rwp_path(
           path, time_step_data.pruned_graph, time_step_data.vtk_data, config.scalar_name,
           config.min_latitude, config.max_latitude
       )
       # Use a unique monotonic index starting from 1
       rwp_id = index + 1
       time_step_data.rwp_info[tuple(path)] = {
           "polygon": polygon,
           "rwp_id": rwp_id,
           "sample_points": sample_points,
           "weighted_longitude": weighted_lon,
           "weighted_latitude": weighted_lat,
       }
   ```

2. Update `list_polygons` construction to use the new integer ID:
   ```python
   list_polygons.append((
       time_step_data.rwp_info[tuple(path)]["polygon"],
       time_step_data.rwp_info[tuple(path)]["rwp_id"],
   ))
   ```

3. In `rwp_polygon.py`, change `get_polygon_for_rwp_path` to return `None` for `polygon_id` (or simply remove it from the return tuple and adjust callers). The ID is now assigned externally.

4. In `tracking_graph.py`, the feature lookup `abs(feature - rwp_info["rwp_id"]) < 1e-2` now compares integers, so change to exact equality:
   ```python
   if feature == rwp_info["rwp_id"]:
   ```

5. Write a test: create two RWP paths with the same peak scalar value, assert they get different `rwp_id`s, and assert both appear in the raster with distinct labels.

**Definition of Done:**
- [ ] `rwp_id` is a unique integer per timestep.
- [ ] No `round(path_max, 2)` remains as an ID source.
- [ ] Two RWPs with identical peak values are distinguishable in the raster.
- [ ] Integration test passes.

---

### Task 2.6 — Fix Bare `except` in Stereographic Transform

**Problem:** `rwp_polygon.py` has `except:` which catches `KeyboardInterrupt`, `SystemExit`, etc., and re-raises a generic `ValueError` with no message, destroying the traceback.

**File to edit:**
- `waper/tracking/rwp_polygon.py`

**Steps:**

1. Replace:
   ```python
   try:
       return transformer.transform(input_xs, input_ys, errcheck=True)
   except:
       print(input_xs, input_ys)
       raise ValueError()
   ```
   with:
   ```python
   try:
       return transformer.transform(input_xs, input_ys, errcheck=True)
   except Exception as e:
       logger.error(
           "Stereographic transform failed for xs=%s, ys=%s: %s",
           input_xs, input_ys, e,
       )
       raise ValueError(
           f"Stereographic transform failed: {e}"
       ) from e
   ```

**Definition of Done:**
- [ ] No bare `except:` remains anywhere in the codebase.
- [ ] The re-raised error includes the original exception message.

---

## Phase 3: Algorithmic Improvements — Identification

These tasks change the scientific behavior of the identification step. Each must be validated against the synthetic test fixtures AND against real ERA data (manually, comparing plots before/after).

### Task 3.1 — Replace Affinity Propagation with DBSCAN/HDBSCAN

**Problem:** Affinity Propagation forces every extremum into a cluster. Isolated outlier extrema far from the primary group inflate the cluster footprint, making crests/troughs look artificially large. AP also has a global `preference` parameter (`median_dist / 5.0`) that doesn't adapt to per-region structure.

**File to edit:**
- `waper/identification/topology.py`

**Steps:**

1. Add `from sklearn.cluster import DBSCAN` (or `from hdbscan import HDBSCAN` if the dependency is acceptable; DBSCAN is already in sklearn).

2. Create a new unified function `cluster_extrema` that replaces both `cluster_max` and `cluster_min`:
   ```python
   def cluster_extrema(
       base_field,
       connectivity_clipped_scalar_field,
       extrema_points,
       scalar_name,
       sign,   # +1 for maxima, -1 for minima
       eps_km=500,    # DBSCAN neighborhood radius in km
       min_samples=1, # minimum cluster size
   ):
   ```

3. Inside the function, for each connected region:
   - If the region has 1 extremum: assign it to its own cluster (same as before).
   - If the region has ≥2 extrema: run DBSCAN with `eps=eps_km` (converted to the appropriate distance unit for the distance matrix) and `min_samples=min_samples` on the precomputed distance matrix.
   - DBSCAN returns label `-1` for noise points. **Discard noise points** — do not assign them to any cluster.

4. The DBSCAN `eps` parameter has a clear physical interpretation: "two extrema belong to the same crest/trough if they are within `eps` km of each other on the sphere." Expose this as `cluster_eps_km` in `WaperConfig` with a default of 500 km.

5. The `min_samples` parameter controls the minimum number of extrema to form a cluster. Default to 1 (any single point can be a cluster, which matches the original behavior for isolated extrema).

6. Remove `cluster_max` and `cluster_min`. Update all callers in `api.py` to call `cluster_extrema(..., sign=+1)` and `cluster_extrema(..., sign=-1)`.

7. Remove the two-point special case (`if len(region_array[k]) == 2: always merge`). DBSCAN handles this naturally: if two points are within `eps`, they cluster; if not, one or both become noise.

8. Add `cluster_eps_km` and `cluster_min_samples` to `WaperConfig`.

9. Run the clustering tests from Task 1.3. The `test_isolated_outlier_far_from_group` test should now PASS (remove the `xfail` mark).

**Definition of Done:**
- [ ] `cluster_max` and `cluster_min` no longer exist.
- [ ] `cluster_extrema` uses DBSCAN.
- [ ] Noise points (label -1) are excluded from all downstream processing.
- [ ] `WaperConfig` has `cluster_eps_km` and `cluster_min_samples`.
- [ ] `test_isolated_outlier_far_from_group` passes.
- [ ] Integration test passes.

---

### Task 3.2 — Fix the Similarity Penalty (`FindCellsAlongLine` Through Sphere Interior)

**Problem:** In `topology.py`, the similarity matrix penalty for clustering uses `vtkCellLocator.FindCellsAlongLine` to find cells between two extrema. This shoots a straight line through 3D Cartesian space, which passes through the *interior* of the sphere, not along the surface. For widely-separated or near-polar points, the ray may not intersect any surface cells, leaving `min_v` at its initialised value of 1000, which massively corrupts the distance.

**File to edit:**
- `waper/identification/topology.py`

**Steps — short-term fix (if keeping VTK for now):**

1. Replace `FindCellsAlongLine` with a surface-based approach: for each pair of extrema (i, j), the Dijkstra path already provides the sequence of vertices along the shortest surface path. Walk this vertex sequence and sample the scalar value at each vertex. Use the minimum (for maxima) or maximum (for minima) sampled value as the penalty.

2. Concretely, after the Dijkstra path is computed:
   ```python
   pts = dijkstra.GetOutput().GetPoints()
   scalar_arr = scalar_field.GetPointData().GetArray(scalar_name)
   id_list = dijkstra.GetIdList()

   min_v = float('inf')
   for ptIdx in range(id_list.GetNumberOfIds()):
       vid = id_list.GetId(ptIdx)
       val = scalar_arr.GetTuple1(vid)
       if val < min_v:
           min_v = val
   ```
   This samples along the *surface* path, not through the sphere interior.

3. Change `min_v` initialisation from `1000` to `float('inf')` and `max_v` from `-1000` to `float('-inf')`.

**Steps — long-term fix (Phase 5 replaces VTK entirely):**

This will be superseded by Task 5.2 which replaces VTK Dijkstra with scipy sparse graph. At that point, sampling along the path is trivial since we have the grid indices directly.

**Definition of Done:**
- [ ] `FindCellsAlongLine` is no longer used for the similarity penalty.
- [ ] Penalty is sampled along the surface geodesic path.
- [ ] No hardcoded `1000` / `-1000` initialisations remain.

---

### Task 3.3 — Use Weighted Centroid for Cluster Representative

**Problem:** The cluster representative point is the single grid cell with the most extreme scalar value, not the centroid. This puts the "center" of a multi-point cluster at an arbitrary edge of the group, distorting edge weights and graph structure.

**Files to edit:**
- `waper/identification/topology.py` (functions `max_cluster_assign`, `min_cluster_assign`)

**Steps:**

1. In `max_cluster_assign`, compute a value-weighted centroid for each cluster:
   ```python
   for i in range(num_points_max):
       cid = cluster_id_max[i]
       lon = max_points['Longitude'][i]
       lat = max_points['Latitude'][i]
       val = max_scalars[i]

       max_pt_dict[cid].append([lon, lat])

       # Track peak value (for scalar attribute)
       if cluster_max_arr[cid] < val:
           cluster_max_arr[cid] = val

       # Accumulate for weighted centroid
       cluster_lon_sum[cid] += lon * val
       cluster_lat_sum[cid] += lat * val
       cluster_weight_sum[cid] += val

   # Compute weighted centroid
   for cid in range(num_max_clusters):
       if cluster_weight_sum[cid] > 0:
           cluster_max_point[cid][0] = cluster_lon_sum[cid] / cluster_weight_sum[cid]
           cluster_max_point[cid][1] = cluster_lat_sum[cid] / cluster_weight_sum[cid]
   ```

2. **Longitude wraparound:** If a cluster straddles 0°/360°, naive averaging fails. Before accumulating, shift all longitudes in the cluster to be continuous (e.g., if the cluster contains both 350° and 10°, shift 10° to 370° before averaging, then wrap result back). Use `get_consistent_longitudes` from `rwp_polygon.py` or implement a simple version.

3. Do the same for `min_cluster_assign`, weighting by `abs(val)`.

4. The `cluster_max_arr` / `cluster_min_arr` (peak value) should remain unchanged — it's used for pruning. Only the `cluster_max_point` / `cluster_min_point` (representative position) changes.

**Definition of Done:**
- [ ] Cluster representative is a weighted centroid, not the peak-value point.
- [ ] Longitude wraparound is handled.
- [ ] `test_two_close_extrema_same_cluster` uses the centroid as the representative.
- [ ] Integration test passes.

---

### Task 3.4 — Fix Path Ranking to Solve Maximum-Weight Independent Set

**Problem:** The `top_paths` filter in `get_ranked_paths` keeps a path only if no overlapping path has higher weight. This is not transitive: path B can be eliminated by path A, which is then eliminated by path C, even though B and C don't overlap. Valid RWPs are dropped through indirect competition.

**File to edit:**
- `waper/identification/rwp_graph.py`

**Steps:**

1. Replace the current filter with a greedy maximum-weight independent set algorithm:
   ```python
   def get_ranked_paths(assoc_graph, max_weight):
       """Extract non-overlapping paths with maximum total weight."""

       # Step 1: Enumerate all candidate paths (same as before)
       path_list = _enumerate_all_candidate_paths(assoc_graph)

       # Step 2: Compute weight for each path
       path_weights = {}
       for path in path_list:
           w = sum(
               assoc_graph[path[i]][path[i+1]]["weight"]
               for i in range(len(path) - 1)
           )
           path_weights[tuple(path)] = w

       # Step 3: Sort paths by weight descending
       sorted_paths = sorted(path_list, key=lambda p: path_weights[tuple(p)], reverse=True)

       # Step 4: Greedy selection — pick highest-weight path, remove all
       #         paths that share a node with it, repeat.
       selected = []
       used_nodes = set()
       for path in sorted_paths:
           path_nodes = set(path)
           if path_nodes.isdisjoint(used_nodes):
               selected.append(path)
               used_nodes.update(path_nodes)

       return selected
   ```

2. Extract the path enumeration into a helper `_enumerate_all_candidate_paths` to keep the main function clean. This is the existing loop over source/sink pairs with `nx.all_simple_paths`.

3. Apply the same fix to `get_track_paths` in `tracking_graph.py`.

**Definition of Done:**
- [ ] Path selection is greedy by weight, not pairwise filter.
- [ ] A path that doesn't overlap with the winner is never dropped.
- [ ] Integration test passes.

---

### Task 3.5 — Fix Longitude Delta Pruning for Wraparound

**Problem:** `prune_association_graph_edges` uses `abs(lon_0 - lon_1) <= WAPER_MIN_LON_DELTA` which doesn't handle the 0°/360° boundary. Clusters at 358° and 2° compute `abs(358 - 2) = 356` instead of the true 4°.

**File to edit:**
- `waper/identification/rwp_graph.py`

**Steps:**

1. Create a helper function:
   ```python
   def _longitude_separation(lon1, lon2):
       """Compute the shortest angular separation in degrees, handling wraparound."""
       delta = abs(lon1 - lon2) % 360
       return min(delta, 360 - delta)
   ```

2. Replace `abs(lon_0 - lon_1)` with `_longitude_separation(lon_0, lon_1)`.

3. Move `WAPER_MIN_LON_DELTA` into `WaperConfig` as `min_longitude_separation` with default 6.

4. Write tests in `tests/test_utils.py`:
   ```python
   def test_lon_separation_normal():
       assert _longitude_separation(10, 20) == 10

   def test_lon_separation_wraparound():
       assert _longitude_separation(358, 2) == 4

   def test_lon_separation_symmetric():
       assert _longitude_separation(2, 358) == 4
   ```

**Definition of Done:**
- [ ] `_longitude_separation` handles wraparound.
- [ ] All 3 tests pass.
- [ ] Integration test passes.

---

### Task 3.6 — Fix Radius Inconsistency

**Problem:** `utils.py` defines `RADIUS_SPHERE = 63.71` (used for VTK mesh construction) and `RADIUS_EARTH = 6.371e6` (used in haversine). The VTK Dijkstra distances are computed on a sphere of radius 63.71, but haversine returns meters. These distance scales differ by a factor of 100,000, which means the clustering similarity matrix (Dijkstra distances) and the edge weight computation (haversine distances) are on completely different scales.

**File to edit:**
- `waper/identification/utils.py`
- `waper/identification/topology.py` (if the `CLUSTER_MAX_DISTANCE` depends on the scale)

**Steps:**

1. Decide on a single distance convention. The simplest is: keep `RADIUS_SPHERE = 63.71` for VTK mesh visualisation, but normalize all distance computations to kilometres.

2. In `haversine_distance`, return distance in **kilometres**:
   ```python
   RADIUS_EARTH_KM = 6371.0

   def haversine_distance(lat1, lon1, lat2, lon2):
       """Return great-circle distance in kilometres."""
       dlat = math.radians(lat1 - lat2)
       dlon = math.radians(lon1 - lon2)
       a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
       c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
       return RADIUS_EARTH_KM * c
   ```

3. In `topology.py`, scale the Dijkstra distances from the VTK mesh to kilometres:
   ```python
   SCALE_FACTOR = RADIUS_EARTH_KM / RADIUS_SPHERE  # ≈ 100.0
   ```
   Multiply all `dist_matrix` entries by `SCALE_FACTOR` after computing them.

4. Update `CLUSTER_MAX_DISTANCE` to be in kilometres (currently 150 — this would be ~150 × 100 = 15,000 km in real units, which is unreasonable. It's probably meant to be 150 on the scaled sphere, i.e. ~15,000 km. Verify against the science and set appropriately — likely `CLUSTER_MAX_DISTANCE = 15000` in km).

5. Update `WAPER_MAX_NODE_DISTANCE` similarly.

6. In `tracking_graph.py`, remove the `distance / 1000` division (line ~80) since haversine now returns km directly.

**Definition of Done:**
- [ ] All distance values throughout the code are in kilometres.
- [ ] `haversine_distance` returns km.
- [ ] `CLUSTER_MAX_DISTANCE` is documented with its unit.
- [ ] Integration test passes.

---

## Phase 4: Algorithmic Improvements — Tracking

### Task 4.1 — Use Concave Hull (Alpha Shape) Instead of Convex Hull

**Problem:** `MultiPoint(...).convex_hull` creates a single convex polygon for the entire RWP path. Since an RWP alternates between positive and negative lobes, the convex hull fills in the gaps, creating an unrealistically large footprint. This inflates overlap during tracking.

**File to edit:**
- `waper/tracking/rwp_polygon.py`

**Steps:**

1. Install/import `shapely`:
   ```python
   from shapely.ops import unary_union
   from shapely.geometry import MultiPoint
   ```

2. If using Shapely ≥ 2.0, use `shapely.concave_hull`:
   ```python
   from shapely import concave_hull

   points = MultiPoint(list(zip(xs, ys)))
   rwp_poly = concave_hull(points, ratio=0.3)  # ratio 0-1, lower = more concave
   ```
   If Shapely < 2.0, use `alphashape` package or compute per-node convex hulls and take their union:
   ```python
   per_node_hulls = []
   for node_points in node_point_groups:
       if len(node_points) >= 3:
           per_node_hulls.append(MultiPoint(node_points).convex_hull)
   rwp_poly = unary_union(per_node_hulls)
   ```

3. The second approach (per-node union) is scientifically better because each crest/trough gets its own hull. Refactor `get_polygon_for_rwp_path` to collect points **per node** and create per-node polygons, then union them.

4. Expose the hull method as a config option: `WaperConfig.hull_method = "concave" | "convex" | "per_node"`.

**Definition of Done:**
- [ ] Default footprint is not a single convex hull.
- [ ] Footprints visually match the RWP structure (narrow along the wave, not filled in).
- [ ] Integration test passes.
- [ ] Tracking results are not degraded.

---

### Task 4.2 — Support Southern Hemisphere

**Problem:** `transform_to_stereographic` hardcodes North Pole stereographic projection. The raster bounds are hardcoded for NH. The `TODO` comment acknowledges this.

**File to edit:**
- `waper/tracking/rwp_polygon.py`

**Steps:**

1. Add a `hemisphere` parameter to `transform_to_stereographic`:
   ```python
   def transform_to_stereographic(input_xs, input_ys, hemisphere="north", inverse=False):
       from_crs = pyproj.crs.CRS(4326)
       if hemisphere == "north":
           to_crs = pyproj.crs.CRS("+proj=stere +lat_0=90 +lon_0=0")
       elif hemisphere == "south":
           to_crs = pyproj.crs.CRS("+proj=stere +lat_0=-90 +lon_0=0")
       else:
           raise ValueError(f"hemisphere must be 'north' or 'south', got '{hemisphere}'")
       ...
   ```

2. Compute `WAPER_X_BOUNDS` and `WAPER_Y_BOUNDS` dynamically from the hemisphere, or make them functions of `hemisphere`.

3. Add `hemisphere` to `WaperConfig` with default `"north"`.

4. Thread the hemisphere through all calls to `transform_to_stereographic`.

5. Write a test: create a wave field in the Southern Hemisphere (latitudes -20 to -80), run the pipeline, assert no crash and polygons are in the correct hemisphere.

**Definition of Done:**
- [ ] `WaperConfig.hemisphere` exists, defaults to `"north"`.
- [ ] SH test passes.
- [ ] NH behavior is unchanged.

---

### Task 4.3 — Replace `all_simple_paths` in Tracking with DAG Longest Path

**Problem:** `get_track_paths` uses `nx.all_simple_paths` which has factorial worst-case complexity. The tracking graph is a DAG (edges go strictly forward in time), so a linear-time DP longest-path algorithm exists.

**File to edit:**
- `waper/tracking/tracking_graph.py`

**Steps:**

1. Replace `get_track_paths` with:
   ```python
   def get_track_paths(tracking_graph):
       """Extract tracks as longest-weight paths in the tracking DAG."""

       # Topological sort (linear time)
       topo_order = list(nx.topological_sort(tracking_graph))

       # DP: for each node, store (best_weight_to_here, predecessor)
       best_weight = {node: 0 for node in topo_order}
       predecessor = {node: None for node in topo_order}

       for node in topo_order:
           for succ in tracking_graph.successors(node):
               edge_wt = tracking_graph[node][succ]["weight"]
               candidate = best_weight[node] + edge_wt
               if candidate > best_weight[succ]:
                   best_weight[succ] = candidate
                   predecessor[succ] = node

       # Extract paths by backtracking from end nodes
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

       # Deduplicate: if two paths share segments, keep the higher-weight one
       # (greedy independent set, same as Task 3.4)
       return _greedy_select_independent_paths(track_paths, tracking_graph)
   ```

2. Complexity: O(V + E), versus O(V! / (V-k)!) worst case for `all_simple_paths`.

3. Write a benchmark test with a synthetic tracking graph of 20 timesteps × 5 features each, and assert it completes in < 1 second.

**Definition of Done:**
- [ ] `nx.all_simple_paths` is no longer called in `tracking_graph.py`.
- [ ] Tracking of 20 timesteps completes in < 1 second.
- [ ] Integration test passes.

---

### Task 4.4 — Decouple Quadtree Merge from Per-Feature Loop

**Problem:** In `build_tracking_graph`, the quadtree `merge` is called once per timestep (correct), but the `edge_list` is the Cartesian product of all features including 0 (background). Also, the merge is recomputed for every feature node at the current timestep, even though it only depends on the pair of timesteps.

**File to edit:**
- `waper/tracking/tracking_graph.py`

**Steps:**

1. Move the merge computation **outside** the feature loop:
   ```python
   if time > 0:
       merge_graph = merge(
           time_step_data[time].quadtree,
           time_step_data[time - 1].quadtree,
       )
       merge_feature_size = compute_size_features(merge_graph)
       prev_feature_size = compute_size_features(time_step_data[time - 1].quadtree)
       curr_feature_size = compute_size_features(time_step_data[time].quadtree)
   ```

2. Filter out feature 0 from the Cartesian product:
   ```python
   prev_features = time_step_data[time - 1].raster_features - {0}
   curr_features = time_step_data[time].raster_features - {0}
   edge_list = list(product(prev_features, curr_features))
   ```

3. This is both a correctness fix (avoids creating spurious edges involving the background) and a performance fix (merge is computed once, not once per feature).

**Definition of Done:**
- [ ] `merge` is called exactly once per pair of consecutive timesteps.
- [ ] Feature 0 is never in the Cartesian product.
- [ ] Integration test passes.

---

## Phase 5: VTK-to-PyVista / SciPy Refactor

The current code mixes raw VTK API calls (e.g., `vtk.vtkDijkstraGraphGeodesicPath`, `vtk.vtkGeometryFilter`, `vtk.vtkTriangleFilter`, `vtk.vtkCellLocator`, `vtk.vtkContourFilter`, `vtk.vtkGradientFilter`, `vtk.vtkConnectivityFilter`, `vtk.vtkIntArray`, `vtk.vtkFloatArray`, `vtk.vtkIdList`) with PyVista's high-level API. The goal is to eliminate all raw VTK calls and use either PyVista wrappers or SciPy equivalents.

### Task 5.1 — Replace VTK Contour and Gradient Filters

**File to edit:**
- `waper/identification/utils.py`

**Steps:**

1. Replace `get_iso_contour` with PyVista's `.contour()`:
   ```python
   def get_iso_contour(scalar_field, value, scalar_name):
       """Extract isocontour at given value."""
       return scalar_field.contour([value], scalars=scalar_name)
   ```
   Note: `api.py` already uses `time_step_data.vtk_data.contour(...)` in one place. The `get_iso_contour` function in `utils.py` may be unused — check and remove if so.

2. Replace `compute_gradients` with PyVista's `.compute_derivative()`:
   ```python
   def compute_gradients(scalar_field, scalar_name):
       return scalar_field.compute_derivative(scalars=scalar_name)
   ```
   Check if this function is called anywhere. If not, delete it.

3. Remove `import vtk` from `utils.py`.

**Definition of Done:**
- [ ] No `vtk.vtkContourFilter` or `vtk.vtkGradientFilter` in the codebase.
- [ ] `import vtk` is removed from `utils.py`.

---

### Task 5.2 — Replace VTK Dijkstra with SciPy Sparse Graph

**File to edit:**
- `waper/identification/topology.py`

This is the largest single refactoring task. Take it in sub-steps.

**Sub-step A: Build a sparse adjacency matrix from the PyVista mesh.**

1. The clipped scalar field is a PyVista `UnstructuredGrid`. Extract its cell connectivity to build a sparse graph:
   ```python
   from scipy.sparse import lil_matrix
   from scipy.sparse.csgraph import shortest_path

   def build_adjacency_matrix(mesh):
       """Build a sparse adjacency matrix from mesh connectivity.

       Edge weights are the Euclidean distances between connected points.
       """
       n = mesh.n_points
       adj = lil_matrix((n, n), dtype=float)

       # Extract edges from cells
       for i in range(mesh.n_cells):
           cell = mesh.get_cell(i)
           point_ids = [cell.point_ids[j] for j in range(cell.n_points)]
           for a_idx in range(len(point_ids)):
               for b_idx in range(a_idx + 1, len(point_ids)):
                   pa = point_ids[a_idx]
                   pb = point_ids[b_idx]
                   dist = np.linalg.norm(mesh.points[pa] - mesh.points[pb])
                   adj[pa, pb] = dist
                   adj[pb, pa] = dist

       return adj.tocsr()
   ```

2. Alternatively, use `mesh.extract_all_edges()` which returns a PolyData of line segments, then parse those.

**Sub-step B: Replace Dijkstra computation.**

1. In `cluster_extrema` (the unified function from Task 3.1), after building the adjacency matrix:
   ```python
   adj = build_adjacency_matrix(mesh)

   # Get point IDs of the extrema
   extrema_ids = extrema_points.point_data["vtkOriginalPointIds"].astype(int)

   # Compute shortest paths between all extrema pairs within each region
   for region_id in range(num_regions):
       region_extrema = [idx for idx in region_indices if regions[idx] == region_id]
       if len(region_extrema) <= 1:
           continue

       # Extract subgraph for this region
       region_point_ids = np.where(all_region_ids == region_id)[0]
       sub_adj = adj[np.ix_(region_point_ids, region_point_ids)]

       # Map extrema IDs to subgraph indices
       id_map = {pid: i for i, pid in enumerate(region_point_ids)}
       local_extrema = [id_map[eid] for eid in extrema_ids_in_region]

       # Compute pairwise shortest paths (only for the extrema rows)
       dist_matrix_sub = shortest_path(sub_adj, indices=local_extrema)
       # dist_matrix_sub[i, j] = shortest path from local_extrema[i] to local_extrema[j]
   ```

**Sub-step C: Replace the similarity penalty.**

1. Along the shortest path (which is now a sequence of grid indices), sample the scalar field values and find the min (for maxima) or max (for minima):
   ```python
   # Use predecessors to reconstruct the path
   dist, predecessors = shortest_path(sub_adj, indices=local_extrema, return_predecessors=True)

   def get_path_penalty(predecessors, source_local, target_local, scalar_values, sign):
       """Walk the shortest path and find the extremal scalar value."""
       path = []
       current = target_local
       while current != source_local:
           path.append(current)
           current = predecessors[source_local, current]
       path.append(source_local)

       path_values = scalar_values[path]
       if sign == +1:  # maxima: penalty is min value along path
           return np.min(path_values)
       else:           # minima: penalty is max value along path
           return np.max(path_values)
   ```

**Sub-step D: Remove all VTK imports from topology.py.**

1. Remove `import vtk`.
2. Remove `vtk.vtkGeometryFilter`, `vtk.vtkTriangleFilter`, `vtk.vtkDijkstraGraphGeodesicPath`, `vtk.vtkCellLocator`, `vtk.vtkIdList`, `vtk.vtkIntArray`.
3. Use `mesh.point_data["Cluster ID"] = cluster_assign` instead of VTK arrays.

**Definition of Done:**
- [ ] `import vtk` is removed from `topology.py`.
- [ ] No VTK Dijkstra, geometry filter, triangle filter, or cell locator remains.
- [ ] Clustering results match (or improve upon) the VTK-based version.
- [ ] Integration test passes.

---

### Task 5.3 — Replace Remaining VTK Calls in `max_min.py`

**File to edit:**
- `waper/identification/max_min.py`

**Steps:**

1. Remove `import vtk`.
2. The `interpolate_cell_values` function uses raw VTK API (`GetNumberOfCells`, `GetCell`, `GetPointId`, `vtkFloatArray`). After Task 2.1, check if this function is still called. If it is, replace with:
   ```python
   def interpolate_cell_values(dataset, scalar_name):
       """Interpolate point data to cell data using PyVista."""
       cell_data = dataset.point_data_to_cell_data()
       dataset.cell_data[f"{scalar_name} Cell Value"] = cell_data[scalar_name]
       return dataset
   ```
3. The `clip_dataset` function already uses PyVista's `clip_scalar` — no change needed.

**Definition of Done:**
- [ ] `import vtk` is removed from `max_min.py`.
- [ ] `interpolate_cell_values` uses PyVista API.

---

### Task 5.4 — Replace VTK Connectivity Filter

**File to edit:**
- `waper/identification/topology.py`

**Steps:**

1. `identify_connected_regions` already uses PyVista:
   ```python
   return dataset.connectivity(largest=False)
   ```
   This is fine. No change needed.

2. `add_connectivity_data_min` uses raw VTK:
   ```python
   connectivity_filter = vtk.vtkConnectivityFilter()
   ```
   Check if this function is called anywhere. If not, delete it. If it is, replace with:
   ```python
   def add_connectivity_data_min(dataset):
       return pv.wrap(dataset).connectivity(largest=False)
   ```

**Definition of Done:**
- [ ] No `vtk.vtkConnectivityFilter` in the codebase.
- [ ] `add_connectivity_data_min` is either deleted or uses PyVista.

---

### Task 5.5 — Replace NetworkX Quadtree with Spatial Index

**Problem:** The quadtree is implemented as a `networkx.DiGraph` with substantial per-node dictionary overhead. For spatial intersection, an R-tree or STRtree operating directly on the polygons would be faster and simpler.

**File to edit:**
- `waper/tracking/quadtree.py`
- `waper/tracking/tracking_graph.py`

**This is a stretch goal** — the current quadtree works correctly (if slowly). Defer to Phase 9 unless performance is a blocking issue.

**Steps (sketch):**

1. Replace the quadtree merge + pixel counting with direct Shapely polygon intersection:
   ```python
   from shapely.ops import unary_union

   def compute_overlap(poly_a, poly_b):
       intersection = poly_a.intersection(poly_b)
       return intersection.area / max(poly_a.area, poly_b.area)
   ```

2. In `build_tracking_graph`, iterate over pairs of polygons from consecutive timesteps and compute overlap directly.

3. Remove `quadtree.py` entirely.
4. Remove `raster_data`, `raster_features`, `quadtree` from `WaperSingleTimestepData`.
5. Remove `rasterize_all_rwps` from `rwp_polygon.py`.
6. Remove `WAPER_IMAGE_SIZE`, `WAPER_NUM_PIXELS`, `WAPER_X_BOUNDS`, `WAPER_Y_BOUNDS`, `WAPER_RASTER_TRANSFORM`.

**Definition of Done:**
- [ ] `quadtree.py` is deleted.
- [ ] Tracking uses direct polygon intersection.
- [ ] Raster-related code is removed.
- [ ] Integration test passes.

---

## Phase 6: Performance Optimisation

### Task 6.1 — Vectorise `interpolate_cell_values`

Already covered by Task 5.3 (using `point_data_to_cell_data()`).

### Task 6.2 — Vectorise Association Graph Construction

Already covered by Task 2.4 (using KD-trees).

### Task 6.3 — Profile and Benchmark

**File to create:** `tests/test_benchmark.py`

**Steps:**

1. Write a benchmark that runs the full pipeline on a synthetic 721×1440 grid (ERA5 resolution) for 1 timestep and measures wall-clock time.

2. Write a benchmark for tracking: 10 timesteps of 721×1440.

3. Use `pytest-benchmark` or simple `time.time()` assertions:
   ```python
   def test_identification_benchmark(large_wave_field):
       import time
       start = time.time()
       result = _identify_rwps(large_wave_field, default_config)
       elapsed = time.time() - start
       assert elapsed < 30  # Should complete in <30s for one timestep
   ```

4. Record baseline timings. After each optimisation task, re-run and verify improvement.

**Definition of Done:**
- [ ] Benchmark tests exist for identification and tracking.
- [ ] Baseline timings are recorded in a comment or markdown file.

---

## Phase 7: Visualisation Overhaul

### Task 7.1 — Make All Plots Accept User-Provided Axes and Projection

**Problem:** Most plot functions create their own `plt.subplot(projection=...)`, preventing integration into user figure layouts. Projections are hardcoded (PlateCarree, Orthographic, Stereographic — inconsistently).

**File to edit:**
- `waper/interface/visualization.py`
- `waper/interface/api.py`

**Steps:**

1. For every `_plot_*` function, ensure the signature accepts `ax=None` and `projection=None`:
   ```python
   def _plot_graph(rwp_graph, scalar_data=None, ax=None, projection=None):
       if projection is None:
           projection = ccrs.PlateCarree()
       if ax is None:
           fig, ax = plt.subplots(subplot_kw={"projection": projection})
       ...
       return ax
   ```

2. For `_plot_clusters`, which currently creates two subplots internally (211, 212), refactor to accept `fig=None, axes=None`:
   ```python
   def _plot_clusters(..., fig=None, axes=None, projection=None):
       if projection is None:
           projection = ccrs.PlateCarree(central_longitude=180)
       if axes is None:
           fig, axes = plt.subplots(2, 1, subplot_kw={"projection": projection})
       ax_top, ax_bottom = axes
       ...
       return fig, axes
   ```

3. In the `Waper` class methods (`plot_clusters`, `plot_association_graph`, etc.), pass through `ax=None` and `projection=None` to the underlying function.

**Definition of Done:**
- [ ] Every `_plot_*` function accepts `ax` and `projection`.
- [ ] No function calls `plt.subplot(...)` with a hardcoded projection unless `ax` is None.
- [ ] User can pass their own axes into any plot method.

---

### Task 7.2 — Add Coastlines and Gridlines

**File to edit:**
- `waper/interface/visualization.py`

**Steps:**

1. In every `_plot_*` function, after creating or receiving the axes, add:
   ```python
   ax.coastlines(linewidth=0.5, color='gray')
   ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
   ```

2. Make this optional via a `geographic_features=True` parameter.

**Definition of Done:**
- [ ] All map plots show coastlines and gridlines by default.
- [ ] `geographic_features=False` suppresses them.

---

### Task 7.3 — Add Hovmöller Diagram for Tracking

**File to create:**
- New function in `waper/interface/visualization.py`
- New method in `Waper` class

**Steps:**

1. Write `_plot_hovmoller(tracking_graph, track_paths)`:
   - X-axis: longitude.
   - Y-axis: time (timestep index).
   - Plot each track as a line showing the weighted longitude at each timestep.
   - Color-code by track index or by intensity.

2. Add `Waper.plot_hovmoller()` method that calls the above.

**Definition of Done:**
- [ ] `plot_hovmoller` produces a longitude-vs-time diagram.
- [ ] Each track is a distinct colored line.

---

### Task 7.4 — Fix Polygon Plotting Projection Mismatch

**Problem:** `_plot_polygons` plots polygon boundaries in stereographic coordinates but scatter points (weighted centers) in PlateCarree. This works by accident because Cartopy transforms, but it's confusing and fragile.

**File to edit:**
- `waper/interface/visualization.py`

**Steps:**

1. Convert polygon exterior coordinates back to lat/lon before plotting:
   ```python
   from ..tracking.rwp_polygon import transform_to_stereographic

   for poly in poly_list:
       stereo_lons, stereo_lats = poly.exterior.coords.xy
       geo_lons, geo_lats = transform_to_stereographic(
           np.array(stereo_lons), np.array(stereo_lats), inverse=True
       )
       ax.plot(geo_lons, geo_lats, transform=ccrs.PlateCarree())
   ```

2. This allows the polygon to be plotted on any projection, not just stereographic.

**Definition of Done:**
- [ ] All plot elements use a consistent geographic CRS.
- [ ] Polygons render correctly on PlateCarree, Orthographic, and Stereographic projections.

---

## Phase 8: Documentation

### Task 8.1 — Fix Template Documentation

**Problem:** `docs/contents/30_usage.rst` references `my_new_project`. `docs/contents/40_modules.rst` references `my_new_project`. `README.rst` has TODO placeholders.

**Files to edit:**
- `docs/contents/30_usage.rst`
- `docs/contents/40_modules.rst`
- `docs/contents/my_new_project.rst`
- `README.rst`

**Steps:**

1. Replace all `my_new_project` references with `waper`.
2. In `30_usage.rst`, write a basic usage example:
   ```rst
   =====
   Usage
   =====

   .. code-block:: python

      import xarray as xr
      from waper import Waper

      ds = xr.open_dataset("era5_v_wind.nc")

      w = Waper(
          data_array=ds,
          scalar_name="v",
          latitude_label="latitude",
          longitude_label="longitude",
          time_label="time",
      )

      w.identify_rwps()
      w.track_rwps()

      ax = w.plot_rwp_graphs(time_index=0)
   ```
3. In `40_modules.rst`, reference `waper` not `my_new_project`.
4. Rename or delete `my_new_project.rst`.
5. In `README.rst`, fill in the TODO placeholders with actual feature descriptions.

**Definition of Done:**
- [ ] `my_new_project` does not appear anywhere in the docs.
- [ ] Usage example is correct and runnable.
- [ ] README lists actual features.

---

### Task 8.2 — Add Docstrings to All Public Functions

**Problem:** Many functions have incomplete or missing docstrings. Type hints are sparse despite `py.typed` marker.

**Files to edit:** All `.py` files in `waper/`.

**Steps:**

1. For every public function (not starting with `_`), ensure a Google-style docstring exists with:
   - One-line summary.
   - `Args:` section with type and description for each parameter.
   - `Returns:` section with type and description.
   - `Raises:` section if applicable.

2. Add type hints to all function signatures. At minimum:
   - `api.py`: `Waper.__init__`, `identify_rwps`, `track_rwps`, all `plot_*` methods.
   - `rwp_graph.py`: `compute_association_graph`, `prune_association_graph_nodes`, `prune_association_graph_edges`, `get_ranked_paths`.
   - `topology.py`: `cluster_extrema` (after Task 3.1).
   - `utils.py`: `haversine_distance`, `is_to_the_east`.

3. Run `mypy waper/ --ignore-missing-imports` and fix any type errors that arise from the new annotations.

**Definition of Done:**
- [ ] Every public function has a docstring.
- [ ] Every public function has type hints on arguments and return.
- [ ] `mypy` reports zero errors (or only errors from third-party stubs).

---

### Task 8.3 — Write an Algorithm Overview in the README

**Steps:**

1. Add a section to `README.rst` (or a separate `docs/algorithm.rst`) that summarises the algorithm in 2-3 paragraphs, referencing Pandey et al. (2020) and Malavika's thesis.

2. Include a diagram (ASCII or image) showing the pipeline:
   ```
   Input (v-wind field)
     → Extrema Detection
     → Connected Region Labeling
     → Clustering (DBSCAN)
     → Association Graph
     → Node Pruning → Edge Pruning
     → Path Extraction
     → Polygon Footprints
     → Rasterisation / Quadtree
     → Temporal Tracking Graph
     → Track Extraction
   ```

3. Link to the full `architecture_and_algorithm.md` for detailed math.

**Definition of Done:**
- [ ] README or docs contains an algorithm overview.
- [ ] Both papers are cited.

---

## Phase 9: Stretch Goals

These are desirable but not critical. They can be tackled after all other phases.

### Task 9.1 — Replace NetworkX Quadtree with Direct Polygon Intersection

See Task 5.5 for full specification.

### Task 9.2 — Parallel Timestep Processing

**Problem:** `identify_rwps` processes timesteps sequentially in a `for` loop. Each timestep is independent during identification.

**Steps:**

1. Use `concurrent.futures.ProcessPoolExecutor` or `joblib.Parallel`:
   ```python
   from joblib import Parallel, delayed

   def identify_rwps(self):
       self._time_step_data = Parallel(n_jobs=-1)(
           delayed(_identify_rwps)(
               self.data_array[self._config.scalar_name][i], self._config
           )
           for i in range(self._num_time_steps)
       )
   ```

2. Ensure `WaperConfig` and `DataArray` are picklable (they should be).

3. VTK objects are NOT picklable. This task depends on Phase 5 (removing VTK).

**Definition of Done:**
- [ ] Identification runs in parallel across timesteps.
- [ ] Results are identical to sequential execution.

---

### Task 9.3 — xarray Integration for Output

**Problem:** Results are stored in ad-hoc dicts and dataclasses. Users must manually extract coordinates.

**Steps:**

1. Add a method `Waper.to_dataset()` that returns an `xarray.Dataset` with:
   - A variable `rwp_id(time, rwp_index)` — the ID of each RWP at each timestep.
   - A variable `rwp_longitude(time, rwp_index)` — weighted longitude.
   - A variable `rwp_latitude(time, rwp_index)` — weighted latitude.
   - A variable `rwp_peak_value(time, rwp_index)` — peak scalar value.

2. Add a method `Waper.tracks_to_dataframe()` that returns a `pandas.DataFrame` with columns: `track_id`, `time`, `longitude`, `latitude`, `peak_value`.

**Definition of Done:**
- [ ] `to_dataset()` returns a well-formed xarray Dataset.
- [ ] `tracks_to_dataframe()` returns a well-formed DataFrame.

---

### Task 9.4 — Add `WaperConfig.from_yaml()` and `WaperConfig.to_yaml()`

**Steps:**

1. Use `dataclasses.asdict` + `yaml.dump` for serialisation.
2. Use `yaml.safe_load` + `WaperConfig(**d)` for deserialisation.
3. This allows reproducible runs with config files.

**Definition of Done:**
- [ ] Round-trip: `WaperConfig.from_yaml(config.to_yaml()) == config`.

---

## Appendix A: File Inventory

| File | Purpose | Key Issues |
|------|---------|------------|
| `waper/__init__.py` | Package root | Exposes internal submodules (Task 0.3) |
| `waper/interface/__init__.py` | Interface subpackage | Same issue |
| `waper/interface/api.py` | Main `Waper` class, `WaperConfig`, orchestration | Dead `logging` function; absolute import; Feature ID bug |
| `waper/interface/visualization.py` | All plotting functions | Hardcoded projections; no coastlines; projection mismatch |
| `waper/identification/__init__.py` | Empty | — |
| `waper/identification/max_min.py` | Extrema detection | O(N²) loops; `check` array bug; boundary bug; dead code |
| `waper/identification/topology.py` | Clustering | AP forcing; VTK Dijkstra; `FindCellsAlongLine` bug; code duplication |
| `waper/identification/rwp_graph.py` | Association graph | Node ID collision; Euclidean distance; lon wraparound; path ranking |
| `waper/identification/utils.py` | Utilities (mesh, distance) | `is_to_the_east` bug; radius inconsistency; raw VTK calls |
| `waper/tracking/__init__.py` | Empty | — |
| `waper/tracking/quadtree.py` | Quadtree spatial index | NetworkX overhead (stretch goal) |
| `waper/tracking/rwp_polygon.py` | Polygon footprints | Feature ID bug; convex hull; NH-only; bare except |
| `waper/tracking/tracking_graph.py` | Temporal tracking graph | `all_simple_paths`; redundant merge; path ranking |
| `tests/smoke_test.py` | Smoke test | Tests `my_new_project` |
| `pyproject.toml` | Build config | No dependencies; wrong metadata |
| `.github/workflows/test.yaml` | CI | References `cookiecutter_python`; wrong Python versions |
| `docs/` | Documentation | All template/placeholder |

---

## Appendix B: Magic Numbers Registry

Every hardcoded constant that should be either configurable or documented with its unit.

| Constant | File | Current Value | Unit | Proposed Location |
|----------|------|---------------|------|-------------------|
| `CLUSTER_MAX_DISTANCE` | `topology.py` | 150 | VTK-sphere units (~15000 km) | `WaperConfig.cluster_max_distance_km` |
| `WAPER_MAX_SCALAR_VALUE` | `rwp_graph.py` | 100 | m/s | `WaperConfig.max_scalar_value` |
| `WAPER_MAX_NODE_DISTANCE` | `rwp_graph.py` | 1000 | VTK-sphere units | `WaperConfig.max_node_distance_km` |
| `WAPER_MIN_LON_DELTA` | `rwp_graph.py` | 6 | degrees | `WaperConfig.min_longitude_separation` |
| `WAPER_SUBSAMPLE` | `rwp_polygon.py` | 5 | points | `WaperConfig.polygon_subsample` |
| `WAPER_IMAGE_SIZE` | `rwp_polygon.py` | 512 | pixels | `WaperConfig.raster_size` |
| `WAPER_CLUSTER_WIDTH` | `rwp_polygon.py` | 60 | degrees | `WaperConfig.cluster_width_degrees` |
| `WAPER_X_BOUNDS` | `rwp_polygon.py` | hardcoded | meters (stereo) | Compute dynamically from hemisphere |
| `WAPER_Y_BOUNDS` | `rwp_polygon.py` | hardcoded | meters (stereo) | Compute dynamically from hemisphere |
| `RADIUS_SPHERE` | `utils.py` | 63.71 | arbitrary (100 km?) | Document clearly; keep as internal |
| `RADIUS_EARTH` | `utils.py` | 6.371e6 | meters | Change to `RADIUS_EARTH_KM = 6371.0` |
| `median_dist / 5.0` | `topology.py` | varies | AP preference | Remove when switching to DBSCAN |
| `path_max / 3.0` | `rwp_polygon.py` | varies | clip fraction | `WaperConfig.polygon_clip_fraction` |
| `0.001` (ray tolerance) | `topology.py` | 0.001 | VTK units | Remove when switching to SciPy |

---

## Appendix C: Dependency Map

Shows which tasks depend on which. **Independent tasks within a phase can be parallelised.**

```
Phase 0 (all independent of each other)
  ├── 0.1 (pyproject.toml)
  ├── 0.2 (dead code removal)
  ├── 0.3 (__init__.py + smoke test)
  └── 0.4 (logging)

Phase 1 (depends on Phase 0)
  ├── 1.1 (fixtures) ← all other Phase 1 tasks depend on this
  ├── 1.2 (test extrema) ← depends on 1.1
  ├── 1.3 (test clustering) ← depends on 1.1
  ├── 1.4 (test association graph) ← depends on 1.1
  ├── 1.5 (test tracking) ← depends on 1.1
  └── 1.6 (integration test) ← depends on 1.1

Phase 2 (depends on Phase 1 for validation)
  ├── 2.1 (extrema detection) ← independent
  ├── 2.2 (node ID collision) ← independent
  ├── 2.3 (is_to_the_east) ← independent
  ├── 2.4 (Euclidean distance) ← independent
  ├── 2.5 (feature ID) ← independent
  └── 2.6 (bare except) ← independent

Phase 3 (depends on Phase 2 for correctness baseline)
  ├── 3.1 (DBSCAN) ← independent
  ├── 3.2 (similarity penalty) ← depends on 3.1 (uses cluster_extrema)
  ├── 3.3 (weighted centroid) ← depends on 3.1 (uses cluster_extrema)
  ├── 3.4 (path ranking) ← independent
  ├── 3.5 (lon wraparound) ← independent
  └── 3.6 (radius units) ← independent

Phase 4 (depends on Phase 3)
  ├── 4.1 (concave hull) ← independent
  ├── 4.2 (southern hemisphere) ← independent
  ├── 4.3 (DAG longest path) ← independent
  └── 4.4 (decouple merge) ← independent

Phase 5 (depends on Phase 3 for clean topology.py)
  ├── 5.1 (VTK contour/gradient) ← independent
  ├── 5.2 (VTK Dijkstra → SciPy) ← depends on 3.1, 3.2
  ├── 5.3 (VTK in max_min) ← depends on 2.1
  ├── 5.4 (VTK connectivity) ← independent
  └── 5.5 (quadtree → R-tree) ← stretch, depends on 4.1

Phase 6 (depends on Phase 5)
  └── 6.3 (benchmark) ← depends on all optimisations

Phase 7 (can start after Phase 2; no algorithmic dependency)
  ├── 7.1 (axes/projection) ← independent
  ├── 7.2 (coastlines) ← independent
  ├── 7.3 (Hovmöller) ← independent
  └── 7.4 (polygon projection fix) ← depends on 4.1

Phase 8 (can start any time)
  ├── 8.1 (fix template docs) ← independent
  ├── 8.2 (docstrings) ← depends on final API shape (Phase 3+)
  └── 8.3 (algorithm overview) ← independent

Phase 9 (after everything else)
  ├── 9.1 (quadtree removal) = 5.5
  ├── 9.2 (parallel timesteps) ← depends on Phase 5 (no VTK)
  ├── 9.3 (xarray output) ← depends on final API shape
  └── 9.4 (YAML config) ← independent
```

---

## Appendix D: Edge Cases to Test

A checklist of boundary conditions that must have explicit test coverage.

| Edge Case | Where It Matters | Expected Behavior |
|-----------|-----------------|-------------------|
| Flat field (all zeros) | Extrema detection | Zero extrema found; no crash |
| Single grid cell above threshold | Clustering | One cluster with one point |
| Wave packet straddling 0°/360° | Association graph, polygon, tracking | Correct associations; polygon wraps; tracking connects |
| Wave packet at the North Pole | VTK mesh, distance computation | Distances are correct; no NaN |
| Wave packet at the South Pole | Stereographic projection | Works if hemisphere="south" |
| Two RWPs with identical peak scalar | Feature ID assignment | Distinct IDs; both in raster |
| Very weak field (all below threshold) | Identification | No RWPs; informative log message |
| Dense field (100+ extrema per region) | Clustering, AP/DBSCAN convergence | Completes in reasonable time; clusters are physically plausible |
| Single timestep (no tracking) | Tracking | Graceful no-op or informative error |
| Timestep with zero RWPs followed by timestep with RWPs | Tracking graph | No spurious edges; no crash |
| Very high resolution grid (0.25° global) | Performance | Completes identification in < 60s |
| Non-global domain (regional subset) | Periodicity assumptions | No wraparound artifacts |
| NaN values in input | Extrema detection | NaNs are ignored or raise clear error |
| Non-uniform latitude spacing | VTK mesh construction | Mesh handles it correctly |
| Longitude starting from -180 instead of 0 | `np.linspace(0, 360, ...)` assumption | Longitudes are correctly mapped |

---

## Appendix E: Complete `WaperConfig` After All Phases

```python
@dataclass(eq=False, frozen=True)
class WaperConfig:

    # --- User-required ---
    scalar_name: str
    latitude_label: str
    longitude_label: str
    time_label: str

    # --- Identification thresholds ---
    clip_value: float = 2.0
    extrema_threshold: float = 10.0
    node_pruning_threshold: float = 20.0
    edge_pruning_threshold: float = 3e-5
    max_edge_weight: float = 1.0
    min_longitude_separation: float = 6.0    # degrees
    max_scalar_value: float = 100.0          # m/s

    # --- Latitude bounds ---
    max_latitude: float | None = None
    min_latitude: float | None = None

    # --- Clustering ---
    cluster_eps_km: float = 500.0            # DBSCAN eps in km
    cluster_min_samples: int = 1             # DBSCAN min_samples

    # --- Polygon footprints ---
    polygon_clip_fraction: float = 3.0       # clip at peak / this
    hull_method: str = "per_node"            # "convex", "concave", "per_node"
    polygon_subsample: int = 5

    # --- Tracking ---
    track_pruning_threshold: float = 0.3     # km (distance threshold)
    hemisphere: str = "north"                # "north" or "south"
    raster_size: int = 512                   # pixels (if using quadtree)

    # --- Internal VTK labels ---
    vtk_latitude_label: str = "Latitude"
    vtk_longitude_label: str = "Longitude"
    vtk_region_label: str = "RegionId"

    # --- Debug ---
    debug: bool = False
```

```


## Assistant

Now let me verify the file was created correctly:

**Tool Call: Read file `conductor/waper_refactoring_spec.md` (lines 1-30)**
Status: Completed

```waper/conductor/waper_refactoring_spec.md
# WAPER Refactoring Specification

## Document Purpose

This specification captures **every** known issue, improvement, and enhancement for the WAPER (Rossby Wave Packet Extraction and Representation) package. Each item is written as a self-contained task with explicit step-by-step instructions so that any developer — or a code-generation LLM — can pick up a single task and execute it without needing to understand the full history of the project.

**How to use this document:** Work through the phases in order. Within each phase, tasks can often be parallelised (noted where possible). Every task ends with a "Definition of Done" checklist. Do not skip the testing tasks — they are load-bearing.

---

## Table of Contents

- [Phase 0: Scaffolding, CI, and Project Hygiene](#phase-0-scaffolding-ci-and-project-hygiene)
- [Phase 1: Testing Infrastructure](#phase-1-testing-infrastructure)
- [Phase 2: Critical Bug Fixes](#phase-2-critical-bug-fixes)
- [Phase 3: Algorithmic Improvements — Identification](#phase-3-algorithmic-improvements--identification)
- [Phase 4: Algorithmic Improvements — Tracking](#phase-4-algorithmic-improvements--tracking)
- [Phase 5: VTK-to-PyVista / SciPy Refactor](#phase-5-vtk-to-pyvista--scipy-refactor)
- [Phase 6: Performance Optimisation](#phase-6-performance-optimisation)
- [Phase 7: Visualisation Overhaul](#phase-7-visualisation-overhaul)
- [Phase 8: Documentation](#phase-8-documentation)
- [Phase 9: Stretch Goals](#phase-9-stretch-goals)
- [Appendix A: File Inventory](#appendix-a-file-inventory)
- [Appendix B: Magic Numbers Registry](#appendix-b-magic-numbers-registry)
- [Appendix C: Dependency Map](#appendix-c-dependency-map)

---

## Phase 0: Scaffolding, CI, and Project Hygiene


```


## Assistant

The spec is created. Let me verify its total length:

**Tool Call: wc -l conductor/waper_refactoring_spec.md**
Status: Completed

Terminal:
```
    1955 conductor/waper_refactoring_spec.md




```


## Assistant

The full specification has been created at `waper/conductor/waper_refactoring_spec.md`. Here's a summary of what it covers:

---

## Structure

The spec is organised into **10 phases** with **40+ tasks**, each with explicit step-by-step instructions and "Definition of Done" checklists:

| Phase | Focus | # Tasks |
|-------|-------|---------|
| **0** | Scaffolding, CI, project hygiene | 4 |
| **1** | Testing infrastructure (fixtures + unit tests) | 6 |
| **2** | Critical bug fixes | 6 |
| **3** | Algorithmic improvements — Identification | 6 |
| **4** | Algorithmic improvements — Tracking | 4 |
| **5** | VTK→PyVista/SciPy refactor | 5 |
| **6** | Performance optimisation | 3 |
| **7** | Visualisation overhaul | 4 |
| **8** | Documentation | 3 |
| **9** | Stretch goals | 4 |

## Key Design Decisions Captured

- **DBSCAN replaces Affinity Propagation** (Task 3.1) — noise points are discarded, `eps_km` has a clear physical interpretation, the two-point merge special case goes away
- **Tuple node IDs** (`("max", k)` / `("min", k)`) replace the integer/negation scheme (Task 2.2)
- **Monotonic integer feature IDs** replace `round(path_max, 2)` (Task 2.5)
- **Greedy max-weight independent set** replaces the non-transitive pairwise filter (Task 3.4)
- **DAG longest-path DP** replaces `nx.all_simple_paths` (Task 4.3)
- **Per-node convex hulls with union** replaces single convex hull (Task 4.1)
- **SciPy sparse graph Dijkstra** replaces VTK Dijkstra (Task 5.2)
- **scipy.ndimage filters** replace Python loops for extrema (Task 2.1)
- **KD-tree** replaces O(N²) nearest-neighbor in association graph (Task 2.4)

## Appendices

- **A: File Inventory** — every file with its issues
- **B: Magic Numbers Registry** — every hardcoded constant with proposed config location
- **C: Dependency Map** — ASCII graph showing task dependencies for parallelisation
- **D: Edge Cases** — 15 boundary conditions that must have test coverage
- **E: Complete `WaperConfig`** — the final shape of the config dataclass after all phases

Each task is written so that someone (or a less-capable LLM) can work on it in isolation — it names the exact files, the exact functions, provides code snippets, and lists what to verify.

