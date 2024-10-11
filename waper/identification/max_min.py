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