import numpy as np
import pyvista as pv
import vtk
from scipy.ndimage import maximum_filter

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
    lons = np.linspace(0, 360, len(longitudes))
    lats = latitudes
    grid_vtk = get_vtk_object_from_data_array(scalar_values, lons, lats, scalar_name)

    numpy_data = scalar_values.values
    r, c = numpy_data.shape

    local_max = maximum_filter(numpy_data, size=3, mode=["constant", "wrap"])
    is_max = (numpy_data == local_max).astype(float)

    vertex_identifiers = np.arange(1, r * c + 1, dtype=float)

    cell_number = grid_vtk.GetNumberOfCells()
    cell_id = np.arange(cell_number)

    grid_vtk.point_data["is max"] = is_max.ravel()
    grid_vtk.point_data["Vertex_id"] = vertex_identifiers
    grid_vtk.cell_data["{} Cell ID".format(scalar_name)] = cell_id

    return grid_vtk


from scipy.ndimage import minimum_filter


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
    lons = np.linspace(0, 360, len(longitudes))
    lats = latitudes
    grid_vtk = get_vtk_object_from_data_array(scalar_values, lons, lats, scalar_name)

    numpy_data = scalar_values.values
    r, c = numpy_data.shape

    local_min = minimum_filter(numpy_data, size=3, mode=["constant", "wrap"])
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


def extract_maxima_points(scalar_field, threshold, scalar_name):
    """Get data corresponding to identified maxima

    Args:
        scalar_field (vtk.vtkUnstructuredGrid): vtk object containing clipped dataset
        threshold (float): discard maxima below threshold
        scalar_name (string): name of variable in scalar_field

    Returns:
        vtk.vtkUnstructuredGrid: array containing identified maxima
    """
    if scalar_field.n_points == 0:
        return scalar_field

    return scalar_field.extract_points(
        (
            (scalar_field.point_data["is max"] == 1)
            & (scalar_field.point_data[scalar_name] > threshold)
        ),
        include_cells=False,
    )


def extract_minima_points(scalar_field, threshold, scalar_name):
    """Get data corresponding to identified minima

    Args:
        scalar_field (vtk.vtkUnstructuredGrid): vtk object containing clipped dataset
        threshold (float): discard minima above threshold
        scalar_name (string): name of variable in scalar_field

    Returns:
        vtk.vtkUnstructuredGrid: array containing identified minima
    """
    if scalar_field.n_points == 0:
        return scalar_field

    return scalar_field.extract_points(
        (
            (scalar_field.point_data["is min"] == 1)
            & (scalar_field.point_data[scalar_name] < threshold)
        ),
        include_cells=False,
    )


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
