import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

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

    cell_id = np.arange(grid_vtk.n_cells)

    grid_vtk.point_data["is max"] = is_max.ravel()
    grid_vtk.point_data["Vertex_id"] = vertex_identifiers
    grid_vtk.cell_data[f"{scalar_name} Cell ID"] = cell_id

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

    cell_id = np.arange(grid_vtk.n_cells)

    grid_vtk.point_data["is min"] = is_min.ravel()
    grid_vtk.point_data["Vertex_id"] = vertex_identifiers
    grid_vtk.cell_data[f"{scalar_name} Cell ID"] = cell_id

    return grid_vtk


def extract_maxima_points(scalar_field, threshold, scalar_name):
    """Get data corresponding to identified maxima

    Args:
        scalar_field (pv.UnstructuredGrid): pyvista object containing clipped dataset
        threshold (float): discard maxima below threshold
        scalar_name (string): name of variable in scalar_field

    Returns:
        pv.UnstructuredGrid: points at the identified maxima
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
        scalar_field (pv.UnstructuredGrid): pyvista object containing clipped dataset
        threshold (float): discard minima above threshold
        scalar_name (string): name of variable in scalar_field

    Returns:
        pv.UnstructuredGrid: points at the identified minima
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
