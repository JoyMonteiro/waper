import pyvista as pv


def get_vtk_object_from_scalar_data(longitude, latitude, scalar_values, array_name="v"):
    """Get vtk object from plain numpy array

    Args:
        longitude (array): coordinates along zonal direction
        latitude (array): coordinates along meridional direction
        scalar_values (array): scalar field to convert to vtk object
    """

    rect = pv.RectilinearGrid(longitude, latitude)
    scalar = scalar_values[::-1, :].ravel()
    rect.point_arrays[array_name] = scalar

    return rect


def extract_position_ids_minima(scalar_field, threshold, scalar_name):

    pos_min_ids = vtk.vtkIdTypeArray()
    num_pts = scalar_field.GetNumberOfPoints()
    is_min_arr = scalar_field.GetPointData().GetArray("is min")
    scalar_arr = scalar_field.GetPointData().GetArray(scalar_name)
    
    for i in range(num_pts):
        if is_min_arr.GetTuple1(i) == 1 and scalar_arr.GetTuple1(i) <= threshold:
            pos_min_ids.InsertNextValue(i)
    return pos_min_ids


def extract_position_ids_maxima(scalar_field, threshold, scalar_name):

    pos_max_ids = vtk.vtkIdTypeArray()
    num_pts = scalar_field.GetNumberOfPoints()
    is_max_arr = scalar_field.GetPointData().GetArray("is max")
    scalar_arr = scalar_field.GetPointData().GetArray(scalar_name)
    
    for i in range(num_pts):
        if is_max_arr.GetTuple1(i) == 1 and scalar_arr.GetTuple1(i) >= threshold:
            pos_max_ids.InsertNextValue(i)
    return pos_max_ids
