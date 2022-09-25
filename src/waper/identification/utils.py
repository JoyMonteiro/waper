import pyvista as pv
import vtk
import numpy as np
import math

RADIUS_EARTH = 6371.0

def get_point_data_label(scalar_name):
    return scalar_name

def get_cell_data_label(scalar_name):
    return "Cell Value {}".format(scalar_name)

def get_vtk_object_from_data_array(data_array, array_name="v"):
    """Get vtk object from xarray dataArray

    Args:
        longitude (array): coordinates along zonal direction
        latitude (array): coordinates along meridional direction
        scalar_values (array): scalar field to convert to vtk object
    """

    rect = pv.RectilinearGrid(data_array[data_array.dims[1]].values, data_array[data_array.dims[0]].values)
    scalar = data_array.values.ravel()
    rect.point_data[array_name] = scalar

    return rect

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