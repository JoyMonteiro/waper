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