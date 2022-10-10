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

    min_lon = 360
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

                if min_lon > np.min(lons):  # store location of most westward cluster.
                    min_lon = np.min(lons)

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

                if min_lon > np.min(lons):  # store location of most westward cluster.
                    min_lon = np.min(lons)

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
