from ..identification import topology
import numpy as np

from shapely.geometry import MultiPoint
from rasterio import features

WAPER_SUBSAMPLE = 5
WAPER_IMAGE_SIZE = 512
WAPER_CLUSTER_WIDTH = 60


def get_consistent_longitudes(longitude_array, min_lon):
    """fix issue with wrap around of longitudes

    Args:
        longitude_array (list): list of longitudes
    """

    final_array = np.array(longitude_array)
    if (np.max(longitude_array) - np.min(longitude_array)) > WAPER_CLUSTER_WIDTH:
        print("Inconsistent, fixing")
        for i in range(len(final_array)):
            print(final_array[i])
            if final_array[i] < min_lon:
                print("*" * 10)
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

    return lons, lats, values


def get_polygon_for_rwp_path(path, assoc_graph, scalar_data, scalar_name):
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

    clip_threshold = path_max / 2.0

    max_clipped_region = topology.identify_connected_regions(
        scalar_data.clip_scalar(scalars=scalar_name, value=clip_threshold, invert=False).clean()
    )

    min_clipped_region = topology.identify_connected_regions(
        scalar_data.clip_scalar(scalars=scalar_name, value=-clip_threshold, invert=True).clean()
    )

    list_rwp_points = []
    list_lons = []
    list_lats = []
    list_values = []

    min_lon = 0
    for node in path:
        if node > 0:
            out = get_region_points_and_values(
                assoc_graph, node, max_clipped_region, clip_threshold, scalar_name
            )
            if out:
                lons, lats, values = out

                if node == path[0]:  # store location of most westward cluster.
                    min_lon = np.min(lons)

                lons = get_consistent_longitudes(lons, min_lon)
                list_lons.extend(lons)
                list_lats.extend(lats)
                list_values.extend(values)

                lons = lons[::WAPER_SUBSAMPLE]
                lats = lats[::WAPER_SUBSAMPLE]
                list_rwp_points.extend(list(zip(lons, lats)))
            else:
                pass
        else:
            out = get_region_points_and_values(
                assoc_graph, node, min_clipped_region, clip_threshold, scalar_name
            )
            if out:
                lons, lats, values = out

                if node == path[0]:  # store location of most westward cluster.
                    min_lon = np.min(lons)

                lons = get_consistent_longitudes(lons, min_lon)
                list_lons.extend(lons)
                list_lats.extend(lats)
                list_values.extend(values)

                lons = lons[::WAPER_SUBSAMPLE]
                lats = lats[::WAPER_SUBSAMPLE]
                list_rwp_points.extend(list(zip(lons, lats)))

            else:
                pass

    polygon_id = round(path_max, 2)

    weighted_latitude = np.average(list_lats, weights=list_values)
    weighted_longitude = np.average(list_lons, weights=list_values)

    return (
        MultiPoint(list_rwp_points).convex_hull,
        polygon_id,
        list_rwp_points,
        weighted_longitude,
        weighted_latitude,
    )


def rasterize_all_rwps(paths, assoc_graph, scalar_data):
    """Get a rasterized image containing all rwp polygons

    Args:
        paths (list): list of all rwp representative paths
        assoc_graph (nx.Graph): association graph
        scalar_data (pv.PolyData): scalar data corresponding to association graph

    Returns:
        tuple: raster image of all polygons and the list of polygons itself
    """

    polygon_list = []

    for path in paths:
        polygon_list.append(get_polygon_for_rwp_path(path, assoc_graph, scalar_data))

    return (
        features.rasterize(
            ((g, i) for g, i in polygon_list),
            out_shape=(WAPER_IMAGE_SIZE, WAPER_IMAGE_SIZE),
            all_touched=True,
        ),
        polygon_list,
    )
