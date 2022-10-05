import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pyproj import transform
import pyvista as pv
import numpy as np

from ..tracking.rwp_polygon import WAPER_X_BOUNDS, WAPER_Y_BOUNDS


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

        # ax.scatter(
        #     lon,
        #     lat,
        #     color=plt.cm.tab20.colors[region_id % 20],
        #     transform=ccrs.PlateCarree(central_longitude=180),
        #     zorder=10,
        # )

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

        # ax.scatter(
        #     lon,
        #     lat,
        #     color=plt.cm.tab20.colors[region_id % 20],
        #     marker="*",
        #     transform=ccrs.PlateCarree(central_longitude=180),
        #     zorder=10,
        # )

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
            # ax.scatter(
            #     point[0],
            #     point[1],
            #     color=plt.cm.tab20.colors[int(cluster_id) % 20],
            #     transform=ccrs.PlateCarree(central_longitude=180),
            #     zorder=10,
            # )
            ax.annotate(
                str(cluster_id),
                (point[0], point[1]),
                bbox=dict(boxstyle="round", fc="white", ec="b"),
                transform=ccrs.PlateCarree(central_longitude=0),
            )

    for cluster_id, points in min_pt_dict.items():

        for point in points:
            # ax.scatter(
            #     point[0],
            #     point[1],
            #     marker="*",
            #     color=plt.cm.tab20.colors[int(cluster_id) % 20],
            #     transform=ccrs.PlateCarree(central_longitude=180),
            #     zorder=10,
            # )
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


def _plot_graph(rwp_graph, scalar_data):

    ax = plt.subplot(projection=ccrs.Orthographic(central_longitude=180, central_latitude=90))

    scalar_data.plot.contour(
        ax=ax,
        levels=12,
        transform=ccrs.PlateCarree(central_longitude=0),
        labels=True,
        colors="k",
        linewidths=2,
        zorder=1,
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


def _plot_rwp_paths(
    rwp_graph,
    paths,
    scalar_data=None,
    path_transform=ccrs.PlateCarree(),
    map_projection=ccrs.Orthographic(central_longitude=180, central_latitude=90),
):

    ax = plt.subplot(projection=map_projection)

    colors = plt.cm.tab20.colors

    if scalar_data != None:
        scalar_data.plot.contour(
            ax=ax,
            levels=12,
            transform=ccrs.PlateCarree(central_longitude=0),
            labels=True,
            colors="k",
            linewidths=2,
            zorder=1,
        )

    for index, path in enumerate(paths):
        for node in path:
            coords = rwp_graph.nodes[node]["coords"]
            ax.scatter(coords[0], coords[1], color="r", transform=path_transform)

        for edge in [(path[i], path[i + 1]) for i in range(len(path) - 1)]:
            node1_coords = rwp_graph.nodes[edge[0]]["coords"]
            node2_coords = rwp_graph.nodes[edge[1]]["coords"]

            delta_coord = 0
            if node1_coords[0] - node2_coords[0] > 180:
                delta_coord = 360

            ax.plot(
                [node1_coords[0], node2_coords[0] + delta_coord],
                [node1_coords[1], node2_coords[1]],
                color=colors[index%20],
                transform=path_transform,
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
):
    ax = plt.subplot(projection=ccrs.Stereographic(central_longitude=0, central_latitude=90))

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

        ax.plot(lons, lats)

        for lon, lat in zip(lons, lats):
            ax.scatter(lon, lat, color="r", s=30, zorder=100)

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
                transform=ccrs.PlateCarree(central_longitude=0), zorder=1000
            )

    if plot_samples:
        for sample_points in sample_points_list:
            for lon, lat in sample_points:
                ax.scatter(lon, lat, color="b", s=5)

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

    # ax.set_global()
    plt.tight_layout()
    return ax
