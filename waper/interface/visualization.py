import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyvista as pv
import numpy as np
from xarray import DataArray
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import MultiPolygon

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

_PLATE_CARREE = ccrs.PlateCarree(central_longitude=0)
_STEREO_NH = ccrs.Stereographic(central_longitude=0, central_latitude=90)


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
    ax.coastlines(linewidth=0.5, color="gray")

    input_data.plot.contourf(
        ax=ax,
        levels=12,
        transform=_PLATE_CARREE,
        zorder=1,
        cmap=NLDivCmap,
        add_colorbar=False,
        add_labels=False,
    )

    input_data.plot.contour(
        ax=ax,
        levels=[-clip_value, clip_value],
        transform=_PLATE_CARREE,
        colors="r",
        linewidths=2,
        zorder=2,
        add_labels=False,
    )

    out = pv.wrap(maxima_points)
    for lon, lat, region_id in zip(
        out[vtk_lon_label], out[vtk_lat_label], out[vtk_region_label]
    ):

        ax.annotate(
            str(region_id + 1),
            (lon, lat),
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="b", alpha=0.7),
            transform=_PLATE_CARREE,
        )

    out = pv.wrap(minima_points)
    for lon, lat, region_id in zip(
        out[vtk_lon_label], out[vtk_lat_label], out[vtk_region_label]
    ):

        ax.annotate(
            str(-region_id - 1),
            (lon, lat),
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="r", alpha=0.7),
            transform=_PLATE_CARREE,
        )

    ax.set_title("Extrema by region", fontsize=9)

    ax = plt.subplot(212, projection=ccrs.PlateCarree(central_longitude=180))
    ax.coastlines(linewidth=0.5, color="gray")

    input_data.plot.contourf(
        ax=ax,
        levels=12,
        transform=_PLATE_CARREE,
        zorder=1,
        cmap=NLDivCmap,
        add_colorbar=False,
        add_labels=False,
    )

    for cluster_id, points in max_pt_dict.items():
        for point in points:
            ax.annotate(
                str(cluster_id),
                (point[0], point[1]),
                fontsize=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="b", alpha=0.7),
                transform=_PLATE_CARREE,
            )

    for cluster_id, points in min_pt_dict.items():
        for point in points:
            ax.annotate(
                str(-cluster_id),
                (point[0], point[1]),
                fontsize=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="r", alpha=0.7),
                transform=_PLATE_CARREE,
            )

    ax.set_title("Extrema by cluster", fontsize=9)
    plt.tight_layout()
    return ax


def _plot_graph(rwp_graph, scalar_data=None, ax=None):

    if ax is None:
        ax = plt.subplot(projection=ccrs.PlateCarree(central_longitude=180))

    ax.coastlines(linewidth=0.5, color="gray")

    if isinstance(scalar_data, DataArray):
        scalar_data.plot.contourf(
            ax=ax,
            levels=12,
            transform=_PLATE_CARREE,
            zorder=1,
            cmap=NLDivCmap,
            add_colorbar=True,
            add_labels=False,
            cbar_kwargs=dict(
                orientation='horizontal',
                shrink=0.6,
                aspect=30
            )
        )

        scalar_data.plot.contour(
            ax=ax,
            levels=12,
            transform=_PLATE_CARREE,
            colors="k",
            linewidths=0.5,
            zorder=2,
            add_labels=False,
        )

    for node in rwp_graph.nodes:
        coords = rwp_graph.nodes[node]["coords"]
        ax.scatter(
            coords[0], coords[1], color="r", s=20, zorder=5,
            transform=_PLATE_CARREE,
        )

    for edge in rwp_graph.edges:
        node1_coords = rwp_graph.nodes[edge[0]]["coords"]
        node2_coords = rwp_graph.nodes[edge[1]]["coords"]

        ax.plot(
            [node1_coords[0], node2_coords[0]],
            [node1_coords[1], node2_coords[1]],
            color="b", linewidth=1.5, zorder=4,
            transform=ccrs.Geodetic(),
        )

    plt.tight_layout()
    return ax


def _plot_rwp_paths(rwp_graph, paths, scalar_data=None, ax=None):

    if ax is None:
        ax = plt.subplot(projection=ccrs.PlateCarree(central_longitude=180))

    ax.coastlines(linewidth=0.5, color="gray")

    colors = plt.cm.tab10.colors

    if isinstance(scalar_data, DataArray):
        scalar_data.plot.contourf(
            ax=ax,
            levels=11,
            transform=_PLATE_CARREE,
            zorder=1,
            cmap=NLDivCmap,
            add_colorbar=True,
            add_labels=False,
            cbar_kwargs=dict(
                orientation='horizontal',
                shrink=0.6,
                aspect=30
            )
        )

        scalar_data.plot.contour(
            ax=ax,
            levels=12,
            transform=_PLATE_CARREE,
            colors="k",
            linewidths=0.5,
            zorder=2,
            add_labels=False,
        )

    for index, path in enumerate(paths):
        path_color = colors[index % len(colors)]

        for node in path:
            coords = rwp_graph.nodes[node]["coords"]
            color = 'r' if node[0] == 'max' else 'b'

            ax.scatter(
                coords[0], coords[1], color=color, s=25, zorder=6,
                edgecolors='k', linewidths=0.3,
                transform=_PLATE_CARREE,
            )

        for edge in [(path[i], path[i + 1]) for i in range(len(path) - 1)]:
            node1_coords = rwp_graph.nodes[edge[0]]["coords"]
            node2_coords = rwp_graph.nodes[edge[1]]["coords"]

            ax.plot(
                [node1_coords[0], node2_coords[0]],
                [node1_coords[1], node2_coords[1]],
                color=path_color, linewidth=2.5, zorder=5,
                transform=ccrs.Geodetic(),
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
    poly_colors=None,
):

    if ax is None:
        ax = plt.subplot(projection=_STEREO_NH)

    ax.coastlines(linewidth=0.5, color="gray")
    ax.set_extent([-180, 180, 20, 90], crs=_PLATE_CARREE)

    if scalar_data is not None:
        scalar_data.plot.contourf(
            ax=ax,
            levels=12,
            transform=_PLATE_CARREE,
            zorder=1,
            cmap=NLDivCmap,
            add_colorbar=False,
            add_labels=False,
        )

    default_colors = plt.cm.tab10.colors

    for idx, poly in enumerate(poly_list):
        parts = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
        if poly_colors is not None:
            color = poly_colors[idx]
        else:
            color = default_colors[idx % len(default_colors)]

        for part in parts:
            lons, lats = part.exterior.coords.xy

            ax.fill(
                list(lons), list(lats),
                facecolor=color, alpha=0.3, edgecolor=color,
                linewidth=1.5, zorder=3,
                transform=_STEREO_NH,
            )

    if weighted_lat_list is not None:
        for index, coords in enumerate(list(zip(weighted_lon_list, weighted_lat_list))):
            lon, lat = coords
            ax.scatter(
                lon, lat,
                transform=_PLATE_CARREE,
                s=50, color="green", zorder=100,
                edgecolors='k', linewidths=0.5,
            )

            ax.annotate(
                str(index),
                (lon, lat),
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="b"),
                transform=_PLATE_CARREE,
                zorder=1000,
            )

    if plot_samples:
        for sample_points in sample_points_list:
            for lon, lat in sample_points:
                ax.scatter(
                    lon, lat,
                    color="b", s=5,
                    transform=_STEREO_NH,
                )

    plt.tight_layout()
    return ax


def _plot_raster(raster_data):
    ax = plt.subplot(projection=_STEREO_NH)
    ax.coastlines(linewidth=0.5, color="gray")
    ax.set_extent([-180, 180, 20, 90], crs=_PLATE_CARREE)

    ax.imshow(
        np.ma.array(raster_data, mask=(raster_data == 0)),
        origin="lower",
        cmap="tab20b",
        extent=(WAPER_X_BOUNDS[1], WAPER_X_BOUNDS[0], WAPER_Y_BOUNDS[1], WAPER_Y_BOUNDS[0]),
        alpha=0.7,
    )

    plt.tight_layout()
    return ax
