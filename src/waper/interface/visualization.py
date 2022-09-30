import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyvista as pv


def _plot_clusters(input_data, maxima_points, 
                   minima_points, max_pt_dict, min_pt_dict,
                   vtk_lon_label, vtk_lat_label, vtk_region_label
                   ):

    ax = plt.subplot(211, projection=ccrs.PlateCarree())

    input_data.plot.contour(
        ax=ax,
        levels=12,
        transform=ccrs.PlateCarree(central_longitude=180),
        labels=True,
        colors="k",
        linewidth=1,
        zorder=1,
    )

    out = pv.wrap(maxima_points)
    for lon, lat, region_id in zip(
        out[vtk_lon_label], out[vtk_lat_label], out[vtk_region_label]
    ):

        ax.scatter(
            lon,
            lat,
            color=plt.cm.tab20b.colors[region_id % 20],
            transform=ccrs.PlateCarree(central_longitude=180),
            zorder=10,
        )

    out = pv.wrap(minima_points)
    for lon, lat, region_id in zip(
        out[vtk_lon_label], out[vtk_lat_label], out[vtk_region_label]
    ):

        ax.scatter(
            lon,
            lat,
            color=plt.cm.tab20b.colors[region_id % 20],
            marker="*",
            transform=ccrs.PlateCarree(central_longitude=180),
            zorder=10,
        )

    ax = plt.subplot(212, projection=ccrs.PlateCarree())

    input_data.plot.contour(
        ax=ax,
        levels=12,
        colors="k",
        transform=ccrs.PlateCarree(central_longitude=180),
        labels=True,
    )

    for cluster_id, points in max_pt_dict.items():

        if len(points) > 1:
            for point in points:
                ax.scatter(
                    point[0],
                    point[1],
                    color=plt.cm.tab20.colors[int(cluster_id) % 20],
                    transform=ccrs.PlateCarree(central_longitude=180),
                    zorder=10,
                )

    for cluster_id, points in min_pt_dict.items():

        if len(points) > 1:
            for point in points:
                ax.scatter(
                    point[0],
                    point[1],
                    marker="*",
                    color=plt.cm.tab20.colors[int(cluster_id) % 20],
                    transform=ccrs.PlateCarree(central_longitude=180),
                    zorder=10,
                )
                
    plt.tight_layout()
    return ax

def _plot_graph(rwp_graph, scalar_data):
    
    ax = plt.subplot(projection=ccrs.PlateCarree())


    scalar_data.plot.contour(ax=ax, levels=12,
                                   transform=ccrs.PlateCarree(central_longitude=180), labels=True, colors='k',
                                    linewidth=1, zorder=1
                                  )
    
    for node in rwp_graph.nodes:
        coords = rwp_graph.nodes[node]['coords']
        ax.scatter(coords[0], coords[1],  color='k', transform=ccrs.PlateCarree(central_longitude=180))
    
    for edge in rwp_graph.edges:
        node1_coords = rwp_graph.nodes[edge[0]]['coords']
        node2_coords = rwp_graph.nodes[edge[1]]['coords']
        
        ax.plot([node1_coords[0], node2_coords[0]],
                [node1_coords[1], node2_coords[1]],
                color='k', transform=ccrs.PlateCarree(central_longitude=180)
            )
        
    plt.tight_layout()
    return ax
        
def _plot_polygons(poly_list, scalar_data, sample_points_list, plot_samples=False):
    ax = plt.subplot(projection=ccrs.PlateCarree(central_longitude=180))
    
    scalar_data.plot.contour(ax=ax, levels=12,
                                   transform=ccrs.PlateCarree(central_longitude=0), labels=True, colors='k',
                                    linewidth=1, zorder=1
                                  )
    
    for poly, sample_points in zip(poly_list, sample_points_list):
        
        lons, lats = poly.exterior.coords.xy

        ax.plot(lons, lats, transform=ccrs.PlateCarree(central_longitude=0))
    
        for lon, lat in zip(lons, lats):
            ax.scatter(lon, lat, transform=ccrs.PlateCarree(central_longitude=0), color='r', s=30, zorder=100)
        
        if plot_samples:
            for lon, lat in sample_points:
                ax.scatter(lon, lat, transform=ccrs.PlateCarree(central_longitude=0), color='b', s=5)
            
    plt.tight_layout()
    return ax