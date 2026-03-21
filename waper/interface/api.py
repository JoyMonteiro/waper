from dataclasses import dataclass

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from networkx import Graph
from numpy import ndarray
from pyvista import PolyData
from tqdm import tqdm
from xarray import DataArray

from ..identification import max_min, rwp_graph, topology, utils
from ..tracking import quadtree, rwp_polygon, tracking_graph
from .visualization import (
    _plot_clusters,
    _plot_graph,
    _plot_polygons,
    _plot_raster,
    _plot_rwp_paths,
)


@dataclass(eq=False, frozen=True)
class WaperConfig:
    debug: bool
    scalar_name: str
    latitude_label: str
    longitude_label: str
    time_label: str

    clip_value: float
    extrema_threshold: float

    max_latitude: float
    min_latitude: float

    node_pruning_threshold: float
    edge_pruning_threshold: float
    max_edge_weight: float

    track_pruning_threshold: float

    cluster_max_eps_km: float = 3000.0
    cluster_min_samples: int = 2
    cluster_xi: float = 0.15
    min_longitude_separation: float = 6.0
    max_aspect_ratio: float = 1.5
    hull_method: str = "per_node"  # "per_node" | "convex" | "concave"
    hemisphere: str = "north"  # "north" | "south"
    penalty_length_scale_km: float = 2000.0

    vtk_latitude_label: str = "Latitude"
    vtk_longitude_label: str = "Longitude"
    vtk_region_label: str = "RegionId"


@dataclass(eq=False)
class WaperSingleTimestepData:
    input_data: DataArray

    vtk_data: PolyData

    number_max_clusters: int
    number_min_clusters: int

    max_cluster_info: dict
    min_cluster_info: dict

    all_minima: PolyData
    all_maxima: PolyData

    association_graph: Graph
    pruned_graph: Graph

    identified_rwp_paths: list

    rwp_info: dict

    raster_data: ndarray
    raster_features: list
    quadtree: Graph

    def __init__(self, input_data: DataArray, config: WaperConfig) -> None:
        self.input_data = input_data
        self.vtk_data = utils.get_vtk_object_from_data_array(
            input_data,
            input_data[config.longitude_label],
            input_data[config.latitude_label],
            array_name=config.scalar_name,
        )
        self.rwp_info = {}
        return


import logging

logger = logging.getLogger(__name__)


def _identify_rwps(
    scalar_data: DataArray, config: WaperConfig
) -> WaperSingleTimestepData:

    input_data = scalar_data
    latitude = input_data[config.latitude_label].values
    longitude = input_data[config.longitude_label].values

    time_step_data = WaperSingleTimestepData(input_data=input_data, config=config)
    # Identify and cluster maxima

    data_with_maxima = max_min.add_maxima_data(
        input_data, config.scalar_name, longitude, latitude
    )

    if config.min_latitude:
        data_with_maxima = data_with_maxima.clip_scalar(
            scalars=config.vtk_latitude_label, invert=False, value=config.min_latitude
        )

    if config.max_latitude:
        data_with_maxima = data_with_maxima.clip_scalar(
            scalars=config.vtk_latitude_label, invert=True, value=config.max_latitude
        )

    clipped_data_with_maxima = data_with_maxima.clip_scalar(
        scalars=config.scalar_name, invert=False, value=config.clip_value
    )

    connectivity = topology.identify_connected_regions(clipped_data_with_maxima)

    maxima_points = max_min.extract_maxima_points(
        connectivity, config.extrema_threshold, config.scalar_name
    )

    clustered_points = topology.cluster_extrema(
        data_with_maxima, connectivity, maxima_points, config.scalar_name, sign=1,
        max_eps_km=config.cluster_max_eps_km, min_samples=config.cluster_min_samples,
        xi=config.cluster_xi, penalty_length_scale_km=config.penalty_length_scale_km,
    )

    (
        cluster_max_arr,
        cluster_max_point,
        max_pt_dict,
        num_max_clusters,
    ) = topology.max_cluster_assign(clustered_points, config.scalar_name)

    time_step_data.all_maxima = clustered_points
    time_step_data.number_max_clusters = num_max_clusters
    time_step_data.max_cluster_info = max_pt_dict

    # Identify and cluster minima

    data_with_minima = max_min.add_minima_data(
        input_data, config.scalar_name, longitude, latitude
    )

    if config.max_latitude:
        data_with_minima = data_with_minima.clip_scalar(
            scalars=config.vtk_latitude_label, invert=True, value=config.max_latitude
        )

    if config.min_latitude:
        data_with_minima = data_with_minima.clip_scalar(
            scalars=config.vtk_latitude_label, invert=False, value=config.min_latitude
        )

    clipped_data_with_minima = data_with_minima.clip_scalar(
        scalars=config.scalar_name, value=-config.clip_value, invert=True
    )

    connectivity = topology.identify_connected_regions(clipped_data_with_minima)

    minima_points = max_min.extract_minima_points(
        connectivity, -config.extrema_threshold, config.scalar_name
    )

    clustered_points = topology.cluster_extrema(
        data_with_minima, connectivity, minima_points, config.scalar_name, sign=-1,
        max_eps_km=config.cluster_max_eps_km, min_samples=config.cluster_min_samples,
        xi=config.cluster_xi, penalty_length_scale_km=config.penalty_length_scale_km,
    )

    (
        cluster_min_arr,
        cluster_min_point,
        min_pt_dict,
        num_min_clusters,
    ) = topology.min_cluster_assign(clustered_points, config.scalar_name)

    time_step_data.all_minima = clustered_points
    time_step_data.number_min_clusters = num_min_clusters
    time_step_data.min_cluster_info = min_pt_dict

    # Compute and Prune Association Graph

    zero_isocontour = time_step_data.vtk_data.contour([0], scalars=config.scalar_name)
    time_step_data.association_graph = rwp_graph.compute_association_graph(
        time_step_data.all_maxima, time_step_data.all_minima, zero_isocontour, config.scalar_name
    )

    node_pruned_graph = rwp_graph.prune_association_graph_nodes(
        time_step_data.association_graph, scalar_threshold=config.node_pruning_threshold
    )

    time_step_data.pruned_graph = rwp_graph.prune_association_graph_edges(
        node_pruned_graph, config.edge_pruning_threshold, config.max_edge_weight,
        config.min_longitude_separation, config.max_aspect_ratio,
    )

    time_step_data.identified_rwp_paths = rwp_graph.get_ranked_paths(
        time_step_data.pruned_graph, config.max_edge_weight
    )

    for index, path in enumerate(time_step_data.identified_rwp_paths):
        (
            polygon,
            sample_points,
            weighted_lon,
            weighted_lat,
        ) = rwp_polygon.get_polygon_for_rwp_path(
            path,
            time_step_data.pruned_graph,
            time_step_data.vtk_data,
            config.scalar_name,
            config.min_latitude,
            config.max_latitude,
            hull_method=config.hull_method,
            hemisphere=config.hemisphere,
        )
        
        rwp_id = index + 1
        
        time_step_data.rwp_info[tuple(path)] = {
            # "path": path,
            "polygon": polygon,
            "rwp_id": rwp_id,
            "sample_points": sample_points,
            "weighted_longitude": weighted_lon,
            "weighted_latitude": weighted_lat,
        }

    list_polygons = []
    for path in time_step_data.identified_rwp_paths:
        list_polygons.append(
            (
                time_step_data.rwp_info[tuple(path)]["polygon"],
                time_step_data.rwp_info[tuple(path)]["rwp_id"],
            )
        )

    if len(list_polygons) == 0:
        logger.warning("No RWPs found at this timestep. Consider adjusting thresholds.")

    time_step_data.raster_data = rwp_polygon.rasterize_all_rwps(list_polygons, hemisphere=config.hemisphere)

    if time_step_data.raster_data is None:
        time_step_data.raster_features = {0}
        time_step_data.quadtree = None
    else:
        features = set(np.unique(time_step_data.raster_data))
        features.add(0)
        time_step_data.raster_features = features
        time_step_data.quadtree = quadtree.create_quadtree(time_step_data.raster_data)

    return time_step_data


def _track_rwps(time_step_data, num_time_steps):

    return tracking_graph.build_tracking_graph(time_step_data, num_time_steps)


class Waper:
    def __init__(
        self,
        data_array,
        scalar_name,
        latitude_label,
        longitude_label,
        time_label,
        clip_value=2,
        extrema_threshold=10,
        max_latitude=None,
        min_latitude=None,
        node_pruning_threshold=20,
        edge_pruning_threshold=3e-5,
        track_pruning_threshold=0.3,
        max_edge_weight=1,
        debug=False,
    ) -> None:

        self._config = WaperConfig(
            scalar_name=scalar_name,
            latitude_label=latitude_label,
            longitude_label=longitude_label,
            time_label=time_label,
            clip_value=clip_value,
            extrema_threshold=extrema_threshold,
            max_latitude=max_latitude,
            min_latitude=min_latitude,
            node_pruning_threshold=node_pruning_threshold,
            edge_pruning_threshold=edge_pruning_threshold,
            track_pruning_threshold=track_pruning_threshold,
            max_edge_weight=max_edge_weight,
            debug=debug,
        )

        self.data_array = data_array
        self._num_time_steps = len(data_array[time_label])
        self._time_step_data = []

        if debug:
            logging.basicConfig(level=logging.DEBUG)

    def identify_rwps(self):

        for i in tqdm(range(self._num_time_steps)):
            self._time_step_data.append(
                _identify_rwps(
                    self.data_array[self._config.scalar_name][i], self._config
                )
            )

    def track_rwps(self, num_time_steps=None):

        self._tracking_graph = _track_rwps(self._time_step_data, num_time_steps)
        self._pruned_tracking_graph = tracking_graph.prune_tracking_graph(
            self._tracking_graph, self._config.track_pruning_threshold
        )

    def plot_clusters(self, time_index):

        time_step_data = self._time_step_data[time_index]
        return _plot_clusters(
            time_step_data.input_data,
            time_step_data.all_maxima,
            time_step_data.all_minima,
            time_step_data.max_cluster_info,
            time_step_data.min_cluster_info,
            self._config.vtk_longitude_label,
            self._config.vtk_latitude_label,
            self._config.vtk_region_label,
            self._config.clip_value,
        )

    def plot_association_graph(self, time_index, ax=None):
        time_step_data = self._time_step_data[time_index]

        return _plot_graph(
            time_step_data.association_graph, time_step_data.input_data, ax=ax
        )

    def plot_pruned_graph(self, time_index, ax=None):
        time_step_data = self._time_step_data[time_index]

        return _plot_graph(
            time_step_data.pruned_graph, time_step_data.input_data, ax=ax
        )

    def plot_rwp_graphs(self, time_index, ax=None, plot_scalar_data=True):
        time_step_data = self._time_step_data[time_index]

        field = None
        if plot_scalar_data:
            field = time_step_data.input_data

        return _plot_rwp_paths(
            time_step_data.pruned_graph,
            time_step_data.identified_rwp_paths,
            field,
            ax=ax,
        )

    def plot_rwp_polygons(self, time_index, plot_samples=False, ax=None):
        time_step_data = self._time_step_data[time_index]

        poly_list = [
            rwp_info["polygon"] for rwp_info in time_step_data.rwp_info.values()
        ]
        sample_points_list = [
            rwp_info["sample_points"] for rwp_info in time_step_data.rwp_info.values()
        ]

        weighted_lon_list = [
            rwp_info["weighted_longitude"]
            for rwp_info in time_step_data.rwp_info.values()
        ]

        weighted_lat_list = [
            rwp_info["weighted_latitude"]
            for rwp_info in time_step_data.rwp_info.values()
        ]

        return _plot_polygons(
            poly_list,
            time_step_data.input_data,
            sample_points_list,
            weighted_lon_list,
            weighted_lat_list,
            plot_samples=plot_samples,
            ax=ax,
        )

    def plot_raster(self, time_index):
        time_step_data = self._time_step_data[time_index]

        return _plot_raster(time_step_data.raster_data)

    def plot_tracks(self, threshold=None):
        pruned = tracking_graph.prune_tracking_graph(
            self._tracking_graph, threshold=threshold
        )
        paths = tracking_graph.get_track_paths(pruned)
        return _plot_rwp_paths(
            pruned,
            paths,
            None,
        )

    def plot_track_polygons(self, path, plot_samples=False, ax=None):

        poly_list = []
        sample_points_list = []
        weighted_lon_list = []
        weighted_lat_list = []
        time_indices = []
        for node in path:
            time_step_data = self._time_step_data[node[0]]

            for rwp in time_step_data.rwp_info.values():
                if abs(rwp["rwp_id"] - node[1]) < 1e-2:
                    poly_list.append(rwp["polygon"])
                    sample_points_list.append(rwp["sample_points"])
                    weighted_lon_list.append(rwp["weighted_longitude"])
                    weighted_lat_list.append(rwp["weighted_latitude"])
                    time_indices.append(node[0])

        cmap = plt.cm.viridis
        if len(time_indices) > 1:
            t_min, t_max = min(time_indices), max(time_indices)
            poly_colors = [
                cmap((t - t_min) / max(t_max - t_min, 1)) for t in time_indices
            ]
        else:
            poly_colors = [cmap(0.5)] * len(poly_list)

        return _plot_polygons(
            poly_list,
            None,
            sample_points_list,
            weighted_lon_list,
            weighted_lat_list,
            plot_samples=plot_samples,
            ax=ax,
            poly_colors=poly_colors,
        )

    def plot_track_rwps(self, path, ax=None):

        rwp_list = []

        if ax is None:
            ax = plt.subplot(
                projection=ccrs.PlateCarree(central_longitude=180)
            )

        for node in path:
            time_step_data = self._time_step_data[node[0]]

            for path, rwp_info in time_step_data.rwp_info.items():
                if rwp_info["rwp_id"] == node[1]:
                    rwp_list.append(([path], time_step_data.pruned_graph))

        for path, pruned_graph in rwp_list:
            _plot_rwp_paths(paths=path, rwp_graph=pruned_graph, ax=ax)

        return ax
