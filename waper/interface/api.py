import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import yaml
from networkx import Graph
from numpy import ndarray
from pyvista import PolyData
from tqdm import tqdm
from xarray import DataArray

from ..identification import max_min, rwp_graph, topology, utils
from ..tracking import rwp_polygon, tracking_graph
from .visualization import (
    _plot_clusters,
    _plot_graph,
    _plot_polygons,
    _plot_raster,
    _plot_rwp_paths,
)

logger = logging.getLogger(__name__)


@dataclass(eq=False, frozen=True)
class WaperConfig:
    """Frozen bundle of every tunable that identification and tracking read.

    The instance is immutable (``frozen=True``): build a new one to change a
    setting rather than mutating an existing :class:`Waper`'s config.

    Units are explicit rather than normalised, because the thresholds live on
    physically different scales:

    * ``track_pruning_threshold`` is the maximum centroid displacement of a
      tracking edge, **in kilometres** (default 8000). The historical default of
      ``0.3`` was a bug: read as km it prunes every edge and leaves an empty
      tracking graph.
    * ``penalty_length_scale_km`` (clustering distance penalty) and
      ``energy_radius_km`` (radius of the per-node energy disk rasterised into
      the energy raster) are also kilometres.
    * ``lat_gate`` is **degrees** of latitude.
    * ``cluster_max_eps_km`` is kilometres; ``cluster_min_samples`` and
      ``cluster_xi`` are OPTICS parameters passed through unchanged.

    ``track_weight_threshold`` is the minimum envelope-overlap weight in (0, 1]
    a tracking edge must carry. It defaults to ``None``, which **disables** the
    overlap gate — the weight has not been calibrated, so distance alone decides.

    The ``vtk_*_label`` fields name the point-data arrays WAPER attaches to the
    VTK mesh, and are not read from the input dataset.
    """

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

    # maximum centroid displacement of a tracking edge, in km
    track_pruning_threshold: float = 8000.0
    # minimum envelope-level overlap weight of a tracking edge, in (0, 1].
    # None disables the gate, which is the default: it has not been calibrated.
    track_weight_threshold: float | None = None

    cluster_max_eps_km: float = 3000.0
    cluster_min_samples: int = 2
    cluster_xi: float = 0.15
    min_longitude_separation: float = 6.0
    max_aspect_ratio: float = 1.5
    hull_method: str = "per_node"  # "per_node" | "convex" | "concave"
    hemisphere: str = "north"  # "north" | "south"
    penalty_length_scale_km: float = 2000.0
    energy_radius_km: float = 500.0
    lat_gate: float = 15.0

    vtk_latitude_label: str = "Latitude"
    vtk_longitude_label: str = "Longitude"
    vtk_region_label: str = "RegionId"

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialise this configuration to YAML.

        Args:
            path: If given, the YAML is also written to this path.

        Returns:
            The YAML document as a string.
        """
        text = yaml.safe_dump(dataclasses.asdict(self), sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_yaml(cls, source: str | Path) -> "WaperConfig":
        """Build a configuration from a YAML file or YAML string.

        Args:
            source: A path to a ``.yaml`` file, or the YAML document itself.

        Returns:
            The deserialised configuration.

        Raises:
            TypeError: If the document contains a key that is not a field of
                this class.
        """
        candidate = Path(source)
        try:
            # A YAML document is not a filename. `is_file()` is the cheapest
            # way to tell the two accepted inputs apart; it raises rather than
            # returning False when the string is too long to be a path.
            is_file = candidate.is_file()
        except OSError:
            is_file = False
        text = candidate.read_text() if is_file else str(source)
        return cls(**yaml.safe_load(text))


@dataclass(eq=False)
class WaperSingleTimestepData:
    """All intermediate state produced for one timestep of the input field.

    ``__init__`` only fills ``input_data``, ``vtk_data`` (the geovista mesh built
    from the field) and an empty ``rwp_info``. Every other attribute is populated
    later by the identification pass, in order: the clustered extrema
    (``all_maxima`` / ``all_minima`` and their ``*_cluster_info`` dicts), the
    ``association_graph`` and its ``pruned_graph``, the ranked
    ``identified_rwp_paths``, then ``rwp_info`` keyed by ``tuple(path)`` holding
    each packet's polygon, integer ``rwp_id``, sample points and energy-weighted
    centroid, and finally the rasters.

    ``energy_raster`` is ``None`` until ``_identify_rwps`` sets it; ``raster_data``
    may be ``None`` when no packet was found, in which case ``raster_features``
    is just ``{0}``.
    """

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
    raster_features: set
    energy_raster: ndarray | None = None  # set by _identify_rwps

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
        connectivity, maxima_points, config.scalar_name, sign=1,
        max_eps_km=config.cluster_max_eps_km, min_samples=config.cluster_min_samples,
        xi=config.cluster_xi, penalty_length_scale_km=config.penalty_length_scale_km,
    )

    (
        _cluster_max_arr,
        _cluster_max_point,
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
        connectivity, minima_points, config.scalar_name, sign=-1,
        max_eps_km=config.cluster_max_eps_km, min_samples=config.cluster_min_samples,
        xi=config.cluster_xi, penalty_length_scale_km=config.penalty_length_scale_km,
    )

    (
        _cluster_min_arr,
        _cluster_min_point,
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
        time_step_data.pruned_graph, config.max_edge_weight, lat_gate=config.lat_gate
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

    energy_nodes = []
    for path in time_step_data.identified_rwp_paths:
        for n in path:
            lon, lat = time_step_data.pruned_graph.nodes[n]["coords"]
            scalar = time_step_data.pruned_graph.nodes[n]["scalar"]
            energy_nodes.append((lon, lat, scalar))
    energy_cells = rwp_polygon.energy_disks(
        energy_nodes, hemisphere=config.hemisphere,
        radius_m=config.energy_radius_km * 1000.0,
    )
    time_step_data.energy_raster = rwp_polygon.rasterize_energy(
        energy_cells, hemisphere=config.hemisphere
    )

    if time_step_data.raster_data is None:
        time_step_data.raster_features = {0}
    else:
        features = set(np.unique(time_step_data.raster_data))
        features.add(0)
        time_step_data.raster_features = features

    return time_step_data


def _track_rwps(time_step_data, num_time_steps):

    return tracking_graph.build_tracking_graph(time_step_data, num_time_steps)


class Waper:
    """Entry point: identify Rossby wave packets in a field and track them in time.

    Construction only records the configuration; no computation happens until you
    call the two stages, **in this order**:

    1. :meth:`identify_rwps` — runs the per-timestep identification over the whole
       time axis and fills ``_time_step_data``.
    2. :meth:`track_rwps` — links those packets across time into
       ``_tracking_graph`` and ``_pruned_tracking_graph``.

    Every ``plot_*`` method reads ``_time_step_data``, so :meth:`identify_rwps`
    must have run first; :meth:`plot_tracks`, :meth:`plot_track_polygons` and
    :meth:`plot_track_rwps` additionally need :meth:`track_rwps`.

    Args:
        data_array: Dataset containing the scalar field to analyse.
        scalar_name: Name of the field variable inside ``data_array``.
        latitude_label: Name of the latitude coordinate.
        longitude_label: Name of the longitude coordinate.
        time_label: Name of the time coordinate; its length sets how many
            timesteps :meth:`identify_rwps` processes.
        clip_value: Field magnitude below which the mesh is clipped away before
            extrema are extracted.
        extrema_threshold: Minimum magnitude for a point to count as an extremum.
        max_latitude: Poleward latitude bound, or ``None`` for no bound.
        min_latitude: Equatorward latitude bound, or ``None`` for no bound.
        node_pruning_threshold: Minimum node scalar kept in the association graph.
        edge_pruning_threshold: Minimum association-graph edge weight kept.
        track_pruning_threshold: Maximum tracking-edge centroid displacement, in km.
        track_weight_threshold: Minimum tracking-edge overlap weight, or ``None``
            to disable that gate (the default).
        max_edge_weight: Upper bound on association-graph edge weight, also used
            when ranking paths.
        debug: Turn on debug-level logging.
        penalty_length_scale_km: Length scale, in km, of the clustering distance
            penalty.
        lat_gate: Latitude gate, in degrees, applied when ranking RWP paths.

    See :class:`WaperConfig` for the settings that have no constructor argument
    and keep their defaults.
    """

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
        track_pruning_threshold=8000.0,
        track_weight_threshold=None,
        max_edge_weight=1,
        debug=False,
        penalty_length_scale_km=2000.0,
        lat_gate=15.0,
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
            track_weight_threshold=track_weight_threshold,
            max_edge_weight=max_edge_weight,
            debug=debug,
            penalty_length_scale_km=penalty_length_scale_km,
            lat_gate=lat_gate,
        )

        self._setup(data_array, self._config)

    def _setup(self, data_array, config: WaperConfig) -> None:
        self._config = config
        self.data_array = data_array
        self._num_time_steps = len(data_array[config.time_label])
        self._time_step_data: list = []

        if config.debug:
            logging.basicConfig(level=logging.DEBUG)

    @classmethod
    def from_config(cls, data_array, config: WaperConfig) -> "Waper":
        """Construct from a :class:`WaperConfig`, reaching every field.

        ``__init__`` exposes 18 of the config's 25 fields as keyword arguments;
        ``hemisphere``, ``hull_method``, ``energy_radius_km`` and the clustering
        parameters are not among them. This is the way to set those — and the way
        to run from a config file:

        >>> waper = Waper.from_config(ds, WaperConfig.from_yaml("run.yaml"))

        Args:
            data_array: The input dataset, indexed by the config's ``time_label``.
            config: A fully specified configuration.

        Returns:
            An unrun ``Waper``. Call ``identify_rwps()`` next.
        """
        obj = cls.__new__(cls)
        obj._setup(data_array, config)
        return obj

    def identify_rwps(self):
        """Identify wave packets at every timestep of the input field.

        Appends one :class:`WaperSingleTimestepData` per timestep to
        ``_time_step_data``, in time order, so a timestep's index into that list
        is its index along the time axis. Progress is reported with a tqdm bar.

        Calling this twice appends a second pass rather than replacing the first.
        A timestep in which no packet survives pruning logs a warning and still
        contributes an (empty) entry.
        """
        for i in tqdm(range(self._num_time_steps)):
            self._time_step_data.append(
                _identify_rwps(
                    self.data_array[self._config.scalar_name][i], self._config
                )
            )

    def track_rwps(self, num_time_steps=None):
        """Link identified wave packets across time into a tracking graph.

        Stores the full graph on ``_tracking_graph`` and a pruned copy on
        ``_pruned_tracking_graph``. Pruning drops edges whose centroid
        displacement exceeds the configured ``track_pruning_threshold``
        kilometres and, if ``track_weight_threshold`` is set, edges whose
        energy-overlap weight falls below it.

        Requires :meth:`identify_rwps` to have run first.

        Args:
            num_time_steps: Number of timesteps to link. ``None`` uses all of them.
        """
        self._tracking_graph = _track_rwps(self._time_step_data, num_time_steps)
        self._pruned_tracking_graph = tracking_graph.prune_tracking_graph(
            self._tracking_graph,
            self._config.track_pruning_threshold,
            weight_threshold=self._config.track_weight_threshold,
        )

    def plot_clusters(self, time_index, projection=None):
        """Plot the clustered maxima and minima for one timestep.

        Draws two stacked PlateCarree panels — maxima above, minima below — with
        points coloured by cluster id over the clipped scalar field. Creates its
        own figure axes, so it takes no ``ax``.

        Args:
            time_index: Index into ``_time_step_data`` (i.e. along the time axis).
            projection: Cartopy projection to draw both panels in. ``None`` keeps
                the whole-globe ``PlateCarree(central_longitude=180)`` default,
                which is hemisphere-agnostic.

        Returns:
            The matplotlib ``Axes`` of the lower (minima) panel.
        """
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
            projection=projection,
        )

    def plot_association_graph(self, time_index, ax=None, projection=None):
        """Plot the unpruned association graph over the scalar field.

        Shows every extremum node and every candidate max–min link found at this
        timestep, before node and edge pruning — useful for judging whether the
        pruning thresholds are throwing away real structure.

        Args:
            time_index: Index into ``_time_step_data``.
            ax: Axes to draw on. Must already carry a cartopy projection. ``None``
                creates one with ``PlateCarree(central_longitude=180)``.
            projection: Cartopy projection for the axes this method creates.
                ``None`` keeps the whole-globe ``PlateCarree(central_longitude=180)``
                default. An ``ax`` you pass in takes precedence over both: its own
                projection is used and ``projection`` is ignored.

        Returns:
            The matplotlib ``Axes`` drawn on.
        """
        time_step_data = self._time_step_data[time_index]

        return _plot_graph(
            time_step_data.association_graph, time_step_data.input_data, ax=ax,
            projection=projection,
        )

    def plot_pruned_graph(self, time_index, ax=None, projection=None):
        """Plot the association graph after node and edge pruning.

        Same rendering as :meth:`plot_association_graph`, but of the graph the
        RWP paths are actually extracted from; comparing the two shows what the
        thresholds removed.

        Args:
            time_index: Index into ``_time_step_data``.
            ax: Axes to draw on. Must already carry a cartopy projection. ``None``
                creates one with ``PlateCarree(central_longitude=180)``.
            projection: Cartopy projection for the axes this method creates.
                ``None`` keeps the whole-globe ``PlateCarree(central_longitude=180)``
                default. An ``ax`` you pass in takes precedence over both.

        Returns:
            The matplotlib ``Axes`` drawn on.
        """
        time_step_data = self._time_step_data[time_index]

        return _plot_graph(
            time_step_data.pruned_graph, time_step_data.input_data, ax=ax,
            projection=projection,
        )

    def plot_rwp_graphs(self, time_index, ax=None, plot_scalar_data=True, projection=None):
        """Plot the identified RWP paths as node chains through the pruned graph.

        Only the ranked paths are drawn, not the whole pruned graph, so this is
        the view of what was accepted as a wave packet at this timestep.

        Args:
            time_index: Index into ``_time_step_data``.
            ax: Axes to draw on. Must already carry a cartopy projection. ``None``
                creates one with ``PlateCarree(central_longitude=180)``.
            plot_scalar_data: Draw the scalar field underneath the paths. Set
                ``False`` for a paths-only figure.
            projection: Cartopy projection for the axes this method creates.
                ``None`` keeps the whole-globe ``PlateCarree(central_longitude=180)``
                default. An ``ax`` you pass in takes precedence over both.

        Returns:
            The matplotlib ``Axes`` drawn on.
        """
        time_step_data = self._time_step_data[time_index]

        field = None
        if plot_scalar_data:
            field = time_step_data.input_data

        return _plot_rwp_paths(
            time_step_data.pruned_graph,
            time_step_data.identified_rwp_paths,
            field,
            ax=ax,
            projection=projection,
        )

    def plot_rwp_polygons(self, time_index, plot_samples=False, ax=None, projection=None):
        """Plot every RWP footprint polygon for one timestep.

        Each packet's hull is drawn over the scalar field, together with its
        energy-weighted centroid. Polygons live in WAPER's polar-stereographic
        CRS, so the default axes are polar stereographic rather than PlateCarree.

        Args:
            time_index: Index into ``_time_step_data``.
            plot_samples: Also scatter the sample points the hull was built from.
            ax: Axes to draw on. Must already carry a cartopy projection. ``None``
                creates a polar-stereographic axes for the configured hemisphere.
            projection: Cartopy projection to *display* in. ``None`` means the
                hemisphere default (:func:`~waper.interface.projections.default_projection`);
                an ``ax`` you pass in takes precedence over both. This changes only
                the display — the polygons are still drawn in the CRS they were
                built in, so overriding it reprojects them rather than moving them.

        Returns:
            The matplotlib ``Axes`` drawn on.
        """
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
            projection=projection,
            hemisphere=self._config.hemisphere,
        )

    def plot_raster(self, time_index, ax=None, projection=None):
        """Plot the rasterised RWP label field for one timestep.

        Shows ``raster_data`` — the polygons burned onto WAPER's polar-stereographic
        grid, each cell holding an ``rwp_id`` — with zero (no packet) masked out.
        This is the array tracking overlaps between timesteps, so it is the view
        to check when a link looks wrong.

        Args:
            time_index: Index into ``_time_step_data``.
            ax: Axes to draw on. Must already carry a cartopy projection. ``None``
                creates a polar-stereographic axes for the configured hemisphere.
            projection: Cartopy projection to *display* in. ``None`` means the
                hemisphere default (:func:`~waper.interface.projections.default_projection`);
                an ``ax`` you pass in takes precedence over both. The raster is
                reprojected from the grid CRS it was burned in, not moved.

        Returns:
            The matplotlib ``Axes`` drawn on.
        """
        time_step_data = self._time_step_data[time_index]

        return _plot_raster(
            time_step_data.raster_data,
            ax=ax,
            projection=projection,
            hemisphere=self._config.hemisphere,
        )

    def plot_tracks(self, threshold=None, weight_threshold=None, projection=None):
        """Plot every track path, re-pruning the tracking graph on the fly.

        Prunes ``_tracking_graph`` with the thresholds given here rather than
        reusing ``_pruned_tracking_graph``, so you can sweep thresholds without
        re-running :meth:`track_rwps`.

        Note the ``None`` semantics differ from the underlying
        ``prune_tracking_graph``: here ``None`` means "fall back to the
        **configured** ``track_pruning_threshold`` / ``track_weight_threshold``",
        whereas ``prune_tracking_graph(g, None)`` means "keep every edge". To
        actually keep every edge, call that function directly.

        Requires :meth:`track_rwps` to have run. Creates its own axes.

        Args:
            threshold: Maximum centroid displacement in km. ``None`` uses the
                configured value.
            weight_threshold: Minimum overlap weight in (0, 1]. ``None`` uses the
                configured value, which is itself ``None`` (gate disabled) by default.
            projection: Cartopy projection for the axes this method creates.
                ``None`` keeps the whole-globe ``PlateCarree(central_longitude=180)``
                default, which is hemisphere-agnostic.

        Returns:
            The matplotlib ``Axes`` drawn on.
        """
        if threshold is None:
            threshold = self._config.track_pruning_threshold
        if weight_threshold is None:
            weight_threshold = self._config.track_weight_threshold
        pruned = tracking_graph.prune_tracking_graph(
            self._tracking_graph,
            threshold=threshold,
            weight_threshold=weight_threshold,
        )
        paths = tracking_graph.get_track_paths(pruned)
        return _plot_rwp_paths(
            pruned,
            paths,
            None,
            projection=projection,
        )

    def plot_track_polygons(self, path, plot_samples=False, ax=None, projection=None):
        """Plot one track's RWP footprints, coloured by time.

        Overlays the polygon of every packet along the track on a single map,
        shaded from dark to light through ``viridis`` in track-time order, so the
        packet's propagation is visible in one figure. No scalar field is drawn.

        Args:
            path: Sequence of tracking-graph nodes, each a ``(time_index, rwp_id)``
                pair — e.g. one element of ``get_track_paths(pruned_graph)``.
            plot_samples: Also scatter the sample points each hull was built from.
            ax: Axes to draw on. Must already carry a cartopy projection. ``None``
                creates a polar-stereographic axes for the configured hemisphere.
            projection: Cartopy projection to *display* in. ``None`` means the
                hemisphere default (:func:`~waper.interface.projections.default_projection`);
                an ``ax`` you pass in takes precedence over both. The polygons keep
                being drawn in the CRS they were built in either way.

        Returns:
            The matplotlib ``Axes`` drawn on.
        """
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
            projection=projection,
            hemisphere=self._config.hemisphere,
        )

    def plot_track_rwps(self, path, ax=None, projection=None):
        """Plot one track's RWP node chains on a single map.

        The graph-level counterpart of :meth:`plot_track_polygons`: instead of the
        footprint hulls it draws the max/min node chain of each packet along the
        track, all overlaid on one axes and without the scalar field.

        Args:
            path: Sequence of tracking-graph nodes, each a ``(time_index, rwp_id)``
                pair.
            ax: Axes to draw on. Must already carry a cartopy projection. ``None``
                creates one with ``PlateCarree(central_longitude=180)``.
            projection: Cartopy projection for the axes this method creates.
                ``None`` keeps the whole-globe ``PlateCarree(central_longitude=180)``
                default. An ``ax`` you pass in takes precedence over both.

        Returns:
            The matplotlib ``Axes`` drawn on.
        """
        rwp_list = []

        if ax is None:
            ax = plt.subplot(
                projection=projection or ccrs.PlateCarree(central_longitude=180)
            )

        for node in path:
            time_step_data = self._time_step_data[node[0]]

            for path, rwp_info in time_step_data.rwp_info.items():
                if rwp_info["rwp_id"] == node[1]:
                    rwp_list.append(([path], time_step_data.pruned_graph))

        for path, pruned_graph in rwp_list:
            _plot_rwp_paths(paths=path, rwp_graph=pruned_graph, ax=ax)

        return ax
