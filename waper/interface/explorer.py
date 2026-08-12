import cartopy.crs as ccrs
import geopandas as gpd
import geoviews as gv
import holoviews as hv

# Imported for their side effect: they register the `.hvplot` accessor on
# pandas DataFrames and xarray DataArrays respectively.
import hvplot.pandas
import hvplot.xarray  # noqa: F401
import numpy as np
import panel as pn
import param
from shapely import wkb

from .colormaps import bokeh_palette, joy_nl8

pn.extension(throttled=True)

NODE_CMAP = {"max": "#b2182b", "min": "#2166ac"}


def _stereo_proj4(hemisphere):
    """proj4 string for the polar-stereographic CRS WAPER builds RWP polygons in."""
    lat0 = -90 if hemisphere == "south" else 90
    return f"+proj=stere +lat_0={lat0} +lon_0=0"


def default_projection(hemisphere):
    """Default *display* projection: polar stereographic for the hemisphere.

    Matches the matplotlib ``_plot_polygons`` view and is seam-free, so
    dateline-crossing packets stay contiguous (Web Mercator tore them apart).
    Override per-call for other workflows — e.g. western disturbances over South
    Asia read better in ``ccrs.Orthographic(central_longitude=75, central_latitude=25)``.
    """
    lat0 = -90 if hemisphere == "south" else 90
    return ccrs.Stereographic(central_latitude=lat0, central_longitude=0)


def _hemisphere(cat):
    return (getattr(cat, "meta", None) or {}).get("hemisphere", "north")


def _domain_extent(lon_lo, lon_hi, lat_lo, lat_hi, projection, n=60):
    """Projected (xlim, ylim) of a lon/lat box, used to clip the map.

    Without this, ``coastline=True`` draws *global* coasts, and the
    off-hemisphere ones project to ~infinity in a polar projection, blowing up
    the axes (the matplotlib viz used ``ax.set_extent`` for this).
    """
    edge = np.linspace(0, 1, n)
    lons = np.concatenate([
        lon_lo + (lon_hi - lon_lo) * edge, np.full(n, lon_hi),
        lon_hi - (lon_hi - lon_lo) * edge, np.full(n, lon_lo)])
    lats = np.concatenate([
        np.full(n, lat_lo), lat_lo + (lat_hi - lat_lo) * edge,
        np.full(n, lat_hi), lat_hi - (lat_hi - lat_lo) * edge])
    xy = projection.transform_points(ccrs.PlateCarree(), lons, lats)
    x, y = xy[:, 0], xy[:, 1]
    ok = np.isfinite(x) & np.isfinite(y)
    return (float(x[ok].min()), float(x[ok].max())), (float(y[ok].min()), float(y[ok].max()))


def nodes_layer(cat, time, projection=None):
    """Scatter the RWP extremum nodes of one timestep, coloured by node type.

    Maxima are red and minima blue; hovering reports the node's scalar value.
    The node lon/lat are plain geographic coordinates, so the **data** CRS is
    always ``PlateCarree`` — ``projection`` only sets how they are *displayed*.
    An empty timestep yields an empty ``gv.Points`` rather than raising, so the
    layer can stay in an overlay across all times.

    Args:
        cat: Catalogue to read from; its ``meta["hemisphere"]`` picks the default
            projection.
        time: Timestep to filter to.
        projection: Cartopy display projection. ``None`` falls back to
            :func:`default_projection` for the catalogue's hemisphere.

    Returns:
        A holoviews/geoviews element ready to compose into an overlay.
    """
    projection = projection or default_projection(_hemisphere(cat))
    df = cat.filter(time=time).nodes()
    if df.empty:
        return gv.Points([], kdims=["lon", "lat"], crs=ccrs.PlateCarree()).opts(
            projection=projection)
    return df.hvplot.points(x="lon", y="lat", c="node_type", cmap=NODE_CMAP,
                            geo=True, projection=projection,
                            hover_cols=["scalar", "node_type"],
                            responsive=True, height=500)


def polygons_layer(cat, time, projection=None):
    """Draw the RWP footprint polygons of one timestep, filled by ``rwp_id``.

    Unlike the node and edge layers, polygons are stored in WAPER's
    polar-stereographic CRS in metres, not lon/lat; that CRS is declared on the
    GeoDataFrame so geoviews reprojects them into the display projection
    correctly. An empty timestep yields an empty ``gv.Polygons``.

    Args:
        cat: Catalogue to read from.
        time: Timestep to filter to.
        projection: Cartopy display projection. ``None`` falls back to
            :func:`default_projection` for the catalogue's hemisphere.

    Returns:
        A holoviews/geoviews element ready to compose into an overlay.
    """
    hemi = _hemisphere(cat)
    projection = projection or default_projection(hemi)
    df = cat.filter(time=time).rwps()
    if df.empty:
        return gv.Polygons([], crs=ccrs.PlateCarree()).opts(projection=projection)
    # Polygons are stored in WAPER's polar-stereographic CRS (metres); declare it
    # so geoviews/cartopy reproject correctly into the display projection.
    gdf = gpd.GeoDataFrame(df.assign(geometry=df["geometry_wkb"].apply(wkb.loads)),
                           geometry="geometry", crs=_stereo_proj4(hemi))
    return gdf.hvplot.polygons(geo=True, projection=projection, fill_alpha=0.3,
                               fill_color="rwp_id", line_color="black",
                               colorbar=False, responsive=True, height=500)


def edges_layer(cat, time, projection=None):
    """Draw the within-timestep RWP graph edges as black great-circle segments.

    Each edge is looked up as a ``(rwp_id, node_id)`` pair in the node table;
    edges whose endpoints are missing from that table are silently skipped, so a
    partially written catalogue still renders. Endpoints are lon/lat, so the
    **data** CRS is always ``PlateCarree`` whatever the display projection is.

    Args:
        cat: Catalogue to read from.
        time: Timestep to filter to.
        projection: Cartopy display projection. ``None`` falls back to
            :func:`default_projection` for the catalogue's hemisphere.

    Returns:
        A ``gv.Path`` ready to compose into an overlay.
    """
    projection = projection or default_projection(_hemisphere(cat))
    nd = cat.filter(time=time).nodes()
    if nd.empty:
        return gv.Path([], crs=ccrs.PlateCarree()).opts(projection=projection)
    nd = nd.set_index(["rwp_id", "node_id"])
    ed = cat.filter(time=time).edges()
    segs = []
    for r in ed.itertuples():
        try:
            a = nd.loc[(r.rwp_id, r.src_node_id)]
            b = nd.loc[(r.rwp_id, r.dst_node_id)]
        except KeyError:
            continue
        segs.append([(a.lon, a.lat), (b.lon, b.lat)])
    return gv.Path(segs, crs=ccrs.PlateCarree()).opts(projection=projection,
                                                      color="black")


def field_layer(field_da, time_index, projection=None):
    """Rasterise the background scalar field for one timestep as a quadmesh.

    The colour limits are symmetric about zero at the 99th percentile of
    ``|field|`` *for that timestep*, so the scale moves between frames — read
    colours qualitatively, not as absolute values. Rendering goes through
    datashader (``rasterize=True``) because a full ERA5 grid is too dense to send
    to the browser as vectors. Coastlines are drawn by this layer.

    The field carries lon/lat coordinates, so its **data** CRS is ``PlateCarree``
    regardless of the display projection.

    Args:
        field_da: DataArray with ``time``, ``latitude`` and ``longitude`` dims.
        time_index: Positional index along ``time`` (``isel``, not ``sel``).
        projection: Cartopy display projection. ``None`` falls back to
            :func:`default_projection` for the **northern** hemisphere — this
            function has no catalogue to read the hemisphere from, so pass it
            explicitly for southern-hemisphere work.

    Returns:
        A holoviews element ready to compose into an overlay.
    """
    projection = projection or default_projection("north")
    da = field_da.isel(time=time_index)
    vmax = float(abs(da).quantile(0.99))
    return da.hvplot.quadmesh(x="longitude", y="latitude", geo=True, project=True,
                              projection=projection,
                              cmap=bokeh_palette(joy_nl8), clim=(-vmax, vmax),
                              rasterize=True, clabel="v (m s⁻¹)",
                              coastline=True, responsive=True, height=500)


class RWPExplorer(pn.viewable.Viewer):
    """Panel app for stepping through a catalogue of identified and tracked RWPs.

    The layout is a map with a time player, a layer toggle, a sortable table of
    track durations whose selection highlights that track on the map in yellow,
    and — when ``field_da`` is given — a Hovmöller panel of the
    latitude-averaged field.

    Rendering is **polar stereographic by default**, matching the matplotlib
    polygon plots and keeping dateline-crossing packets contiguous (Web Mercator
    tears them). The display projection is overridable per instance via
    ``projection``: the motivating case is western disturbances over South Asia,
    which read far better in
    ``ccrs.Orthographic(central_longitude=75, central_latitude=25)``. The map is
    clipped to the analysis hemisphere (lat 15–90, or -90 to -15) so global
    coastlines do not blow up a polar-projection axes.

    Args:
        cat: Catalogue of nodes, edges, RWP polygons and tracks.
        n_times: Number of timesteps; sets the time slider's upper bound.
        field_da: Optional background field. When present, the ``field`` layer is
            offered and enabled by default, and the Hovmöller panel appears.
        projection: Cartopy display projection. ``None`` falls back to
            :func:`default_projection` for the catalogue's hemisphere.
        **params: Forwarded to ``param.Parameterized`` (e.g. ``time``, ``layers``).

    Display it with ``.servable()``, or just let a notebook cell render it.
    """

    time = param.Integer(default=0, bounds=(0, 0))
    layers = param.ListSelector(default=["polygons", "nodes"],
                                objects=["nodes", "edges", "polygons"])

    def __init__(self, cat, n_times=1, field_da=None, projection=None, **params):
        self.cat = cat
        self.n_times = n_times
        self.field_da = field_da
        # Display projection: caller-supplied, else the hemisphere's polar stereo.
        self.projection = projection or default_projection(_hemisphere(cat))

        # Determine available layers
        layer_objects = ["nodes", "edges", "polygons"]
        if self.field_da is not None:
            layer_objects = ["field", *layer_objects]

        super().__init__(**params)
        self.param.time.bounds = (0, max(0, n_times - 1))
        self.param.layers.objects = layer_objects

        # If field is available, enable it by default
        if self.field_da is not None:
            self.layers = ["field", "polygons", "nodes"]

        # Get track paths for highlighting
        self._paths, self._coords = self.cat._track_paths()

        # Tabulator for tracks
        track_df = self.cat.track_durations()
        self._track_table = pn.widgets.Tabulator(
            track_df,
            label="Track Durations",
            selectable=True,
            show_index=False,
            configuration={"headerSort": True},
            sizing_mode="stretch_width"
        )

        # Set up dynamic maps
        nodes = hv.DynamicMap(pn.bind(self._nodes, self.param.time, self.param.layers))
        edges = hv.DynamicMap(pn.bind(self._edges, self.param.time, self.param.layers))
        polys = hv.DynamicMap(pn.bind(self._polys, self.param.time, self.param.layers))
        field = hv.DynamicMap(pn.bind(self._field, self.param.time, self.param.layers))
        highlight = hv.DynamicMap(pn.bind(self._highlight, self._track_table.param.selection))

        # Clip the map to the analysis hemisphere so global coastlines don't
        # blow up the polar-projection axes. Default: full longitude, lat 15–90
        # (north) / -90 to -15 (south) — matching the matplotlib viz extent.
        if _hemisphere(cat) == "south":
            lat_lo, lat_hi = -90.0, -15.0
        else:
            lat_lo, lat_hi = 15.0, 90.0
        xlim, ylim = _domain_extent(-180.0, 180.0, lat_lo, lat_hi, self.projection)

        # Combine overlays
        overlay = (field * polys * edges * nodes * highlight).opts(xlim=xlim, ylim=ylim)
        self._map = pn.pane.HoloViews(
            overlay,
            sizing_mode="stretch_width",
            theme="light_minimal"
        )

        self._slider = pn.widgets.Player.from_param(self.param.time, name="Time Step", sizing_mode="stretch_width")
        self._toggles = pn.widgets.CheckButtonGroup.from_param(self.param.layers)

        # Hovmöller side panel (longitude vs time — a plain image, not a map)
        if self.field_da is not None:
            vmax = float(abs(self.field_da).quantile(0.99))
            hov = self.field_da.mean("latitude").hvplot.image(
                x="longitude",
                y="time",
                geo=False,
                cmap=bokeh_palette(joy_nl8),
                clim=(-vmax, vmax),
                clabel="v (m s⁻¹)",
                title="Hovmöller Diagram",
                height=300,
                responsive=True
            )
            self._hovmoeller = pn.pane.HoloViews(hov, sizing_mode="stretch_width")
        else:
            self._hovmoeller = pn.Spacer()

    def _nodes(self, time, layers):
        if "nodes" in layers:
            return nodes_layer(self.cat, time, projection=self.projection)
        return gv.Points([], kdims=["lon", "lat"], crs=ccrs.PlateCarree()).opts(
            projection=self.projection)

    def _edges(self, time, layers):
        if "edges" in layers:
            return edges_layer(self.cat, time, projection=self.projection)
        return gv.Path([], crs=ccrs.PlateCarree()).opts(projection=self.projection)

    def _polys(self, time, layers):
        if "polygons" in layers:
            return polygons_layer(self.cat, time, projection=self.projection)
        return gv.Polygons([], crs=ccrs.PlateCarree()).opts(projection=self.projection)

    def _field(self, time, layers):
        if "field" in layers and self.field_da is not None:
            return field_layer(self.field_da, time, projection=self.projection)
        return gv.Points([], kdims=["lon", "lat"], crs=ccrs.PlateCarree()).opts(
            projection=self.projection)

    def _highlight(self, selection):
        empty = gv.Path([], crs=ccrs.PlateCarree()).opts(projection=self.projection)
        if not selection or not self._paths:
            return empty
        idx = selection[0]
        if idx >= len(self._paths):
            return empty
        path_keys = self._paths[idx]
        pts = []
        for k in path_keys:
            if k in self._coords:
                lon, lat, _ = self._coords[k]
                pts.append((lon, lat))
        if len(pts) < 2:
            return empty
        return gv.Path([pts], crs=ccrs.PlateCarree()).opts(
            projection=self.projection, color="yellow", line_width=4, alpha=0.9)

    def __panel__(self):
        sidebar = pn.Column(
            pn.pane.Markdown("### Layer Controls"),
            self._toggles,
            self._track_table,
            self._hovmoeller,
            sizing_mode="stretch_height",
            width=350
        )
        main_content = pn.Column(
            self._slider,
            self._map,
            sizing_mode="stretch_width"
        )
        return pn.Row(sidebar, main_content, sizing_mode="stretch_both")
