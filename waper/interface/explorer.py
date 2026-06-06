import geopandas as gpd
import holoviews as hv
import hvplot.pandas  # noqa
import hvplot.xarray  # noqa
import numpy as np
import pandas as pd
from shapely import wkb
import panel as pn
import param
from .colormaps import joy_nl8, bokeh_palette

pn.extension(throttled=True)

NODE_CMAP = {"max": "#b2182b", "min": "#2166ac"}

def nodes_layer(cat, time):
    df = cat.filter(time=time).nodes()
    if df.empty:
        return hv.Points([], kdims=["lon", "lat"])
    return df.hvplot.points(x="lon", y="lat", c="node_type", cmap=NODE_CMAP,
                            geo=True, hover_cols=["scalar", "node_type"],
                            responsive=True, height=500)

def polygons_layer(cat, time):
    df = cat.filter(time=time).rwps()
    if df.empty:
        return hv.Polygons([])
    gdf = gpd.GeoDataFrame(df.assign(geometry=df["geometry_wkb"].apply(wkb.loads)),
                           geometry="geometry", crs="EPSG:4326")
    return gdf.hvplot.polygons(geo=True, alpha=0.25, fill_color="rwp_id", line_color="black",
                               colorbar=False, responsive=True, height=500)

def edges_layer(cat, time):
    nd = cat.filter(time=time).nodes()
    if nd.empty:
        return hv.Path([])
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
    return hv.Path(segs).opts(color="black")

def field_layer(field_da, time_index):
    da = field_da.isel(time=time_index)
    vmax = float(abs(da).quantile(0.99))
    return da.hvplot.quadmesh(x="longitude", y="latitude", geo=True, project=True,
                              cmap=bokeh_palette(joy_nl8), clim=(-vmax, vmax),
                              rasterize=True, clabel="v (m s⁻¹)",
                              coastline=True, responsive=True, height=500)

class RWPExplorer(pn.viewable.Viewer):
    time = param.Integer(default=0, bounds=(0, 0))
    layers = param.ListSelector(default=["polygons", "nodes"],
                                objects=["nodes", "edges", "polygons"])

    def __init__(self, cat, n_times=1, field_da=None, **params):
        self.cat = cat
        self.n_times = n_times
        self.field_da = field_da
        
        # Determine available layers
        layer_objects = ["nodes", "edges", "polygons"]
        if self.field_da is not None:
            layer_objects = ["field"] + layer_objects
            
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
        
        # Combine overlays
        self._map = pn.pane.HoloViews(
            field * polys * edges * nodes * highlight,
            sizing_mode="stretch_width",
            theme="light_minimal"
        )
        
        self._slider = pn.widgets.Player.from_param(self.param.time, name="Time Step", sizing_mode="stretch_width")
        self._toggles = pn.widgets.CheckButtonGroup.from_param(self.param.layers)
        
        # Hovmöller side panel
        if self.field_da is not None:
            vmax = float(abs(self.field_da).quantile(0.99))
            hov = self.field_da.mean("latitude").hvplot.image(
                x="longitude",
                y="time",
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
        return nodes_layer(self.cat, time) if "nodes" in layers else hv.Points([], kdims=["lon", "lat"])
        
    def _edges(self, time, layers):
        return edges_layer(self.cat, time) if "edges" in layers else hv.Path([])
        
    def _polys(self, time, layers):
        return polygons_layer(self.cat, time) if "polygons" in layers else hv.Polygons([])
        
    def _field(self, time, layers):
        if "field" in layers and self.field_da is not None:
            return field_layer(self.field_da, time)
        return hv.Image(np.zeros((2, 2)), kdims=["longitude", "latitude"])

    def _highlight(self, selection):
        if not selection or not self._paths:
            return hv.Path([])
        idx = selection[0]
        if idx >= len(self._paths):
            return hv.Path([])
        path_keys = self._paths[idx]
        pts = []
        for k in path_keys:
            if k in self._coords:
                lon, lat, _ = self._coords[k]
                pts.append((lon, lat))
        if len(pts) < 2:
            return hv.Path([])
        return hv.Path([pts]).opts(color="yellow", line_width=4, alpha=0.9)

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
