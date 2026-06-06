import geopandas as gpd
import holoviews as hv
import hvplot.pandas  # noqa
from shapely import wkb
from .colormaps import joy_nl8, bokeh_palette

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
    return gdf.hvplot.polygons(geo=True, alpha=0.25, c="rwp_id",
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
