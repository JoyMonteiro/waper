import pandas as pd
from shapely import wkb

from waper.identification.utils import _longitude_separation
from waper.tracking.rwp_polygon import transform_to_stereographic


def _serialize_node_id(n):
    if isinstance(n, tuple):
        val = n[1]
        return val if n[0] == "max" else -val - 1
    return int(n)

def extract_nodes(waper) -> pd.DataFrame:
    rows = []
    for t, tsd in enumerate(waper._time_step_data):
        g = tsd.pruned_graph
        for path in tsd.identified_rwp_paths:
            rwp_id = tsd.rwp_info[tuple(path)]["rwp_id"]
            for n in path:
                nd = g.nodes[n]
                lon, lat = nd["coords"]
                rows.append({"time": t, "rwp_id": rwp_id, "node_id": int(_serialize_node_id(n)),
                                 "node_type": nd["node_type"], "lon": float(lon), "lat": float(lat),
                                 "scalar": float(nd["scalar"]), "cluster_id": int(nd["cluster_id"]),
                                 "region_id": int(nd["region_id"])})
    return pd.DataFrame(rows)

def extract_edges(waper) -> pd.DataFrame:
    rows = []
    for t, tsd in enumerate(waper._time_step_data):
        for path in tsd.identified_rwp_paths:
            rwp_id = tsd.rwp_info[tuple(path)]["rwp_id"]
            for a, b in zip(path[:-1], path[1:]):
                rows.append({"time": t, "rwp_id": rwp_id,
                                 "src_node_id": int(_serialize_node_id(a)),
                                 "dst_node_id": int(_serialize_node_id(b))})
    return pd.DataFrame(rows)

def _zonal_extent_deg(lons):
    if len(lons) < 2:
        return 0.0
    return max(_longitude_separation(a, b) for a in lons for b in lons)

def extract_rwps(waper) -> pd.DataFrame:
    # NOTE: the polygon is kept in WAPER's native polar-stereographic CRS
    # (metres, +proj=stere +lat_0=±90). Polar stereographic has no antimeridian
    # seam, so dateline-straddling packets stay contiguous — converting to
    # lon/lat here would tear them apart. The catalogue's `hemisphere` meta tells
    # consumers which stereographic CRS the geometry is in (see explorer).
    rows = []
    for t, tsd in enumerate(waper._time_step_data):
        g = tsd.pruned_graph
        for path in tsd.identified_rwp_paths:
            info = tsd.rwp_info[tuple(path)]
            scalars = [abs(g.nodes[n]["scalar"]) for n in path]
            lons = [g.nodes[n]["coords"][0] for n in path]
            rows.append({
                "time": t, "rwp_id": info["rwp_id"],
                "weighted_lon": float(info["weighted_longitude"]),
                "weighted_lat": float(info["weighted_latitude"]),
                "peak_amp": float(max(scalars)), "n_nodes": len(path),
                "zonal_extent_deg": float(_zonal_extent_deg(lons)),
                "geometry_wkb": wkb.dumps(info["polygon"]),
            })
    return pd.DataFrame(rows)

def extract_samples(waper) -> pd.DataFrame:
    hemisphere = waper._config.hemisphere
    rows = []
    for t, tsd in enumerate(waper._time_step_data):
        for path in tsd.identified_rwp_paths:
            info = tsd.rwp_info[tuple(path)]
            pts = info["sample_points"]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            lons, lats = transform_to_stereographic(xs, ys, hemisphere=hemisphere, inverse=True)
            for i, (lon, lat) in enumerate(zip(lons, lats)):
                rows.append({"time": t, "rwp_id": info["rwp_id"], "pt_idx": i,
                                 "lon": float(lon), "lat": float(lat)})
    return pd.DataFrame(rows)

def _key(time, feature):
    return f"{int(time)}:{int(feature)}"

def extract_track_nodes(waper) -> pd.DataFrame:
    g = waper._tracking_graph
    rows = []
    for (time, feature), nd in g.nodes(data=True):
        lon, lat = nd["coords"]
        rows.append({"time": int(time), "feature": int(feature),
                         "lon": float(lon), "lat": float(lat), "key": _key(time, feature)})
    return pd.DataFrame(rows)

def extract_track_edges(waper) -> pd.DataFrame:
    g = waper._tracking_graph
    rows = []
    for (a, b, ed) in g.edges(data=True):
        (t0, f0), (t1, f1) = a, b
        rows.append({"src": _key(t0, f0), "dst": _key(t1, f1),
                         "time_from": int(t0), "feat_from": int(f0),
                         "time_to": int(t1), "feat_to": int(f1),
                         "weight": float(ed.get("weight", 0.0)),
                         "distance": float(ed.get("distance", 0.0))})
    return pd.DataFrame(rows)
