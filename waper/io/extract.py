from itertools import pairwise

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
    """Flatten the extrema nodes of every identified packet into a table.

    Args:
        waper: A :class:`~waper.interface.api.Waper` after ``identify_rwps()``.

    Returns:
        DataFrame with columns ``time`` (integer index into the time axis, not a
        date), ``rwp_id``, ``node_id``, ``node_type`` (``"max"`` or ``"min"``),
        ``lon``, ``lat`` (degrees), ``scalar`` (the identified field's own units,
        signed), ``cluster_id`` and ``region_id``. One row per node per packet
        per timestep, in path order; a node shared by two packets appears once
        per packet. ``node_id`` is the serialised form of the graph's
        ``(kind, index)`` node key: ``index`` for maxima and ``-index - 1`` for
        minima, so maxima and minima never collide. Empty if nothing was
        identified.
    """
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
    """Flatten the consecutive node-to-node links of every identified packet.

    Args:
        waper: A :class:`~waper.interface.api.Waper` after ``identify_rwps()``.

    Returns:
        DataFrame with columns ``time``, ``rwp_id``, ``src_node_id`` and
        ``dst_node_id``. One row per adjacent pair along a packet's node path, so
        a packet of ``n`` nodes contributes ``n - 1`` rows. The node ids use the
        same serialisation as :func:`extract_nodes`. Empty if nothing was
        identified.
    """
    rows = []
    for t, tsd in enumerate(waper._time_step_data):
        for path in tsd.identified_rwp_paths:
            rwp_id = tsd.rwp_info[tuple(path)]["rwp_id"]
            for a, b in pairwise(path):
                rows.append({"time": t, "rwp_id": rwp_id,
                                 "src_node_id": int(_serialize_node_id(a)),
                                 "dst_node_id": int(_serialize_node_id(b))})
    return pd.DataFrame(rows)

def _zonal_extent_deg(lons):
    if len(lons) < 2:
        return 0.0
    return max(_longitude_separation(a, b) for a in lons for b in lons)

def extract_rwps(waper) -> pd.DataFrame:
    """Summarise each identified wave packet as a single row.

    Args:
        waper: A :class:`~waper.interface.api.Waper` after ``identify_rwps()``.

    Returns:
        DataFrame with columns ``time``, ``rwp_id``, ``weighted_lon``,
        ``weighted_lat`` (the energy-weighted centroid, degrees), ``peak_amp``
        (largest ``abs(scalar)`` over the packet's nodes, in the identified
        field's units), ``n_nodes``, ``zonal_extent_deg`` (the largest shortest-arc
        longitude separation between any two of the packet's nodes, degrees, so
        in ``[0, 180]``) and ``geometry_wkb`` (the footprint polygon as WKB).
        One row per packet per timestep; empty if nothing was identified.

        The polygon is kept in WAPER's native polar-stereographic CRS (metres) —
        see the note in the source and the catalogue's ``hemisphere`` metadata.
    """
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
    """Flatten each packet's sample points back to geographic coordinates.

    The stored sample points are projected (polar-stereographic, metres); they
    are inverted here to longitude/latitude using the run's hemisphere.

    Args:
        waper: A :class:`~waper.interface.api.Waper` after ``identify_rwps()``.

    Returns:
        DataFrame with columns ``time``, ``rwp_id``, ``pt_idx`` (0-based position
        within that packet's sample list), ``lon`` and ``lat`` (degrees). One row
        per sample point per packet per timestep; packets with no sample points
        are skipped, and the frame is empty if there are none at all.
    """
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
            # strict: pyproj returns one output per input coordinate, so the two
            # arrays are the same length as `pts`.
            for i, (lon, lat) in enumerate(zip(lons, lats, strict=True)):
                rows.append({"time": t, "rwp_id": info["rwp_id"], "pt_idx": i,
                                 "lon": float(lon), "lat": float(lat)})
    return pd.DataFrame(rows)

def _key(time, feature):
    return f"{int(time)}:{int(feature)}"

def extract_track_nodes(waper) -> pd.DataFrame:
    """Flatten the vertices of the tracking graph.

    Args:
        waper: A :class:`~waper.interface.api.Waper` after ``track_rwps()``.

    Returns:
        DataFrame with columns ``time`` and ``feature`` (the two halves of the
        graph's ``(time, feature)`` node key), ``lon`` and ``lat`` (the feature's
        centroid, degrees) and ``key`` (the ``"<time>:<feature>"`` string used as
        the join key by :func:`extract_track_edges`). One row per tracking-graph
        node, i.e. per tracked feature per timestep.
    """
    g = waper._tracking_graph
    rows = []
    for (time, feature), nd in g.nodes(data=True):
        lon, lat = nd["coords"]
        rows.append({"time": int(time), "feature": int(feature),
                         "lon": float(lon), "lat": float(lat), "key": _key(time, feature)})
    return pd.DataFrame(rows)

def extract_track_edges(waper) -> pd.DataFrame:
    """Flatten the time-linking edges of the tracking graph.

    Args:
        waper: A :class:`~waper.interface.api.Waper` after ``track_rwps()``.

    Returns:
        DataFrame with columns ``src`` and ``dst`` (``"<time>:<feature>"`` keys
        matching :func:`extract_track_nodes`), ``time_from``, ``feat_from``,
        ``time_to``, ``feat_to``, ``weight`` (the energy overlap of the two
        footprints normalised to ``(0, 1]`` by the larger feature energy) and
        ``distance`` (haversine centroid displacement in **km**). One row per
        edge, always directed forward in time. Missing attributes default to
        ``0.0``.
    """
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
