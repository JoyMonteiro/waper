import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import extract


def write_meta(path, meta: dict) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

def read_meta(path) -> dict:
    return json.loads((Path(path) / "meta.json").read_text())

_TABLES = {
    "nodes": extract.extract_nodes,
    "edges": extract.extract_edges,
    "rwps": extract.extract_rwps,
    "samples": extract.extract_samples,
    "track_nodes": extract.extract_track_nodes,
    "track_edges": extract.extract_track_edges,
}

def _write_table(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "part.parquet", engine="pyarrow", index=False)

def save_catalogue(waper, path, *, meta=None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, fn in _TABLES.items():
        _write_table(fn(waper), path / name)
    # Record the hemisphere so consumers know which polar-stereographic CRS the
    # `rwps.geometry_wkb` polygons live in (the explorer reprojects with it).
    meta = dict(meta or {})
    meta.setdefault("hemisphere", waper._config.hemisphere)
    write_meta(path, meta)


class Catalogue:
    def __init__(self, path, _filters=None):
        self.path = Path(path)
        self.meta = read_meta(self.path) if (self.path/"meta.json").exists() else {}
        self._cache = {}
        self._filters = _filters or {}

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            files = sorted((self.path / name).glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no parquet for table {name!r} in {self.path}")
            self._cache[name] = pd.concat(
                (pd.read_parquet(f, engine="pyarrow") for f in files), ignore_index=True)
        return self._cache[name]

    def filter(self, **kw):
        f = dict(self._filters)
        f.update(kw)
        c = Catalogue(self.path, _filters=f)
        c._cache = self._cache
        return c

    def _apply(self, df, has_amp=False, has_lonlat=False):
        f = self._filters
        if "time" in f and "time" in df:
            df = df[df["time"] == f["time"]]
        if has_amp and "min_amp" in f and "peak_amp" in df:
            df = df[df["peak_amp"] >= f["min_amp"]]
        if has_lonlat and "region" in f:
            w, e, s, n = f["region"]
            lon = df["lon"] if "lon" in df else df["weighted_lon"]
            lat = df["lat"] if "lat" in df else df["weighted_lat"]
            df = df[(lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)]
        return df.reset_index(drop=True)

    def rwps(self):    return self._apply(self.table("rwps"), has_amp=True, has_lonlat=True)
    def nodes(self):   return self._apply(self.table("nodes"), has_lonlat=True)
    def edges(self):   return self._apply(self.table("edges"))
    def samples(self): return self._apply(self.table("samples"), has_lonlat=True)
    def tracks(self):  return self.table("track_edges")

    def amplitudes(self):
        return self.rwps()[["time", "rwp_id", "peak_amp"]]

    def zonal_extent(self):
        return self.rwps()[["time", "rwp_id", "zonal_extent_deg"]]

    def implied_wavenumber(self):
        from waper.identification.utils import _longitude_separation
        out = []
        nodes = self.nodes()
        if nodes.empty:
            return pd.DataFrame(columns=["time", "rwp_id", "implied_wavenumber"])
        for (t, rid), g in nodes.groupby(["time", "rwp_id"]):
            lons = sorted(g["lon"].tolist())
            if len(lons) < 2:
                wn = np.nan
            else:
                gaps = [_longitude_separation(a, b) for a, b in zip(lons[:-1], lons[1:])]
                spacing = float(np.mean(gaps))
                wn = 180.0 / spacing if spacing > 0 else np.nan
            out.append({"time": t, "rwp_id": rid, "implied_wavenumber": wn})
        return pd.DataFrame(out)

    def _track_paths(self):
        """Rebuild the DiGraph and return longest-weight track paths (list of node keys)."""
        import networkx as nx

        from waper.tracking import tracking_graph as tg
        te = self.table("track_edges")
        tn = self.table("track_nodes")
        if te.empty:
            return [], {}
        g = nx.from_pandas_edgelist(te, "src", "dst",
                                    edge_attr=["weight", "distance"], create_using=nx.DiGraph)
        coords = {r.key: (r.lon, r.lat, r.time) for r in tn.itertuples()}
        for k, (lon, lat, _t) in coords.items():
            if k in g:
                g.nodes[k]["coords"] = (lon, lat)
        return tg.get_track_paths(g), coords

    def track_durations(self):
        dt = float(self.meta.get("dt_hours", self.meta.get("cadence_hours", 6)))
        paths, coords = self._track_paths()
        rows = []
        for i, p in enumerate(paths):
            t0 = coords[p[0]][2]
            t1 = coords[p[-1]][2]
            rows.append({"track_id": i, "duration_steps": t1 - t0,
                             "duration_hours": (t1 - t0) * dt})
        return pd.DataFrame(rows)

    def track_propagation(self):
        from waper.identification.utils import _longitude_separation
        paths, coords = self._track_paths()
        rows = []
        for i, p in enumerate(paths):
            lon0 = coords[p[0]][0]
            lon1 = coords[p[-1]][0]
            rows.append({"track_id": i, "propagation_deg": _longitude_separation(lon1, lon0)})
        return pd.DataFrame(rows)

    def group_velocity(self):
        from waper.identification.utils import haversine_distance
        dt = float(self.meta.get("dt_hours", self.meta.get("cadence_hours", 6))) * 3600.0
        paths, coords = self._track_paths()
        rows = []
        for i, p in enumerate(paths):
            if len(p) < 2:
                continue
            lon0, lat0, t0 = coords[p[0]]
            lon1, lat1, t1 = coords[p[-1]]
            km = haversine_distance(lat0, lon0, lat1, lon1)
            secs = (t1 - t0) * dt
            rows.append({"track_id": i, "group_velocity_ms": (km * 1000.0) / secs if secs else np.nan})
        return pd.DataFrame(rows)

    def _digraph(self):
        import networkx as nx
        te = self.table("track_edges")
        return nx.from_pandas_edgelist(te, "src", "dst",
                                       edge_attr=["weight", "distance"], create_using=nx.DiGraph) \
               if not te.empty else nx.DiGraph()

    def _degree_table(self, g, which):
        deg = g.in_degree() if which == "in" else g.out_degree()
        col = "in_degree" if which == "in" else "out_degree"
        rows = []
        for k, d in deg:
            if d > 1:
                t, f = k.split(":")
                rows.append({ "key": k, "time": int(t), "feature": int(f), col: d })
        return pd.DataFrame(rows, columns=["key", "time", "feature", col])

    def merges(self): return self._degree_table(self._digraph(), "in")
    def splits(self): return self._degree_table(self._digraph(), "out")

    def tracks_through(self, box):
        """Track ids whose any centroid falls in box=(w,e,s,n)."""
        w, e, s, n = box
        tn = self.table("track_nodes")
        inbox = tn[(tn.lon >= w) & (tn.lon <= e) & (tn.lat >= s) & (tn.lat <= n)]
        keys = set(inbox["key"])
        paths, _ = self._track_paths()
        return [i for i, p in enumerate(paths) if any(k in keys for k in p)]

    def provenance(self, track_id):
        """Genesis (lon,lat,time) = first node of the track path."""
        paths, coords = self._track_paths()
        lon, lat, t = coords[paths[track_id][0]]
        return {"track_id": track_id, "genesis_lon": lon, "genesis_lat": lat, "genesis_time": t}

    def amplitude_pdf(self, bins=20):
        a = self.amplitudes()["peak_amp"].to_numpy()
        if a.size == 0:
            return pd.DataFrame(columns=["bin_left", "bin_right", "density"])
        dens, edges = np.histogram(a, bins=bins, density=True)
        return pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "density": dens})

    def duration_pdf(self, bins=20):
        d = self.track_durations()["duration_hours"].to_numpy()
        if d.size == 0:
            return pd.DataFrame(columns=["bin_left", "bin_right", "density"])
        dens, edges = np.histogram(d, bins=bins, density=True)
        return pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "density": dens})

    def seasonal_cycle(self, time_to_month=None):
        """Monthly RWP count. time_to_month maps a `time` index to month 1-12."""
        r = self.rwps().copy()
        if r.empty:
            return pd.DataFrame(columns=["month" if time_to_month else "time", "count"])
        if time_to_month is not None:
            r["month"] = r["time"].map(time_to_month)
            return r.groupby("month").size().rename("count").reset_index()
        return r.groupby("time").size().rename("count").reset_index()

    def spatial_frequency(self, dlon=10, dlat=10):
        r = self.rwps()
        if r.empty:
            return pd.DataFrame(columns=["lon_bin", "lat_bin", "count"])
        lon_bin = (r["weighted_lon"] // dlon) * dlon
        lat_bin = (r["weighted_lat"] // dlat) * dlat
        out = r.assign(lon_bin=lon_bin, lat_bin=lat_bin)
        return out.groupby(["lon_bin", "lat_bin"]).size().rename("count").reset_index()

    def cross_stat_correlations(self):
        r = self.rwps()
        amp_extent = r["peak_amp"].corr(r["zonal_extent_deg"]) if len(r) > 2 else np.nan
        return {"amp_vs_extent_r": float(amp_extent) if not np.isnan(amp_extent) else np.nan}

    def rwps_in(self, box):
        w, e, s, n = box
        r = self.rwps()
        return r[(r.weighted_lon >= w) & (r.weighted_lon <= e) & (r.weighted_lat >= s) & (r.weighted_lat <= n)]

    def packet_at(self, point, time):
        """rwp_id whose weighted centroid is nearest to point at this time (or None)."""
        from waper.identification.utils import haversine_distance
        lon0, lat0 = point
        r = self.filter(time=time).rwps()
        if r.empty:
            return None
        d = r.apply(lambda x: haversine_distance(lat0, lon0, x.weighted_lat, x.weighted_lon), axis=1)
        return int(r.iloc[int(d.values.argmin())]["rwp_id"])

    def phase_at(self, point, time):
        """Region's wave-phase: nearest node + fractional position between bracketing nodes."""
        from waper.identification.utils import _longitude_separation
        lon0, _lat0 = point
        nd = self.filter(time=time).nodes()
        if nd.empty:
            return {"nearest_node_type": None, "fractional_position": np.nan, "nearest_node_lon": np.nan}
        nd = nd.assign(dlon=nd["lon"].apply(lambda L: _longitude_separation(L, lon0)))
        nearest = nd.loc[nd["dlon"].idxmin()]
        west = nd[nd["lon"] <= lon0].sort_values("lon")
        east = nd[nd["lon"] > lon0].sort_values("lon")
        if len(west) and len(east):
            a = west.iloc[-1]["lon"]
            b = east.iloc[0]["lon"]
            span = _longitude_separation(b, a)
            frac = _longitude_separation(lon0, a) / span if span > 0 else np.nan
        else:
            frac = np.nan
        return {"nearest_node_type": nearest["node_type"],
                    "fractional_position": float(frac) if not np.isnan(frac) else np.nan,
                    "nearest_node_lon": float(nearest["lon"])}

    def match_points(self, other_df, radius_km=850):
        """For each external point (lon,lat,time), True if within radius of any RWP centroid."""
        from waper.identification.utils import haversine_distance
        r = self.rwps()
        out = []
        for row in other_df.itertuples():
            same_t = r[r["time"] == row.time]
            hit = any(haversine_distance(row.lat, row.lon, x.weighted_lat, x.weighted_lon) <= radius_km
                      for x in same_t.itertuples())
            out.append(bool(hit))
        return other_df.assign(matched=out)


def load_catalogue(path) -> "Catalogue":
    return Catalogue(path)
