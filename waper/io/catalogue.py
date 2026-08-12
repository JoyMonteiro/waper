import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from . import extract


def write_meta(path, meta: dict) -> None:
    """Write a catalogue's metadata to ``<path>/meta.json``.

    Creates ``path`` (and parents) if needed. Values that are not JSON
    serialisable are stringified.

    Args:
        path: Catalogue directory.
        meta: Metadata mapping to store.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

def read_meta(path) -> dict:
    """Read a catalogue's ``meta.json``.

    Args:
        path: Catalogue directory.

    Returns:
        The stored metadata mapping.

    Raises:
        FileNotFoundError: If the directory has no ``meta.json``.
    """
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
    """Write a completed run to disk as a queryable catalogue.

    Each of the six tables (``nodes``, ``edges``, ``rwps``, ``samples``,
    ``track_nodes``, ``track_edges``) is extracted from ``waper`` and written to
    ``<path>/<table>/part.parquet``, alongside a ``meta.json``.

    Args:
        waper: A :class:`~waper.interface.api.Waper` after ``identify_rwps()``
            and ``track_rwps()``.
        path: Destination directory; created if it does not exist. Existing
            ``part.parquet`` files are overwritten.
        meta: Extra metadata to record. ``hemisphere`` is filled in from the
            run's config unless the caller already supplied it; consumers need
            it to know which polar-stereographic CRS ``rwps.geometry_wkb`` is in.
    """
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
    """The on-disk, queryable form of a :class:`~waper.interface.api.Waper` run.

    A catalogue is a directory written by :func:`save_catalogue` and opened by
    :func:`load_catalogue`. It holds the six parquet tables produced by
    :mod:`waper.io.extract` plus a ``meta.json``, and exposes them through
    :meth:`table` (raw, cached) and through the query methods below, which apply
    the currently active filters.

    Filters are set with :meth:`filter`, which returns a new ``Catalogue``
    sharing this one's parquet cache rather than mutating in place, so chained
    queries stay cheap.

    ``meta`` carries the run's ``hemisphere``, which
    :func:`waper.interface.explorer._hemisphere` reads (defaulting to
    ``"north"``) to pick the default map projection and to declare the CRS of
    the stored polygons.

    Everything here works in the integer ``time`` index of the run, not in
    dates. Durations are converted to hours using the ``dt_hours`` (or
    ``cadence_hours``) metadata key, which defaults to 6.
    """

    def __init__(self, path, _filters=None):
        """Open the catalogue directory at ``path``.

        Args:
            path: Catalogue directory. Its ``meta.json`` is read now if present,
                otherwise ``meta`` is an empty dict; the parquet tables are read
                lazily.
            _filters: Internal. The filter mapping propagated by :meth:`filter`.
        """
        self.path = Path(path)
        self.meta = read_meta(self.path) if (self.path/"meta.json").exists() else {}
        self._cache = {}
        self._filters = _filters or {}

    def table(self, name: str) -> pd.DataFrame:
        """Return one whole table, unfiltered.

        All ``*.parquet`` parts under ``<path>/<name>/`` are concatenated on
        first use and cached for the lifetime of this object (and of any
        :meth:`filter` view derived from it).

        Args:
            name: One of ``nodes``, ``edges``, ``rwps``, ``samples``,
                ``track_nodes``, ``track_edges``.

        Returns:
            The table as written by the matching :mod:`waper.io.extract`
            function.

        Raises:
            FileNotFoundError: If the table directory holds no parquet files.
        """
        if name not in self._cache:
            files = sorted((self.path / name).glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no parquet for table {name!r} in {self.path}")
            self._cache[name] = pd.concat(
                (pd.read_parquet(f, engine="pyarrow") for f in files), ignore_index=True)
        return self._cache[name]

    def filter(self, **kw):
        """Return a filtered view of this catalogue.

        Filters accumulate: keys given here override same-named keys already
        set, and the rest are kept. Only the query methods honour them —
        :meth:`table` always returns the raw table.

        Args:
            **kw: Recognised keys are ``time`` (keep rows at that integer time
                index), ``min_amp`` (keep packets with ``peak_amp`` at or above
                it; only applies to :meth:`rwps` and what derives from it) and
                ``region`` as a ``(west, east, south, north)`` degree box (only
                applies to tables carrying coordinates). Unrecognised keys are
                stored but ignored.

        Returns:
            A new ``Catalogue`` over the same path, sharing this one's cache.
        """
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

    def rwps(self):
        """Per-packet summary rows, filtered.

        Returns:
            The ``rwps`` table (see :func:`waper.io.extract.extract_rwps`), with
            the ``time``, ``min_amp`` and ``region`` filters applied; ``region``
            tests the weighted centroid. One row per packet per timestep, index
            reset.
        """
        return self._apply(self.table("rwps"), has_amp=True, has_lonlat=True)

    def nodes(self):
        """Per-node rows of every packet, filtered.

        Returns:
            The ``nodes`` table (see :func:`waper.io.extract.extract_nodes`),
            with the ``time`` and ``region`` filters applied; ``region`` tests
            each node's own ``lon``/``lat``, so it can keep part of a packet.
            ``min_amp`` does not apply. Index reset.
        """
        return self._apply(self.table("nodes"), has_lonlat=True)

    def edges(self):
        """Per-link rows within packets, filtered.

        Returns:
            The ``edges`` table (see :func:`waper.io.extract.extract_edges`),
            with only the ``time`` filter applied — the table carries no
            coordinates or amplitude. Index reset.
        """
        return self._apply(self.table("edges"))

    def samples(self):
        """Per-sample-point rows of every packet, filtered.

        Returns:
            The ``samples`` table (see
            :func:`waper.io.extract.extract_samples`), with the ``time`` and
            ``region`` filters applied to each point. Index reset.
        """
        return self._apply(self.table("samples"), has_lonlat=True)

    def tracks(self):
        """The raw time-linking edges of the tracking graph.

        Filters are **not** applied.

        Returns:
            The ``track_edges`` table (see
            :func:`waper.io.extract.extract_track_edges`), one row per edge.
        """
        return self.table("track_edges")

    def amplitudes(self):
        """Peak amplitude of every packet.

        Returns:
            DataFrame with columns ``time``, ``rwp_id`` and ``peak_amp`` (the
            identified field's units), one row per packet per timestep, honouring
            the active filters.
        """
        return self.rwps()[["time", "rwp_id", "peak_amp"]]

    def zonal_extent(self):
        """Longitudinal span of every packet.

        Returns:
            DataFrame with columns ``time``, ``rwp_id`` and ``zonal_extent_deg``
            — the largest shortest-arc longitude separation between any two of
            the packet's nodes, in degrees and therefore in ``[0, 180]``. One row
            per packet per timestep, honouring the active filters.
        """
        return self.rwps()[["time", "rwp_id", "zonal_extent_deg"]]

    def implied_wavenumber(self):
        """Zonal wavenumber implied by each packet's node spacing.

        A packet's nodes alternate between maxima and minima, so the mean
        separation between longitudinally adjacent nodes is half a wavelength and
        the implied wavenumber is ``180 / spacing``. Node longitudes are sorted
        numerically before pairing, so the estimate is taken across the 0/360
        seam for packets that straddle it.

        Returns:
            DataFrame with columns ``time``, ``rwp_id`` and
            ``implied_wavenumber`` (dimensionless waves per latitude circle). One
            row per packet per timestep that survives the filters; ``NaN`` where
            the packet has fewer than two nodes or zero mean spacing. Empty (with
            those three columns) when no nodes pass the filters.
        """
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
                gaps = [_longitude_separation(a, b) for a, b in pairwise(lons)]
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
        """Lifetime of each track.

        Tracks are the longest-weight paths through the stored tracking graph, so
        ``track_id`` is just this catalogue's index into that path list and is
        not stable across catalogues. Filters are **not** applied.

        Returns:
            DataFrame with columns ``track_id``, ``duration_steps`` (the time
            index of the last node minus that of the first, so a two-timestep
            track scores 1) and ``duration_hours`` (that times the ``dt_hours``
            /``cadence_hours`` metadata, default 6). One row per track; a
            column-less empty frame when there are no tracking edges.
        """
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
        """How far each track moved in longitude, end to end.

        The value is the *shortest-arc* longitude separation between the track's
        first and last node, so it is **unsigned and bounded by 180 degrees**:
        it carries no eastward/westward sense and it saturates for tracks that
        travel more than half way around a latitude circle. Only the two
        endpoints enter — the path in between is ignored.

        Filters are **not** applied, and ``track_id`` indexes this catalogue's
        track-path list.

        Returns:
            DataFrame with columns ``track_id`` and ``propagation_deg`` (degrees,
            in ``[0, 180]``). One row per track; a column-less empty frame when
            there are no tracking edges.
        """
        from waper.identification.utils import _longitude_separation
        paths, coords = self._track_paths()
        rows = []
        for i, p in enumerate(paths):
            lon0 = coords[p[0]][0]
            lon1 = coords[p[-1]][0]
            rows.append({"track_id": i, "propagation_deg": _longitude_separation(lon1, lon0)})
        return pd.DataFrame(rows)

    def group_velocity(self):
        """Mean end-to-end propagation speed of each track.

        This is the great-circle (haversine) distance between the track's first
        and last node divided by the elapsed time, where elapsed time is
        ``(t_last - t_first)`` steps times the ``dt_hours``/``cadence_hours``
        metadata (default 6 h). It is therefore a **speed, not a velocity**:
        haversine distance is non-negative, so the result is always positive and
        encodes no direction — an eastward and a westward track of equal reach
        score identically, and there is no sign convention to read off. It is
        also a straight endpoint-to-endpoint average: a track that doubles back
        scores low, and a packet circling more than half the globe is
        under-measured because the great-circle distance takes the short way
        round.

        Filters are **not** applied, and ``track_id`` indexes this catalogue's
        track-path list.

        Returns:
            DataFrame with columns ``track_id`` and ``group_velocity_ms``
            (metres per second, non-negative; ``NaN`` when first and last node
            share a time index). One row per track of at least two nodes —
            single-node tracks are dropped, so ``track_id`` can have gaps — and a
            column-less empty frame when there are no tracking edges.
        """
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

    def merges(self):
        """Tracking-graph nodes where two or more features converge.

        Filters are **not** applied.

        Returns:
            DataFrame with columns ``key`` (``"<time>:<feature>"``), ``time``,
            ``feature`` and ``in_degree``, one row per node with an in-degree
            above 1. Empty (with those columns) if there are no merges.
        """
        return self._degree_table(self._digraph(), "in")

    def splits(self):
        """Tracking-graph nodes from which two or more features diverge.

        Filters are **not** applied.

        Returns:
            DataFrame with columns ``key`` (``"<time>:<feature>"``), ``time``,
            ``feature`` and ``out_degree``, one row per node with an out-degree
            above 1. Empty (with those columns) if there are no splits.
        """
        return self._degree_table(self._digraph(), "out")

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
        """Normalised histogram of per-packet peak amplitude.

        Args:
            bins: Bin count or explicit edges, passed to :func:`numpy.histogram`.

        Returns:
            DataFrame with columns ``bin_left``, ``bin_right`` (amplitude, in the
            identified field's units) and ``density`` (a probability density, so
            it integrates to 1 over the bins rather than summing to 1). One row
            per bin, from the filtered :meth:`amplitudes`; empty with those
            columns if no packet passes the filters.
        """
        a = self.amplitudes()["peak_amp"].to_numpy()
        if a.size == 0:
            return pd.DataFrame(columns=["bin_left", "bin_right", "density"])
        dens, edges = np.histogram(a, bins=bins, density=True)
        return pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "density": dens})

    def duration_pdf(self, bins=20):
        """Normalised histogram of track lifetime.

        Args:
            bins: Bin count or explicit edges, passed to :func:`numpy.histogram`.

        Returns:
            DataFrame with columns ``bin_left``, ``bin_right`` (**hours**) and
            ``density`` (a probability density over hours). One row per bin, over
            the unfiltered :meth:`track_durations`; empty with those columns if
            there are no tracks.
        """
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
        """Count packet occurrences on a regular lon/lat grid.

        Each packet is binned by its weighted centroid, so a packet lasting ten
        timesteps contributes ten counts.

        Args:
            dlon: Longitude bin width in degrees.
            dlat: Latitude bin width in degrees.

        Returns:
            DataFrame with columns ``lon_bin``, ``lat_bin`` (the **left/lower
            edge** of the bin, in degrees) and ``count``. One row per occupied
            bin; empty bins are omitted, and the frame is empty with those
            columns if no packet passes the filters.
        """
        r = self.rwps()
        if r.empty:
            return pd.DataFrame(columns=["lon_bin", "lat_bin", "count"])
        lon_bin = (r["weighted_lon"] // dlon) * dlon
        lat_bin = (r["weighted_lat"] // dlat) * dlat
        out = r.assign(lon_bin=lon_bin, lat_bin=lat_bin)
        return out.groupby(["lon_bin", "lat_bin"]).size().rename("count").reset_index()

    def cross_stat_correlations(self):
        """Correlate the per-packet summary statistics against each other.

        Returns:
            Dict with the single key ``amp_vs_extent_r``: the Pearson
            correlation of ``peak_amp`` with ``zonal_extent_deg`` over the
            filtered packets, or ``NaN`` when fewer than three packets pass the
            filters (or when either series is constant).
        """
        r = self.rwps()
        amp_extent = r["peak_amp"].corr(r["zonal_extent_deg"]) if len(r) > 2 else np.nan
        return {"amp_vs_extent_r": float(amp_extent) if not np.isnan(amp_extent) else np.nan}

    def rwps_in(self, box):
        """Packets whose weighted centroid falls inside a lon/lat box.

        The test is a plain inclusive comparison on both axes, so the box may not
        cross the 0/360 longitude seam. This is applied on top of any active
        filters (including a ``region`` filter).

        Args:
            box: ``(west, east, south, north)`` in degrees.

        Returns:
            The matching subset of :meth:`rwps`, same columns, one row per packet
            per timestep. The index is inherited from :meth:`rwps`, not reset.
        """
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
    """Open a catalogue directory written by :func:`save_catalogue`.

    Args:
        path: Catalogue directory.

    Returns:
        An unfiltered :class:`Catalogue`. The parquet tables are read lazily on
        first query, so a missing or malformed table only fails at that point.
    """
    return Catalogue(path)
