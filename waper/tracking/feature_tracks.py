"""Feature-track layer: track individual crests/troughs across time.

Read-only post-processing over a completed `identify_rwps()` run. The tracked
primitive is a single extremum (crest/trough), which moves continuously, rather
than the RWP group, whose membership flips between timesteps.
"""
from dataclasses import dataclass, field
from shapely.geometry import MultiPoint
from ..identification import topology
from .rwp_polygon import get_region_points_and_values, transform_to_stereographic


@dataclass
class Feature:
    time: int
    cluster_id: int
    node_type: str          # "max" | "min"
    lon: float
    lat: float
    scalar: float           # signed amplitude
    footprint: object       # shapely geometry in stereographic metres
    strength: str           # "strong" | "weak"


def feature_overlap(a: Feature, b: Feature) -> float:
    """Footprint intersection area; 0 for different node_type."""
    if a.node_type != b.node_type:
        return 0.0
    return float(a.footprint.intersection(b.footprint).area)


def match_features(prev, curr) -> dict:
    """One-to-one assignment prev-index -> curr-index, greedy by descending
    footprint overlap. Only positive-overlap, same-type pairs are eligible."""
    scored = []
    for i, a in enumerate(prev):
        for j, b in enumerate(curr):
            ov = feature_overlap(a, b)
            if ov > 0.0:
                scored.append((ov, i, j))
    scored.sort(reverse=True)
    used_prev, used_curr, match = set(), set(), {}
    for _, i, j in scored:
        if i in used_prev or j in used_curr:
            continue
        match[i] = j
        used_prev.add(i)
        used_curr.add(j)
    return match


@dataclass
class TrackStep:
    time: int
    lon: float
    lat: float
    scalar: float
    node_type: str
    recovered: bool


@dataclass
class FeatureTrack:
    track_id: int
    steps: list = field(default_factory=list)


def _step(feature: Feature, recovered: bool) -> TrackStep:
    return TrackStep(time=feature.time, lon=feature.lon, lat=feature.lat,
                     scalar=feature.scalar, node_type=feature.node_type,
                     recovered=recovered)


def _strong(features):
    return [f for f in features if f.strength == "strong"]


def _weak(features):
    return [f for f in features if f.strength == "weak"]


def _in_band(feature: Feature, lat_bounds) -> bool:
    if lat_bounds is None:
        return True
    lo, hi = lat_bounds
    return lo <= feature.lat <= hi


def track_features(features_by_time, max_recover_steps: int = 2,
                   lat_bounds=None) -> list:
    """Build continuous feature tracks across timesteps.

    A track is seeded from a strong feature and extended each step to the
    maximally-overlapping strong feature of the same type. If no strong match
    exists, the track may be continued through an overlapping *weak* feature
    (recovery), flagged on that step; it terminates after `max_recover_steps`
    consecutive recovered steps, or when its head leaves `lat_bounds`
    (``(min_lat, max_lat)`` or ``None``).
    """
    tracks = []
    active = []  # list of [track, head_feature, weak_streak]
    next_id = 0
    if features_by_time:
        for f in _strong(features_by_time[0]):
            tr = FeatureTrack(next_id, [_step(f, recovered=False)]); next_id += 1
            tracks.append(tr); active.append([tr, f, 0])

    for t in range(1, len(features_by_time)):
        curr_strong = _strong(features_by_time[t])
        curr_weak = _weak(features_by_time[t])
        heads = [a[1] for a in active]

        strong_match = match_features(heads, curr_strong)
        unmatched_heads = [hi for hi in range(len(active)) if hi not in strong_match]
        weak_match = match_features([active[hi][1] for hi in unmatched_heads], curr_weak)
        # remap weak_match local indices back to head indices
        weak_match = {unmatched_heads[k]: v for k, v in weak_match.items()}

        new_active = []
        for hi, a in enumerate(active):
            if hi in strong_match:
                f = curr_strong[strong_match[hi]]
                if not _in_band(f, lat_bounds):
                    continue                       # leaves band -> terminate
                a[0].steps.append(_step(f, recovered=False))
                new_active.append([a[0], f, 0])
            elif hi in weak_match:
                f = curr_weak[weak_match[hi]]
                if not _in_band(f, lat_bounds) or a[2] + 1 > max_recover_steps:
                    continue                       # band exit or budget exhausted -> terminate
                a[0].steps.append(_step(f, recovered=True))
                new_active.append([a[0], f, a[2] + 1])
            # else: no match at all -> terminate

        matched_curr = set(strong_match.values())
        for j, f in enumerate(curr_strong):
            if j not in matched_curr:
                if not _in_band(f, lat_bounds):
                    continue
                tr = FeatureTrack(next_id, [_step(f, recovered=False)]); next_id += 1
                tracks.append(tr); new_active.append([tr, f, 0])
        active = new_active

    return tracks


import numpy as np
import pandas as pd


def feature_tracks_to_dataframe(tracks) -> pd.DataFrame:
    rows = []
    for tr in tracks:
        for s in tr.steps:
            rows.append(dict(track_id=tr.track_id, time=s.time, lon=s.lon, lat=s.lat,
                             scalar=s.scalar, node_type=s.node_type, recovered=s.recovered))
    return pd.DataFrame(rows, columns=["track_id", "time", "lon", "lat", "scalar",
                                       "node_type", "recovered"])


def phase_velocity(track, dt_hours: float) -> float:
    """Mean eastward propagation in degrees/hour along the track (wraparound-safe).
    Returns nan for a single-step track."""
    steps = track.steps
    if len(steps) < 2:
        return float("nan")
    east = 0.0
    for a, b in zip(steps[:-1], steps[1:]):
        east += ((b.lon - a.lon + 180.0) % 360.0) - 180.0
    span_hours = (steps[-1].time - steps[0].time) * dt_hours
    return east / span_hours if span_hours else float("nan")


def _footprint_from_region(lons, lats, hemisphere):
    xs, ys = transform_to_stereographic(np.asarray(lons), np.asarray(lats),
                                        hemisphere=hemisphere)
    pts = list(zip(np.atleast_1d(xs), np.atleast_1d(ys)))
    geom = MultiPoint(pts)
    return geom.convex_hull if len(pts) >= 3 else geom.buffer(1e4)


def extract_features(tsd, time, scalar_name, clip_value, amplitude_threshold,
                     hemisphere="north"):
    """All extrema of one timestep as Features, footprint = convex hull of the
    extremum's connected-region sample points at a single global `clip_value`."""
    scalar_data = tsd.vtk_data
    g = tsd.association_graph
    max_region = topology.identify_connected_regions(
        scalar_data.clip_scalar(scalars=scalar_name, value=clip_value, invert=False).clean())
    min_region = topology.identify_connected_regions(
        scalar_data.clip_scalar(scalars=scalar_name, value=-clip_value, invert=True).clean())

    features = []
    for node in g.nodes():
        attrs = g.nodes[node]
        region = max_region if attrs["node_type"] == "max" else min_region
        out = get_region_points_and_values(g, node, region, clip_value, scalar_name)
        if out is None:
            continue
        lons, lats, _ = out
        if len(np.atleast_1d(lons)) == 0:
            continue
        lon, lat = attrs["coords"]
        scalar = float(attrs["scalar"])
        features.append(Feature(
            time=time, cluster_id=int(attrs["cluster_id"]), node_type=attrs["node_type"],
            lon=float(lon), lat=float(lat), scalar=scalar,
            footprint=_footprint_from_region(lons, lats, hemisphere),
            strength=("strong" if abs(scalar) >= amplitude_threshold else "weak"),
        ))
    return features
