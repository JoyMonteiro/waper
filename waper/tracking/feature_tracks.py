"""Feature-track layer: track individual crests/troughs across time.

Read-only post-processing over a completed `identify_rwps()` run. The tracked
primitive is a single extremum (crest/trough), which moves continuously, rather
than the RWP group, whose membership flips between timesteps.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint
from shapely.geometry.base import BaseGeometry

from ..identification import topology
from .rwp_polygon import transform_to_stereographic


@dataclass
class Feature:
    time: int
    cluster_id: int
    node_type: str          # "max" | "min"
    lon: float
    lat: float
    scalar: float           # signed amplitude
    footprint: BaseGeometry  # shapely geometry in stereographic metres
    strength: str           # "strong" | "weak"


def feature_overlap(a: Feature, b: Feature) -> float:
    """Footprint IoU (intersection / union); 0 for different node_type.

    IoU normalises for feature size so a large diffuse footprint cannot win
    the matching competition against a smaller, better-aligned one purely by
    contributing more raw intersection area.
    """
    if a.node_type != b.node_type:
        return 0.0
    inter = float(a.footprint.intersection(b.footprint).area)
    if inter == 0.0:
        return 0.0
    union = a.footprint.area + b.footprint.area - inter
    return inter / union if union > 0.0 else 0.0


def match_features(prev, curr, max_displacement_deg=None) -> dict:
    """Greedy assignment prev-index -> list of curr-indices, by descending IoU.

    Each curr feature is claimed by at most one prev (no merges), but a prev
    may claim multiple curr features (splits). Pairs whose centroid displacement
    exceeds `max_displacement_deg` are excluded before scoring.
    """
    scored = []
    for i, a in enumerate(prev):
        for j, b in enumerate(curr):
            if max_displacement_deg is not None:
                dlon = ((b.lon - a.lon + 180.0) % 360.0) - 180.0
                dlat = b.lat - a.lat
                if (dlon ** 2 + dlat ** 2) ** 0.5 > max_displacement_deg:
                    continue
            ov = feature_overlap(a, b)
            if ov > 0.0:
                scored.append((ov, i, j))
    scored.sort(reverse=True)
    used_curr = set()
    match: dict = {}
    for _, i, j in scored:
        if j in used_curr:
            continue
        match.setdefault(i, []).append(j)
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
    parent_id: Optional[int] = None  # set when this track split off from another


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
                   lat_bounds=None, max_displacement_deg=None,
                   min_split_iou: float = 0.05) -> list:
    """Build continuous feature tracks across timesteps.

    A track is seeded from a strong feature and extended each step to the
    maximally-overlapping strong feature of the same type. If no strong match
    exists, the track may be continued through an overlapping *weak* feature
    (recovery), flagged on that step; it terminates after `max_recover_steps`
    consecutive recovered steps, or when its head leaves `lat_bounds`
    (``(min_lat, max_lat)`` or ``None``).

    When a feature matches multiple curr features (split), the primary child
    continues the existing track; additional children spawn new tracks only if
    their IoU with the head is at least `min_split_iou`, preventing spurious
    splits from large convex-hull footprints with incidental overlap.
    """
    tracks = []
    active = []  # list of [track, head_feature, weak_streak]
    next_id = 0
    if features_by_time:
        for f in _strong(features_by_time[0]):
            tr = FeatureTrack(next_id, [_step(f, recovered=False)])
            next_id += 1
            tracks.append(tr)
            active.append([tr, f, 0])

    for t in range(1, len(features_by_time)):
        curr_strong = _strong(features_by_time[t])
        curr_weak = _weak(features_by_time[t])
        heads = [a[1] for a in active]

        strong_match = match_features(heads, curr_strong,
                                      max_displacement_deg=max_displacement_deg)
        unmatched_heads = [hi for hi in range(len(active)) if hi not in strong_match]
        weak_match = match_features([active[hi][1] for hi in unmatched_heads], curr_weak,
                                    max_displacement_deg=max_displacement_deg)
        weak_match = {unmatched_heads[k]: v for k, v in weak_match.items()}

        new_active = []
        for hi, a in enumerate(active):
            if hi in strong_match:
                children = strong_match[hi]
                parent_history = list(a[0].steps)  # snapshot before appending

                # Primary child: continue the existing track
                f0 = curr_strong[children[0]]
                if _in_band(f0, lat_bounds):
                    a[0].steps.append(_step(f0, recovered=False))
                    new_active.append([a[0], f0, 0])

                # Additional children (splits): spawn only if IoU is substantial
                head = a[1]
                for j in children[1:]:
                    f = curr_strong[j]
                    if not _in_band(f, lat_bounds):
                        continue
                    if feature_overlap(head, f) < min_split_iou:
                        continue
                    tr = FeatureTrack(next_id,
                                      [*parent_history, _step(f, recovered=False)],
                                      parent_id=a[0].track_id)
                    next_id += 1
                    tracks.append(tr)
                    new_active.append([tr, f, 0])

            elif hi in weak_match:
                children = weak_match[hi]
                parent_history = list(a[0].steps)

                f0 = curr_weak[children[0]]
                if _in_band(f0, lat_bounds) and a[2] + 1 <= max_recover_steps:
                    a[0].steps.append(_step(f0, recovered=True))
                    new_active.append([a[0], f0, a[2] + 1])

                for j in children[1:]:
                    f = curr_weak[j]
                    if not _in_band(f, lat_bounds) or a[2] + 1 > max_recover_steps:
                        continue
                    tr = FeatureTrack(next_id,
                                      [*parent_history, _step(f, recovered=True)],
                                      parent_id=a[0].track_id)
                    next_id += 1
                    tracks.append(tr)
                    new_active.append([tr, f, a[2] + 1])
            # else: no match -> track terminates

        matched_curr = set()
        for children_list in strong_match.values():
            matched_curr.update(children_list)
        for j, f in enumerate(curr_strong):
            if j not in matched_curr:
                if not _in_band(f, lat_bounds):
                    continue
                tr = FeatureTrack(next_id, [_step(f, recovered=False)])
                next_id += 1
                tracks.append(tr)
                new_active.append([tr, f, 0])
        active = new_active

    return tracks


def feature_tracks_to_dataframe(tracks) -> pd.DataFrame:
    rows = []
    for tr in tracks:
        for s in tr.steps:
            rows.append({"track_id": tr.track_id, "time": s.time, "lon": s.lon, "lat": s.lat,
                             "scalar": s.scalar, "node_type": s.node_type, "recovered": s.recovered})
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


def _build_point_adjacency(mesh, mask):
    """Adjacency dict (global point index → set of neighbours) restricted to mask."""
    adj = {int(i): set() for i in np.where(mask)[0]}
    faces = mesh.faces
    # fast path: all-triangle mesh (pyvista flat format [3,a,b,c, 3,a,b,c, ...])
    if len(faces) > 0 and faces[0] == 3 and len(faces) % 4 == 0:
        tris = faces.reshape(-1, 4)[:, 1:]
        for ca, cb in ((0, 1), (1, 2), (0, 2)):
            a_arr, b_arr = tris[:, ca], tris[:, cb]
            both = mask[a_arr] & mask[b_arr]
            for a, b in zip(a_arr[both].tolist(), b_arr[both].tolist()):
                adj[a].add(b)
                adj[b].add(a)
    else:
        i = 0
        while i < len(faces):
            n = int(faces[i])
            cell = faces[i + 1: i + 1 + n].tolist()
            i += n + 1
            in_r = [p for p in cell if mask[p]]
            for ai in range(len(in_r)):
                for bi in range(ai + 1, len(in_r)):
                    adj[in_r[ai]].add(in_r[bi])
                    adj[in_r[bi]].add(in_r[ai])
    return adj


def _bfs_partition(adj, seed_global_pts):
    """Multi-source BFS; returns {global_pt_idx: seed_index}."""
    from collections import deque
    assignment = {}
    queue = deque()
    for seed_idx, pt in enumerate(seed_global_pts):
        if pt not in assignment:
            assignment[pt] = seed_idx
            queue.append(pt)
    while queue:
        pt = queue.popleft()
        s = assignment[pt]
        for nb in adj.get(pt, ()):
            if nb not in assignment:
                assignment[nb] = s
                queue.append(nb)
    return assignment


def extract_features(tsd, time, scalar_name, clip_value, amplitude_threshold,
                     hemisphere="north", footprint_fraction=None):
    """All extrema of one timestep as Features with BFS-partitioned footprints.

    When multiple extrema share a connected region at `clip_value`, the region
    points are partitioned by seeded flood fill on the mesh: each point is
    assigned to the extremum whose BFS wavefront reaches it first, so footprint
    boundaries fall at the natural saddle points between features.

    `footprint_fraction`: if set (e.g. 3), only mesh points whose |scalar| is
    at least 1/footprint_fraction of the extremum's |scalar| are included in
    the convex hull, preventing large diffuse tails from inflating footprints.
    """
    scalar_data = tsd.vtk_data
    g = tsd.association_graph
    max_region = topology.identify_connected_regions(
        scalar_data.clip_scalar(scalars=scalar_name, value=clip_value, invert=False).clean())
    min_region = topology.identify_connected_regions(
        scalar_data.clip_scalar(scalars=scalar_name, value=-clip_value, invert=True).clean())

    # Pass 1: map each node to its (node_type, region_id)
    node_region = {}
    for node in g.nodes():
        attrs = g.nodes[node]
        if abs(attrs["scalar"]) < clip_value:
            continue
        region = max_region if attrs["node_type"] == "max" else min_region
        closest = region.find_closest_point(attrs["spherical_coords"])
        rid = int(region.point_data["RegionId"][closest])
        node_region[node] = (attrs["node_type"], rid)

    # Pass 2: group nodes that share a region
    groups = {}
    for node, key in node_region.items():
        groups.setdefault(key, []).append(node)

    features = []
    for (ntype, rid), nodes_in_region in groups.items():
        region = max_region if ntype == "max" else min_region
        mask = region.point_data["RegionId"] == rid
        global_idxs = np.where(mask)[0]
        if len(global_idxs) == 0:
            continue

        lons_r    = region["Longitude"][mask]
        lats_r    = region["Latitude"][mask]
        scalars_r = np.asarray(region[scalar_name][mask], dtype=float)
        xs_r, ys_r = transform_to_stereographic(np.asarray(lons_r), np.asarray(lats_r),
                                                 hemisphere=hemisphere)

        if len(nodes_in_region) == 1:
            node_lons    = {nodes_in_region[0]: lons_r}
            node_lats    = {nodes_in_region[0]: lats_r}
            node_scalars = {nodes_in_region[0]: scalars_r}
        else:
            # Seed = closest region point (in stereo) to each extremum
            seeds = []
            for node in nodes_in_region:
                attrs = g.nodes[node]
                ex, ey = transform_to_stereographic(
                    np.array([float(attrs["coords"][0])]),
                    np.array([float(attrs["coords"][1])]),
                    hemisphere=hemisphere)
                local_nearest = int(np.argmin((xs_r - ex) ** 2 + (ys_r - ey) ** 2))
                seeds.append(int(global_idxs[local_nearest]))

            adj = _build_point_adjacency(region, mask)
            assign_map = _bfs_partition(adj, seeds)

            node_lons    = {node: [] for node in nodes_in_region}
            node_lats    = {node: [] for node in nodes_in_region}
            node_scalars = {node: [] for node in nodes_in_region}
            for i, gidx in enumerate(global_idxs):
                s = assign_map.get(int(gidx))
                if s is not None:
                    node_lons[nodes_in_region[s]].append(float(lons_r[i]))
                    node_lats[nodes_in_region[s]].append(float(lats_r[i]))
                    node_scalars[nodes_in_region[s]].append(float(scalars_r[i]))
            node_lons    = {n: np.array(v) for n, v in node_lons.items()}
            node_lats    = {n: np.array(v) for n, v in node_lats.items()}
            node_scalars = {n: np.array(v) for n, v in node_scalars.items()}

        for node in nodes_in_region:
            lons_f, lats_f = node_lons[node], node_lats[node]
            if len(np.atleast_1d(lons_f)) == 0:
                continue
            attrs = g.nodes[node]
            lon, lat = attrs["coords"]
            scalar = float(attrs["scalar"])

            # Trim footprint to points within 1/footprint_fraction of peak amplitude
            if footprint_fraction is not None:
                sv = node_scalars[node]
                cutoff = scalar / footprint_fraction  # signed: positive for max, negative for min
                keep = sv >= cutoff if ntype == "max" else sv <= cutoff
                lons_f = lons_f[keep]
                lats_f = lats_f[keep]
                if len(lons_f) == 0:
                    lons_f = np.array([lon])
                    lats_f = np.array([lat])

            features.append(Feature(
                time=time, cluster_id=int(attrs["cluster_id"]), node_type=ntype,
                lon=float(lon), lat=float(lat), scalar=scalar,
                footprint=_footprint_from_region(lons_f, lats_f, hemisphere),
                strength="strong" if abs(scalar) >= amplitude_threshold else "weak",
            ))
    return features
