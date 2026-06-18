"""Feature-track layer: track individual crests/troughs across time.

Read-only post-processing over a completed `identify_rwps()` run. The tracked
primitive is a single extremum (crest/trough), which moves continuously, rather
than the RWP group, whose membership flips between timesteps.
"""
from dataclasses import dataclass, field


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


def track_features(features_by_time) -> list:
    """Build continuous feature tracks across timesteps. A track is seeded from a
    strong feature and extended each step to the maximally-overlapping strong
    feature of the same type; unmatched strong features at t are deaths, unmatched
    strong features at t+1 are births."""
    tracks = []
    # active = list of [track, head_feature]; seed from the first non-empty step
    active = []
    next_id = 0
    if features_by_time:
        for f in _strong(features_by_time[0]):
            tr = FeatureTrack(next_id, [_step(f, recovered=False)]); next_id += 1
            tracks.append(tr); active.append([tr, f])

    for t in range(1, len(features_by_time)):
        curr_strong = _strong(features_by_time[t])
        heads = [a[1] for a in active]
        match = match_features(heads, curr_strong)
        new_active = []
        matched_curr = set(match.values())
        for hi, a in enumerate(active):
            if hi in match:
                f = curr_strong[match[hi]]
                a[0].steps.append(_step(f, recovered=False))
                new_active.append([a[0], f])
            # else: track dies (dropped from active)
        # births: strong curr features not matched to any head
        for j, f in enumerate(curr_strong):
            if j not in matched_curr:
                tr = FeatureTrack(next_id, [_step(f, recovered=False)]); next_id += 1
                tracks.append(tr); new_active.append([tr, f])
        active = new_active

    return tracks
