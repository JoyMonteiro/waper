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
