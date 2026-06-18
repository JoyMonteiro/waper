from shapely.geometry import box
from waper.tracking.feature_tracks import Feature, feature_overlap, match_features


def _feat(t, ntype, x0, strength="strong", scalar=20.0):
    # footprint is a 10x10 stereographic box starting at x0
    return Feature(time=t, cluster_id=0, node_type=ntype, lon=0.0, lat=40.0,
                   scalar=scalar, footprint=box(x0, 0, x0 + 10, 10), strength=strength)


def test_overlap_same_type_area():
    a = _feat(0, "max", 0); b = _feat(1, "max", 5)
    assert feature_overlap(a, b) == 50.0          # 5 wide x 10 tall


def test_overlap_zero_for_different_type():
    a = _feat(0, "max", 0); b = _feat(1, "min", 0)   # identical box, different type
    assert feature_overlap(a, b) == 0.0


def test_match_is_one_to_one_greedy_by_overlap():
    prev = [_feat(0, "max", 0), _feat(0, "max", 100)]
    curr = [_feat(1, "max", 2), _feat(1, "max", 4)]   # both overlap prev[0]; prev[1] overlaps neither
    m = match_features(prev, curr)
    assert m == {0: 0}                                # prev[0] takes its best (curr[0], overlap 8>6); curr[1] free but prev exhausted
