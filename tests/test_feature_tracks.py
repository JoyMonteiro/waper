from shapely.geometry import box
from waper.tracking.feature_tracks import Feature, feature_overlap, match_features, track_features, FeatureTrack


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


def test_shifted_feature_becomes_one_track():
    fb = [[_feat(0, "max", 0)], [_feat(1, "max", 4)], [_feat(2, "max", 8)]]  # overlaps step to step
    tracks = track_features(fb)
    assert len(tracks) == 1
    assert [s.time for s in tracks[0].steps] == [0, 1, 2]


def test_unmatched_curr_feature_is_a_birth():
    fb = [[_feat(0, "max", 0)], [_feat(1, "max", 4), _feat(1, "max", 100)]]
    tracks = track_features(fb)
    assert len(tracks) == 2                       # the continued one + the newborn
    assert max(len(t.steps) for t in tracks) == 2
    assert min(len(t.steps) for t in tracks) == 1


def test_unmatched_prev_feature_dies():
    fb = [[_feat(0, "max", 0)], [_feat(1, "max", 100)]]   # no overlap -> no match
    tracks = track_features(fb)
    assert len(tracks) == 2
    assert all(len(t.steps) == 1 for t in tracks)
