import pytest
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


def test_feature_recovered_through_weak_step():
    # strong at t0, only a WEAK overlapping feature at t1, strong again at t2
    fb = [
        [_feat(0, "max", 0)],
        [_feat(1, "max", 4, strength="weak")],
        [_feat(2, "max", 8)],
    ]
    tracks = track_features(fb, max_recover_steps=2)
    assert len(tracks) == 1
    steps = tracks[0].steps
    assert [s.time for s in steps] == [0, 1, 2]
    assert steps[1].recovered is True and steps[2].recovered is False


def test_track_dies_after_recovery_budget_exhausted():
    fb = [
        [_feat(0, "max", 0)],
        [_feat(1, "max", 4, strength="weak")],
        [_feat(2, "max", 8, strength="weak")],
        [_feat(3, "max", 12, strength="weak")],   # 3rd consecutive weak > budget(2)
    ]
    tracks = track_features(fb, max_recover_steps=2)
    assert len(tracks) == 1
    assert [s.time for s in tracks[0].steps] == [0, 1, 2]   # terminated before t3


def test_track_ends_when_feature_leaves_lat_band():
    f0 = _feat(0, "max", 0); f0.lat = 60.0
    f1 = _feat(1, "max", 4); f1.lat = 85.0          # outside band
    tracks = track_features([[f0], [f1]], lat_bounds=(20.0, 80.0))
    assert len(tracks) == 1 and len(tracks[0].steps) == 1


import pandas as pd
from waper.tracking.feature_tracks import feature_tracks_to_dataframe, phase_velocity


def test_dataframe_has_one_row_per_step():
    fb = [[_feat(0, "max", 0)], [_feat(1, "max", 4)]]
    df = feature_tracks_to_dataframe(track_features(fb))
    assert set(df.columns) >= {"track_id", "time", "lon", "lat", "scalar", "node_type", "recovered"}
    assert len(df) == 2


def test_phase_velocity_eastward_degrees_per_hour():
    f0 = _feat(0, "max", 0); f0.lon = 10.0
    f1 = _feat(1, "max", 4); f1.lon = 16.0     # +6 deg over 6 h -> 1.0 deg/h east
    (track,) = track_features([[f0], [f1]])
    assert abs(phase_velocity(track, dt_hours=6.0) - 1.0) < 1e-9


def test_phase_velocity_handles_dateline():
    f0 = _feat(0, "max", 0); f0.lon = 179.0
    f1 = _feat(1, "max", 4); f1.lon = -179.0   # +2 deg east across dateline
    (track,) = track_features([[f0], [f1]])
    assert abs(phase_velocity(track, dt_hours=2.0) - 1.0) < 1e-9


import xarray as xr
from waper.interface.api import Waper
from waper.tracking.feature_tracks import extract_features, Feature


def test_extract_features_from_real_timestep(two_timestep_field):
    ds = xr.Dataset({"v": two_timestep_field})
    w = Waper(data_array=ds, scalar_name="v",
              latitude_label="latitude", longitude_label="longitude", time_label="time",
              clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80.1,
              node_pruning_threshold=15, edge_pruning_threshold=3e-5,
              track_pruning_threshold=0.3, max_edge_weight=1, debug=False)
    w.identify_rwps()
    feats = extract_features(w._time_step_data[0], time=0, scalar_name="v",
                             clip_value=2, amplitude_threshold=10)
    assert len(feats) > 0
    assert all(isinstance(f, Feature) for f in feats)
    assert all(f.node_type in ("max", "min") for f in feats)
    assert all(f.strength in ("strong", "weak") for f in feats)
    assert all(f.footprint.area > 0 for f in feats)        # real region hulls, not points
    # there are at least as many features as pruned RWP nodes (nothing was dropped)
    n_pruned_nodes = sum(len(p) for p in w._time_step_data[0].identified_rwp_paths)
    assert len(feats) >= n_pruned_nodes


import os

DATASET = "datasets/forecast_bust.nc"


@pytest.mark.skipif(not os.path.exists(DATASET), reason="forecast_bust.nc not present")
def test_feature_tracks_are_continuous_on_real_data():
    import numpy as np, xarray as xr
    from waper.interface.api import Waper
    from waper.tracking.feature_tracks import extract_features, track_features

    ds = xr.open_dataset(DATASET)
    av = np.abs(ds["v"].values).ravel()
    thr = float(np.percentile(av, 90))
    w = Waper(data_array=ds, scalar_name="v",
              latitude_label="latitude", longitude_label="longitude", time_label="time",
              clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80,
              node_pruning_threshold=20, edge_pruning_threshold=0.02, max_edge_weight=1,
              track_pruning_threshold=0.3)
    w.identify_rwps()
    fb = [extract_features(w._time_step_data[t], t, "v", 2, thr)
          for t in range(ds.sizes["time"])]
    tracks = track_features(fb, max_recover_steps=2, lat_bounds=(20.0, 80.0))
    # at least one feature is tracked across several steps (not all singletons)
    assert max(len(t.steps) for t in tracks) >= 5
