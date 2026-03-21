import numpy as np
import pytest
import xarray as xr

from waper.identification import max_min, topology


def _create_and_process_field(v, lons, lats, threshold=5, max_eps_km=1500, xi=0.05, penalty_length_scale_km=2000.0):
    da = xr.DataArray(
        v,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="v",
    )
    data_with_max = max_min.add_maxima_data(da, "v", lons, lats)
    clipped = max_min.clip_dataset(data_with_max, "v", threshold=threshold)
    connectivity = topology.identify_connected_regions(clipped)
    maxima_points = max_min.extract_maxima_points(connectivity, threshold, "v")
    clustered = topology.cluster_extrema(
        data_with_max, connectivity, maxima_points, "v",
        sign=1, max_eps_km=max_eps_km, xi=xi,
        penalty_length_scale_km=penalty_length_scale_km,
    )
    return clustered


def test_single_extremum_per_region_is_own_cluster():
    lons = np.arange(0, 360, 5)
    lats = np.arange(20, 80.1, 5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # One isolated maximum
    v = 30 * np.exp(-((lon2d - 180) ** 2 + (lat2d - 50) ** 2) / 100)

    clustered = _create_and_process_field(v, lons, lats, threshold=10)
    assert clustered.n_points == 1
    assert "Cluster ID" in clustered.point_data
    assert clustered.point_data["Cluster ID"][0] == 0


def test_two_close_extrema_same_cluster():
    lons = np.arange(0, 360, 5)
    lats = np.arange(20, 80.1, 5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Two maxima close to each other (5 degrees apart)
    v1 = 30 * np.exp(-((lon2d - 180) ** 2 + (lat2d - 50) ** 2) / 20)
    v2 = 30 * np.exp(-((lon2d - 185) ** 2 + (lat2d - 50) ** 2) / 20)
    v = v1 + v2

    clustered = _create_and_process_field(v, lons, lats, threshold=10)
    assert clustered.n_points == 2
    # They should be in the same cluster
    assert (
        clustered.point_data["Cluster ID"][0] == clustered.point_data["Cluster ID"][1]
    )


def test_two_distant_extrema_different_clusters():
    lons = np.arange(0, 360, 5)
    lats = np.arange(20, 80.1, 5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Two maxima far apart (60 degrees), but make sure they are connected so they end up in same region initially
    # Actually, if they are in different regions, they get different clusters anyway.
    # To test clustering algorithm itself, they should be in the SAME connected region.
    # A broad base with two peaks.
    base = 15 * np.exp(-((lon2d - 210) ** 2 + (lat2d - 50) ** 2) / 5000)  # very broad
    v1 = 30 * np.exp(-((lon2d - 180) ** 2 + (lat2d - 50) ** 2) / 50)
    v2 = 30 * np.exp(-((lon2d - 240) ** 2 + (lat2d - 50) ** 2) / 50)
    v = base + v1 + v2

    clustered = _create_and_process_field(v, lons, lats, threshold=5)
    # At least two maxima
    assert clustered.n_points >= 2
    # OPTICS labels distant points as noise, which get reassigned as
    # singleton clusters — so they end up in different clusters.
    clusters = np.unique(clustered.point_data["Cluster ID"])
    assert len(clusters) > 1


def test_centroid_representative():
    lons = np.arange(0, 360, 5)
    lats = np.arange(20, 80.1, 5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Asymmetric cluster connected by a broad base
    base = 15 * np.exp(-((lon2d - 185) ** 2 + (lat2d - 50) ** 2) / 500)
    # Main peak at lon=180, lat=50, value=30
    v1 = 30 * np.exp(-((lon2d - 180) ** 2 + (lat2d - 50) ** 2) / 10)
    # Secondary peak at lon=190, lat=50, value=20
    v2 = 20 * np.exp(-((lon2d - 190) ** 2 + (lat2d - 50) ** 2) / 10)
    
    # They will fuse into one cluster (eps=1000km covers 10 deg at lat=50)
    # The base ensures they are in the same connected region > threshold (5)
    v = base + v1 + v2

    clustered = _create_and_process_field(v, lons, lats, threshold=5, max_eps_km=1500)
    
    # Check max_cluster_assign
    (
        cluster_max_arr,
        cluster_max_point,
        max_pt_dict,
        num_max_clusters,
    ) = topology.max_cluster_assign(clustered, "v")

    assert num_max_clusters == 1
    
    # The absolute peak is at lon=180, but the centroid should be pulled towards 190
    centroid_lon = cluster_max_point[0][0]
    
    # It must be strictly greater than 180.0
    assert centroid_lon > 180.0
    # It must be strictly less than 190.0
    assert centroid_lon < 190.0
    
    # The max value should still be strictly tracked as the peak (around 30)
    assert cluster_max_arr[0] >= 30.0
def test_isolated_outlier_far_from_group():
    lons = np.arange(0, 360, 2.5)
    lats = np.arange(20, 80.1, 2.5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Broad base to connect them
    base = 15 * np.exp(-((lon2d - 200) ** 2 + (lat2d - 50) ** 2) / 5000)

    # 5 tight maxima
    v = base.copy()
    for offset in [-10, -5, 0, 5, 10]:
        v += 20 * np.exp(-((lon2d - (180 + offset)) ** 2 + (lat2d - 50) ** 2) / 10)

    # 1 outlier maximum far away
    v += 20 * np.exp(-((lon2d - 240) ** 2 + (lat2d - 50) ** 2) / 10)

    clustered = _create_and_process_field(v, lons, lats, threshold=10)

    clusters = np.unique(clustered.point_data["Cluster ID"])
    
    # No noise labels should remain — noise points are reassigned as singleton clusters
    assert -1 not in clusters

    # The outlier should be in its own cluster, separate from the main group
    assert len(clusters) > 1
