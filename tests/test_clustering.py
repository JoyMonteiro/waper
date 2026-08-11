import numpy as np
import pyvista as pv
import xarray as xr
from scipy.sparse.csgraph import dijkstra

from waper.identification import max_min, topology
from waper.identification.topology import (
    _NO_PREDECESSOR,
    _path_extremes,
    _surface_graph,
)


def _two_triangles():
    """Two triangles sharing the edge 1-2, in a plane.

        0 --- 1
          \\  | \\
           \\ |  \\
             2 --- 3
    """
    points = np.array(
        [[0.0, 1.0, 0.0], [3.0, 1.0, 0.0], [0.0, -3.0, 0.0], [3.0, -3.0, 0.0]]
    )
    faces = np.hstack([[3, 0, 1, 2], [3, 1, 3, 2]])
    return pv.PolyData(points, faces)


def test_surface_graph_edge_weights_are_euclidean_distances():
    graph = _surface_graph(_two_triangles()).toarray()

    assert graph[0, 1] == 3.0  # horizontal
    assert graph[0, 2] == 4.0  # vertical
    assert graph[1, 2] == 5.0  # the shared diagonal, 3-4-5
    # Symmetric, and non-adjacent points carry no edge.
    assert np.array_equal(graph, graph.T)
    assert graph[0, 3] == 0.0


def test_surface_graph_does_not_double_the_shared_edge():
    """Edge 1-2 belongs to both triangles. Summing the duplicate on the
    coo -> csr conversion would report 10.0 instead of 5.0."""
    graph = _surface_graph(_two_triangles())

    assert graph[1, 2] == 5.0
    assert graph.nnz == 2 * 5  # 5 undirected edges, stored both ways


def test_path_extremes_matches_an_explicit_walk_up_the_tree():
    """The pointer-doubling bottleneck must agree with walking each shortest
    path vertex by vertex, which is what the VTK implementation did."""
    rng = np.random.default_rng(0)
    mesh = pv.Sphere(theta_resolution=20, phi_resolution=20).triangulate()
    values = rng.normal(size=mesh.n_points)
    graph = _surface_graph(mesh)

    source = 7
    _, predecessors = dijkstra(
        graph, directed=False, indices=[source], return_predecessors=True
    )

    for sign in (1, -1):
        fast = _path_extremes(predecessors[0], values, sign)
        for target in range(0, mesh.n_points, 13):
            path = [target]
            while path[-1] != source:
                parent = predecessors[0][path[-1]]
                assert parent != _NO_PREDECESSOR
                path.append(parent)
            expected = values[path].min() if sign > 0 else values[path].max()
            assert fast[target] == expected


def test_path_extremes_leaves_unreachable_vertices_alone():
    """Vertices in another component have no predecessor; the helper must not
    index out of bounds on scipy's -9999 sentinel."""
    values = np.array([1.0, 2.0, 3.0])
    predecessors = np.array([_NO_PREDECESSOR, 0, _NO_PREDECESSOR])

    result = _path_extremes(predecessors, values, sign=1)

    assert result[0] == 1.0  # the source
    assert result[1] == 1.0  # min over the path 0 -> 1
    assert result[2] == 3.0  # detached: its own value, and never queried


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
        connectivity, maxima_points, "v",
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

    clustered = _create_and_process_field(v, lons, lats, threshold=5, max_eps_km=1500, penalty_length_scale_km=1000.0)
    
    # Check max_cluster_assign
    (
        cluster_max_arr,
        cluster_max_point,
        _max_pt_dict,
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


def test_hill_climbing_penalty_separates_dipped_maxima():
    """Two maxima connected by a same-sign valley should be split by hill-climbing penalty."""
    lons = np.arange(0, 360, 2.5)
    lats = np.arange(20, 80.1, 2.5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    base = 10 * np.exp(-((lon2d - 190) ** 2 + (lat2d - 50) ** 2) / 1000)
    peak1 = 20 * np.exp(-((lon2d - 175) ** 2 + (lat2d - 50) ** 2) / 50)
    peak2 = 20 * np.exp(-((lon2d - 205) ** 2 + (lat2d - 50) ** 2) / 50)
    v = base + peak1 + peak2

    clustered = _create_and_process_field(
        v, lons, lats, threshold=5, max_eps_km=3000,
        penalty_length_scale_km=2000.0,
    )

    clusters = np.unique(clustered.point_data["Cluster ID"])
    # The penalty should separate them: fractional descent ~(30-6)/30 = 0.8
    # penalty = 0.8 * 2000 = 1600 km added to geodesic distance
    assert len(clusters) >= 2, (
        f"Expected >= 2 clusters (hill-climbing should separate dipped maxima), got {len(clusters)}"
    )

def test_hill_climbing_no_penalty_when_ridge_stays_high():
    """Two maxima on the same ridge (no significant dip) should stay in one cluster."""
    lons = np.arange(0, 360, 2.5)
    lats = np.arange(20, 80.1, 2.5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Two peaks close together with overlapping high-amplitude regions
    peak1 = 30 * np.exp(-((lon2d - 180) ** 2 + (lat2d - 50) ** 2) / 80)
    peak2 = 25 * np.exp(-((lon2d - 195) ** 2 + (lat2d - 50) ** 2) / 80)
    v = peak1 + peak2

    clustered = _create_and_process_field(
        v, lons, lats, threshold=10, max_eps_km=3000,
        penalty_length_scale_km=2000.0,
    )

    if clustered.n_points >= 2:
        # If two maxima are detected, they should be in the same cluster
        clusters = np.unique(clustered.point_data["Cluster ID"])
        assert len(clusters) == 1, (
            f"Expected 1 cluster (ridge stays high, no penalty), got {len(clusters)}"
        )

def _create_and_process_minima_field(v, lons, lats, threshold=5, max_eps_km=1500, xi=0.05, penalty_length_scale_km=2000.0):
    """Like _create_and_process_field but for minima (sign=-1)."""
    da = xr.DataArray(
        v, dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons}, name="v",
    )
    data_with_min = max_min.add_minima_data(da, "v", lons, lats)
    clipped = max_min.clip_dataset(data_with_min, "v", threshold=-threshold, invert=True)
    connectivity = topology.identify_connected_regions(clipped)
    minima_points = max_min.extract_minima_points(connectivity, -threshold, "v")
    clustered = topology.cluster_extrema(
        connectivity, minima_points, "v",
        sign=-1, max_eps_km=max_eps_km, xi=xi,
        penalty_length_scale_km=penalty_length_scale_km,
    )
    return clustered

def test_hill_climbing_penalty_separates_ridged_minima():
    """Two minima connected by a same-sign ridge should be split by hill-climbing penalty."""
    lons = np.arange(0, 360, 2.5)
    lats = np.arange(20, 80.1, 2.5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    base = -10 * np.exp(-((lon2d - 190) ** 2 + (lat2d - 50) ** 2) / 1000)
    trough1 = -20 * np.exp(-((lon2d - 175) ** 2 + (lat2d - 50) ** 2) / 50)
    trough2 = -20 * np.exp(-((lon2d - 205) ** 2 + (lat2d - 50) ** 2) / 50)
    v = base + trough1 + trough2

    clustered = _create_and_process_minima_field(
        v, lons, lats, threshold=5, max_eps_km=3000,
        penalty_length_scale_km=2000.0,
    )

    clusters = np.unique(clustered.point_data["Cluster ID"])
    assert len(clusters) >= 2, (
        f"Expected >= 2 clusters (hill-climbing should separate ridged minima), got {len(clusters)}"
    )
