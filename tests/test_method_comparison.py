import os

import pytest
import numpy as np
import networkx as nx
from scripts.method_comparison.metrics import (
    iou, disagreement_decomposition, detection_agreement,
)
from scripts.method_comparison.masks import (
    pixel_lonlat_grid, band_mask, compute_rwp_envelope,
    zimin_mask, edge_pruning_mask, node_amplitude_mask,
    t21_truncate, temporal_running_mean,
)
from scripts.method_comparison.run_sweep import load_dataset, run_base_waper
from scripts.method_comparison.run_sweep import compute_zimin_masks, sweep
from scripts.method_comparison.run_sweep import DATA_PATH

# The slow tests below read DATA_PATH, which is gitignored and absent on a fresh clone.
needs_data = pytest.mark.skipif(
    not os.path.exists(DATA_PATH), reason=f"{DATA_PATH} not present"
)


def test_iou_disjoint_is_zero():
    a = np.zeros((4, 4), bool); a[0, 0] = True
    b = np.zeros((4, 4), bool); b[3, 3] = True
    assert iou(a, b) == 0.0


def test_iou_identical_is_one():
    a = np.zeros((4, 4), bool); a[1:3, 1:3] = True
    assert iou(a, a.copy()) == 1.0


def test_iou_both_empty_is_one():
    a = np.zeros((4, 4), bool)
    assert iou(a, a.copy()) == 1.0


def test_iou_contained():
    a = np.zeros((4, 4), bool); a[0:2, 0:2] = True   # 4 cells
    b = np.zeros((4, 4), bool); b[0, 0] = True        # 1 cell, subset of a
    # intersection 1, union 4
    assert iou(a, b) == 0.25


def test_disagreement_decomposition():
    band = np.ones((4, 4), bool)            # 16 cells
    method = np.zeros((4, 4), bool); method[0, :] = True   # 4 cells
    ref = np.zeros((4, 4), bool); ref[0, 0] = True; ref[3, 3] = True  # 2 cells
    # method_only = method & ~ref within band = 3 cells -> 3/16
    # ref_only = ref & ~method within band = 1 cell (3,3) -> 1/16
    m_only, r_only = disagreement_decomposition(method, ref, band)
    assert abs(m_only - 3 / 16) < 1e-9
    assert abs(r_only - 1 / 16) < 1e-9


def test_detection_agreement():
    empty = np.zeros((4, 4), bool)
    nonempty = np.zeros((4, 4), bool); nonempty[0, 0] = True
    assert detection_agreement(nonempty, nonempty.copy()) is True
    assert detection_agreement(empty, empty.copy()) is True
    assert detection_agreement(nonempty, empty) is False


def test_pixel_lonlat_grid_shapes_and_ranges():
    lon, lat = pixel_lonlat_grid("north")
    assert lon.shape == (512, 512)
    assert lat.shape == (512, 512)
    # NH stereographic grid: latitudes span roughly 0..90 in the disc, NaN/<0 in corners
    finite = np.isfinite(lat)
    assert lat[finite].max() > 85.0
    assert (lon[finite] >= 0).all() and (lon[finite] <= 360).all()


def test_band_mask_excludes_outside():
    bm = band_mask(20.0, 80.0, "north")
    lon, lat = pixel_lonlat_grid("north")
    inside = bm
    # every True pixel must have latitude in [20, 80]
    assert (lat[inside] >= 20.0).all()
    assert (lat[inside] <= 80.0).all()
    assert bm.sum() > 0


def test_compute_rwp_envelope_recovers_modulation():
    # v(x) = A(x) * cos(k x), A slowly varying, k inside the 3-11 band
    nlon = 360
    x = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
    A = 10.0 + 5.0 * np.cos(x)              # wavenumber-1 modulation (outside band)
    carrier = np.cos(7 * x)                 # wavenumber 7 (inside band)
    v = (A * carrier)[None, :]              # shape (1, nlon)
    env = compute_rwp_envelope(v, (3, 11))
    # envelope should track A(x) away from the wrap edges
    interior = slice(30, nlon - 30)
    rel_err = np.abs(env[0, interior] - A[interior]) / A[interior]
    assert rel_err.max() < 0.15


class _FakeTSD:
    def __init__(self, raster):
        self.raster_data = raster


def test_edge_pruning_mask_none_raster_is_empty():
    bm = band_mask()
    m = edge_pruning_mask(_FakeTSD(None), bm)
    assert m.shape == (512, 512)
    assert m.sum() == 0


def test_edge_pruning_mask_thresholds_and_bands():
    bm = band_mask()
    raster = np.zeros((512, 512), dtype=np.int32)
    raster[bm] = 1            # label inside band
    raster[~bm] = 2           # label outside band (must be dropped)
    m = edge_pruning_mask(_FakeTSD(raster), bm)
    assert m.sum() == bm.sum()
    assert not m[~bm].any()


def test_zimin_mask_thresholds_within_band():
    bm = band_mask()
    ds_lon = np.arange(0, 360, 1.0)
    ds_lat = np.arange(0, 91, 1.0)          # NH 1-deg
    env = np.zeros((ds_lat.size, ds_lon.size))
    # strong envelope only at 50N, 100E -> should appear; a strong patch at 5N should not
    env[50, 100] = 30.0
    env[5, 100] = 30.0
    m = zimin_mask(env, ds_lon, ds_lat, bm, threshold=14.0)
    assert m.shape == (512, 512)
    assert m.sum() > 0
    _, plat = pixel_lonlat_grid("north")
    assert (plat[m] >= 20.0).all() and (plat[m] <= 80.0).all()


def test_t21_truncate_smooths_and_preserves_shape():
    # t21_truncate imports spharm (pyspharm) lazily. It has no wheels and needs a
    # Fortran toolchain to build, so it is not a declared dependency — skip rather
    # than fail where it is absent, as with datashader in tests/interface/.
    pytest.importorskip("spharm", exc_type=ImportError)
    # large-scale (wavenumber 6) + small-scale (wavenumber 40) on an ascending NH grid
    nlat, nlon = 90, 360
    lat = np.linspace(0.5, 89.5, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    LON, LAT = np.meshgrid(lon, lat)
    big = np.cos(np.deg2rad(LAT)) * np.sin(np.deg2rad(6 * LON))
    small = 0.5 * np.sin(np.deg2rad(40 * LON))
    field = big + small
    out = t21_truncate(field, ntrunc=21)
    assert out.shape == field.shape
    # T21 removes the wavenumber-40 component -> closer to the large-scale part
    assert np.std(out - big) < np.std(field - big)


def test_temporal_running_mean_smears_spike_and_infers_step():
    import numpy as np
    nt = 48
    stack = np.zeros((nt, 4, 4))
    stack[24] = 10.0  # a one-hour spike
    times = np.datetime64("2011-04-01T00") + np.arange(nt) * np.timedelta64(1, "h")
    out = temporal_running_mean(stack, 24, times)
    assert out.shape == stack.shape
    assert out[24].max() < 10.0          # spike smeared by the 24-step window
    assert out[24].max() > 0.0           # but spread into the window
    # window <= 1 step is a no-op
    same = temporal_running_mean(stack, 0, times)
    assert np.array_equal(same, stack)


def test_node_amplitude_mask_keeps_only_strong_nodes():
    bm = band_mask()
    g = nx.Graph()
    # one strong cluster near 50N/100E, one weak cluster near 50N/200E
    g.add_node(("max", 0), scalar=30.0,
               cluster_extrema=[((99.0, 49.0), 0, 30.0), ((101.0, 51.0), 0, 28.0),
                                ((100.0, 50.0), 0, 29.0)])
    g.add_node(("max", 1), scalar=8.0,
               cluster_extrema=[((199.0, 49.0), 1, 8.0), ((201.0, 51.0), 1, 7.0),
                                ((200.0, 50.0), 1, 6.0)])
    m = node_amplitude_mask(g, st=20.0, band=bm)
    assert m.shape == (512, 512)
    assert m.sum() > 0
    # nothing should be burned near 200E (weak node dropped); check via lon grid
    plon, plat = pixel_lonlat_grid("north")
    near_weak = m & (np.abs(plon - 200.0) < 5.0) & (np.abs(plat - 50.0) < 5.0)
    assert near_weak.sum() == 0


@pytest.mark.slow
@needs_data
def test_load_dataset_shape():
    v = load_dataset()
    assert set(v.dims) == {"time", "latitude", "longitude"}
    assert float(v.longitude.min()) >= 0.0 and float(v.longitude.max()) < 360.0
    # 1-degree coarsened NH
    assert v.latitude.size <= 91 + 1
    assert v.time.size == 720


@pytest.mark.slow
@needs_data
def test_run_base_waper_has_association_graphs():
    v = load_dataset().isel(time=slice(0, 2))
    w = run_base_waper(v)
    assert len(w._time_step_data) == 2
    assert w._time_step_data[0].association_graph is not None


@pytest.mark.slow
@needs_data
def test_sweep_smoke_two_timesteps():
    v = load_dataset().isel(time=slice(0, 2))
    df = sweep(v, gt_grid=[0.02], st_grid=[20])
    assert set(df.columns) == {
        "method", "threshold", "mean_iou", "detection_agreement",
        "mean_method_only_frac", "mean_zimin_only_frac", "n_timesteps",
    }
    assert set(df["method"]) == {"edge_pruning", "node_amplitude"}
    assert (df["n_timesteps"] == 2).all()
    assert (df["mean_iou"] >= 0).all() and (df["mean_iou"] <= 1).all()
