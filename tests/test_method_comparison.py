import numpy as np
from scripts.method_comparison.metrics import (
    iou, disagreement_decomposition, detection_agreement,
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


from scripts.method_comparison.masks import (
    pixel_lonlat_grid, band_mask, compute_rwp_envelope,
)


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
