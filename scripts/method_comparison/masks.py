"""Mask builders that reduce each RWP-identification method to a (512,512) bool
mask on the shared north-polar stereographic grid."""
import numpy as np
from scipy.signal import hilbert

from waper.tracking.rwp_polygon import (
    WAPER_IMAGE_SIZE,
    _get_raster_transform,
    transform_to_stereographic,
)


def pixel_lonlat_grid(hemisphere="north"):
    """Longitude/latitude (degrees) of every pixel centre on the 512x512 grid."""
    n = WAPER_IMAGE_SIZE
    tf = _get_raster_transform(hemisphere)
    cols, rows = np.meshgrid(np.arange(n), np.arange(n))
    # Affine maps (col, row) -> (x, y) in stereographic metres; works elementwise.
    xs, ys = tf * (cols + 0.5, rows + 0.5)
    lon, lat = transform_to_stereographic(
        np.asarray(xs), np.asarray(ys), hemisphere=hemisphere, inverse=True
    )
    lon = np.asarray(lon).reshape(n, n) % 360.0
    lat = np.asarray(lat).reshape(n, n)
    return lon, lat


def band_mask(lat_min=20.0, lat_max=80.0, hemisphere="north"):
    """Boolean (512,512) mask, True where the pixel latitude is in [lat_min, lat_max]."""
    _, lat = pixel_lonlat_grid(hemisphere)
    with np.errstate(invalid="ignore"):
        return np.isfinite(lat) & (lat >= lat_min) & (lat <= lat_max)


def compute_rwp_envelope(v, wavenumber_range=(3, 11)):
    """Zimin et al. (2003/2006) Hilbert envelope of a 2D (lat, lon) field.

    FFT along longitude, zero wavenumbers outside the band, inverse FFT to a
    band-passed real field, Hilbert transform -> analytic signal, magnitude.
    """
    v = np.asarray(v, dtype=float)
    nlon = v.shape[-1]
    F = np.fft.fft(v, axis=-1)
    k = np.abs(np.fft.fftfreq(nlon, d=1.0 / nlon))  # integer zonal wavenumbers
    lo, hi = wavenumber_range
    keep = (k >= lo) & (k <= hi)
    F_filt = F * keep
    v_band = np.fft.ifft(F_filt, axis=-1).real
    return np.abs(hilbert(v_band, axis=-1))


from scipy.interpolate import RegularGridInterpolator

from waper.tracking.rwp_polygon import rasterize_all_rwps
from waper.tracking.feature_tracks import _footprint_from_region


def _empty_mask():
    return np.zeros((WAPER_IMAGE_SIZE, WAPER_IMAGE_SIZE), dtype=bool)


def zimin_mask(envelope, ds_lon, ds_lat, band, threshold=14.0, hemisphere="north"):
    """Threshold the Hilbert envelope at `threshold`, sampled onto the shared grid."""
    plon, plat = pixel_lonlat_grid(hemisphere)
    interp = RegularGridInterpolator(
        (ds_lat, ds_lon), envelope, method="nearest",
        bounds_error=False, fill_value=0.0,
    )
    pts = np.stack([plat.ravel(), plon.ravel()], axis=-1)
    E = interp(pts).reshape(plat.shape)
    return (E >= threshold) & band


def edge_pruning_mask(time_step_data, band):
    """Boolean mask from a WAPER timestep's rasterized RWP footprints."""
    raster = time_step_data.raster_data
    if raster is None:
        return _empty_mask()
    return (np.asarray(raster) > 0) & band


def node_amplitude_mask(association_graph, st, band, hemisphere="north"):
    """Per-cluster footprints for nodes whose |scalar| >= st (no edge connection)."""
    polys = []
    for _, attr in association_graph.nodes(data=True):
        if abs(attr["scalar"]) < st:
            continue
        coords = [pt[0] for pt in attr["cluster_extrema"]]  # pt = ((lon,lat), cid, scalar)
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        geom = _footprint_from_region(lons, lats, hemisphere)
        polys.append((geom, len(polys) + 1))
    raster = rasterize_all_rwps(polys, hemisphere=hemisphere)
    if raster is None:
        return _empty_mask()
    return (np.asarray(raster) > 0) & band
