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
