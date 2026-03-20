import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def simple_wave_field():
    """A synthetic v-wind with 3 clear crests and 2 troughs.

    The field is:  v(lon, lat) = A(lon) * sin(k * lon)
    where A(lon) is a Gaussian envelope centered at lon=180
    and k gives ~4 full wavelengths across 360 degrees.

    Latitudes span 20N to 80N. Longitudes span 0 to 359.
    """
    lons = np.arange(0, 360, 2.5)  # 144 points
    lats = np.arange(20, 80.1, 2.5)  # 25 points
    lon2d, lat2d = np.meshgrid(lons, lats)

    k = 2 * np.pi * 4 / 360  # wavenumber 4
    envelope = (
        30
        * np.exp(-((lon2d - 202.5) ** 2) / (2 * 90**2))
        * np.exp(-((lat2d - 50) ** 2) / (2 * 10**2))
    )
    v = envelope * np.sin(k * lon2d)

    da = xr.DataArray(
        v,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="v",
    )
    return da


@pytest.fixture
def two_timestep_field():
    """A field with 2 timesteps where the wave packet has shifted ~5° east."""
    lons = np.arange(0, 360, 2.5)
    lats = np.arange(20, 80.1, 2.5)
    times = [0, 1]

    da_list = []
    for t in times:
        lon2d, lat2d = np.meshgrid(lons, lats)
        k = 2 * np.pi * 4 / 360
        # Shift the envelope and wave by 5 degrees per timestep
        shift = t * 5
        envelope = (
            30
            * np.exp(-((lon2d - (202.5 + shift)) ** 2) / (2 * 90**2))
            * np.exp(-((lat2d - 50) ** 2) / (2 * 10**2))
        )
        v = envelope * np.sin(k * (lon2d - shift))
        da_list.append(v)

    da = xr.DataArray(
        da_list,
        dims=["time", "latitude", "longitude"],
        coords={"time": times, "latitude": lats, "longitude": lons},
        name="v",
    )
    return da


@pytest.fixture
def single_maximum_field():
    """A single isolated Gaussian bump (one clear maximum)."""
    lons = np.arange(0, 360, 2.5)
    lats = np.arange(20, 80.1, 2.5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    v = 30 * np.exp(-((lon2d - 180) ** 2 + (lat2d - 50) ** 2) / (2 * 10**2))

    da = xr.DataArray(
        v,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="v",
    )
    return da


@pytest.fixture
def flat_field():
    """A field that is identically zero everywhere."""
    lons = np.arange(0, 360, 2.5)
    lats = np.arange(20, 80.1, 2.5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    v = np.zeros_like(lon2d)

    da = xr.DataArray(
        v,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="v",
    )
    return da


@pytest.fixture
def date_line_wave_field():
    """Wave packet straddling the 0°/360° boundary."""
    lons = np.arange(0, 360, 2.5)
    lats = np.arange(20, 80.1, 2.5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    k = 2 * np.pi * 4 / 360
    # Envelope centered at lon=0 (wraps around to 360)
    envelope1 = (
        30
        * np.exp(-((lon2d) ** 2) / (2 * 40**2))
        * np.exp(-((lat2d - 50) ** 2) / (2 * 10**2))
    )
    envelope2 = (
        30
        * np.exp(-((lon2d - 360) ** 2) / (2 * 40**2))
        * np.exp(-((lat2d - 50) ** 2) / (2 * 10**2))
    )
    envelope = np.maximum(envelope1, envelope2)
    v = envelope * np.sin(k * lon2d)

    da = xr.DataArray(
        v,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="v",
    )
    return da


@pytest.fixture
def southern_hemisphere_wave_field():
    """A synthetic wave field in the Southern Hemisphere (lats -80 to -20)."""
    lons = np.arange(0, 360, 2.5)
    lats = np.arange(-80, -19.9, 2.5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    k = 2 * np.pi * 4 / 360
    envelope = (
        30
        * np.exp(-((lon2d - 202.5) ** 2) / (2 * 90**2))
        * np.exp(-((lat2d + 50) ** 2) / (2 * 10**2))
    )
    v = envelope * np.sin(k * lon2d)

    return xr.DataArray(
        v,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="v",
    )
