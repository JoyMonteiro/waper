import numpy as np
import xarray as xr

from waper.identification import max_min


def test_finds_known_maxima(simple_wave_field):
    da = simple_wave_field
    lons = da["longitude"].values
    lats = da["latitude"].values

    data_with_max = max_min.add_maxima_data(da, "v", lons, lats)
    maxima_points = max_min.extract_maxima_points(
        data_with_max, threshold=10, scalar_name="v"
    )

    # 3 crests in simple_wave_field
    assert maxima_points.n_points == 3
    assert all(val > 0 for val in maxima_points.point_data["v"])


def test_finds_known_minima(simple_wave_field):
    da = simple_wave_field
    lons = da["longitude"].values
    lats = da["latitude"].values

    data_with_min = max_min.add_minima_data(da, "v", lons, lats)
    minima_points = max_min.extract_minima_points(
        data_with_min, threshold=-10, scalar_name="v"
    )

    # 2 troughs in simple_wave_field
    assert minima_points.n_points == 2
    assert all(val < 0 for val in minima_points.point_data["v"])


def test_threshold_filters_weak_extrema():
    lons = np.arange(0, 360, 2.5)
    lats = np.arange(20, 80.1, 2.5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # One strong max (v=30) and one weak max (v=3)
    v = 30 * np.exp(-((lon2d - 180) ** 2 + (lat2d - 50) ** 2) / 100) + 3 * np.exp(
        -((lon2d - 90) ** 2 + (lat2d - 50) ** 2) / 100
    )

    da = xr.DataArray(
        v,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="v",
    )

    data_with_max = max_min.add_maxima_data(da, "v", lons, lats)

    # Threshold=5 should only keep the strong one
    maxima_points = max_min.extract_maxima_points(
        data_with_max, threshold=5, scalar_name="v"
    )
    assert maxima_points.n_points == 1
    assert maxima_points.point_data["v"][0] > 29


def test_periodic_boundary_maxima(date_line_wave_field):
    da = date_line_wave_field
    lons = da["longitude"].values
    lats = da["latitude"].values

    data_with_max = max_min.add_maxima_data(da, "v", lons, lats)
    maxima_points = max_min.extract_maxima_points(
        data_with_max, threshold=10, scalar_name="v"
    )

    # Expected 1 maximum around date line
    assert maxima_points.n_points == 1


def test_flat_field_no_extrema(flat_field):
    da = flat_field
    lons = da["longitude"].values
    lats = da["latitude"].values

    data_with_max = max_min.add_maxima_data(da, "v", lons, lats)
    maxima_points = max_min.extract_maxima_points(
        data_with_max, threshold=1, scalar_name="v"
    )
    assert maxima_points.n_points == 0

    data_with_min = max_min.add_minima_data(da, "v", lons, lats)
    minima_points = max_min.extract_minima_points(
        data_with_min, threshold=-1, scalar_name="v"
    )
    assert minima_points.n_points == 0


def test_maxima_and_minima_do_not_overlap(simple_wave_field):
    da = simple_wave_field
    lons = da["longitude"].values
    lats = da["latitude"].values

    data_with_max = max_min.add_maxima_data(da, "v", lons, lats)
    data_with_min = max_min.add_minima_data(da, "v", lons, lats)

    is_max = data_with_max.point_data["is max"]
    is_min = data_with_min.point_data["is min"]

    overlap = np.logical_and(is_max == 1, is_min == 1)
    assert not np.any(overlap)
