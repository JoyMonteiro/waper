"""Timing baselines for identification and tracking.

Marked `slow`: CI runs `pytest -m "not slow"` and skips these. Run them
deliberately with `pytest -m slow tests/test_benchmark.py -s` and record the
numbers in results/benchmarks.md.
"""

import time

import numpy as np
import pytest
import xarray as xr

from waper.interface.api import Waper, WaperConfig


def _synthetic_field(n_time, n_lat, n_lon):
    """A zonally-wavenumber-6 field with a latitudinally-confined envelope.

    The amplitude is 30, not the `extrema_threshold` of 10 it has to clear: at
    amplitude 20 the envelope-modulated peaks sit close enough to the threshold
    that a 10-degree phase advance drops every packet after the first timestep,
    and the tracking benchmark then times an empty graph.
    """
    lats = np.linspace(20, 80, n_lat)
    lons = np.linspace(0, 360, n_lon, endpoint=False)
    times = np.arange(n_time)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    envelope = np.exp(-(((lat_grid - 50) / 12.0) ** 2))
    frames = [
        30 * envelope * np.sin(np.deg2rad(6 * lon_grid + 10 * t))
        for t in times
    ]

    return xr.DataArray(
        np.stack(frames),
        dims=["time", "latitude", "longitude"],
        coords={"time": times, "latitude": lats, "longitude": lons},
        name="v",
    )


def _config():
    return WaperConfig(
        debug=False, scalar_name="v", latitude_label="latitude",
        longitude_label="longitude", time_label="time", clip_value=2,
        extrema_threshold=10, max_latitude=80.1, min_latitude=20,
        node_pruning_threshold=20, edge_pruning_threshold=3e-5,
        max_edge_weight=1,
    )


@pytest.mark.slow
def test_identification_benchmark_1p5_degree():
    # 121 x 240 — a 1.5-degree global grid, one timestep.
    field = _synthetic_field(1, 121, 240)
    waper = Waper.from_config(xr.Dataset({"v": field}), _config())

    start = time.perf_counter()
    waper.identify_rwps()
    elapsed = time.perf_counter() - start

    print(f"\nidentification, 1 timestep @ 121x240: {elapsed:.2f}s")
    assert waper._time_step_data[0].identified_rwp_paths, "benchmarked an empty field"
    assert elapsed < 120, "an order of magnitude slower than the recorded baseline"


@pytest.mark.slow
def test_tracking_benchmark_10_timesteps():
    field = _synthetic_field(10, 121, 240)
    waper = Waper.from_config(xr.Dataset({"v": field}), _config())
    waper.identify_rwps()

    start = time.perf_counter()
    waper.track_rwps()
    elapsed = time.perf_counter() - start

    print(f"\ntracking, 10 timesteps @ 121x240: {elapsed:.2f}s")
    # A tracking graph with no edges takes no time to build; timing one would
    # record a baseline that measures nothing.
    assert waper._tracking_graph.number_of_edges() > 0, "benchmarked an empty graph"
    assert elapsed < 60, "an order of magnitude slower than the recorded baseline"
