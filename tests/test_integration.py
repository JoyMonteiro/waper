import pytest
import xarray as xr

from waper.interface.api import Waper


def test_full_pipeline_synthetic(two_timestep_field):
    # Waper expects a dataset with time, lat, lon
    ds = xr.Dataset({"v": two_timestep_field})
    waper_obj = Waper(
        data_array=ds,
        scalar_name="v",
        latitude_label="latitude",
        longitude_label="longitude",
        time_label="time",
        clip_value=2,
        extrema_threshold=10,
        max_latitude=80.1,
        min_latitude=20,
        node_pruning_threshold=15,
        edge_pruning_threshold=3e-5,
        track_pruning_threshold=0.3,
        max_edge_weight=1,
        debug=False,
    )

    waper_obj.identify_rwps()

    # Assert rwps were found in at least the first timestep
    assert len(waper_obj._time_step_data[0].identified_rwp_paths) > 0

    # Track the rwps
    waper_obj.track_rwps()

    # Assert tracking graph was built and has edges
    assert waper_obj._tracking_graph is not None
    assert len(waper_obj._tracking_graph.edges) > 0


def test_full_pipeline_flat_field_graceful(flat_field):
    import numpy as np
    import xarray as xr

    # Make a 2 timestep flat field
    times = [0, 1]
    da_list = [flat_field.values, flat_field.values]

    da = xr.DataArray(
        da_list,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": times,
            "latitude": flat_field["latitude"],
            "longitude": flat_field["longitude"],
        },
        name="v",
    )

    ds = xr.Dataset({"v": da})

    waper_obj = Waper(
        data_array=ds,
        scalar_name="v",
        latitude_label="latitude",
        longitude_label="longitude",
        time_label="time",
        clip_value=2,
        extrema_threshold=10,
        max_latitude=80.1,
        min_latitude=20,
        node_pruning_threshold=15,
        edge_pruning_threshold=3e-5,
        track_pruning_threshold=0.3,
        max_edge_weight=1,
        debug=False,
    )

    waper_obj.identify_rwps()

    # No rwps should be found
    for ts in waper_obj._time_step_data:
        assert len(ts.identified_rwp_paths) == 0

    waper_obj.track_rwps()

    # Tracking graph should be empty
    assert len(waper_obj._tracking_graph.edges) == 0
