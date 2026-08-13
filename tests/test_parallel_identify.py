import pickle
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from scripts.method_comparison.run_sweep import DATA_PATH, load_dataset
from waper.interface.api import Waper, WaperConfig, _identify_rwps


def _config(**overrides):
    base = {
        "debug": False, "scalar_name": "v", "latitude_label": "latitude",
        "longitude_label": "longitude", "time_label": "time", "clip_value": 2,
        "extrema_threshold": 10, "max_latitude": 80.1, "min_latitude": 20,
        "node_pruning_threshold": 20, "edge_pruning_threshold": 3e-5,
        "max_edge_weight": 1,
    }
    base.update(overrides)
    return WaperConfig(**base)


def test_config_pickles():
    config = _config()
    assert pickle.loads(pickle.dumps(config)).scalar_name == "v"


def test_single_timestep_result_survives_a_process_boundary(two_timestep_field):
    # Everything a worker returns has to pickle: two PolyData, two networkx
    # Graphs, a DataArray and two ndarrays.
    result = _identify_rwps(two_timestep_field[0], _config())

    restored = pickle.loads(pickle.dumps(result))

    assert restored.vtk_data.n_points == result.vtk_data.n_points
    assert len(restored.identified_rwp_paths) == len(result.identified_rwp_paths)
    assert restored.association_graph.number_of_edges() == \
        result.association_graph.number_of_edges()
    np.testing.assert_array_equal(restored.raster_data, result.raster_data)


def _paths_signature(waper):
    """A comparable summary of what identification produced."""
    return [
        [sorted(map(str, path)) for path in ts.identified_rwp_paths]
        for ts in waper._time_step_data
    ]


def test_parallel_identification_matches_sequential(two_timestep_field):
    ds = xr.Dataset({"v": two_timestep_field})

    sequential = Waper.from_config(ds, _config())
    sequential.identify_rwps()

    parallel = Waper.from_config(ds, _config())
    parallel.identify_rwps(n_jobs=2)

    assert _paths_signature(parallel) == _paths_signature(sequential)
    for par_ts, seq_ts in zip(
        parallel._time_step_data, sequential._time_step_data, strict=True
    ):
        np.testing.assert_array_equal(par_ts.raster_data, seq_ts.raster_data)


def test_timesteps_come_back_in_order(two_timestep_field):
    # A worker pool completes out of order; _time_step_data is indexed by
    # timestep everywhere downstream, so ordering is load-bearing.
    ds = xr.Dataset({"v": two_timestep_field})
    waper = Waper.from_config(ds, _config())

    waper.identify_rwps(n_jobs=2)

    times = [ts.input_data["time"].item() for ts in waper._time_step_data]
    assert times == sorted(times)


def test_n_jobs_1_is_the_sequential_path(two_timestep_field):
    waper = Waper.from_config(xr.Dataset({"v": two_timestep_field}), _config())

    waper.identify_rwps(n_jobs=1)

    assert len(waper._time_step_data) == 2


@pytest.mark.slow
@pytest.mark.skipif(not Path(DATA_PATH).exists(), reason="652 MB input is gitignored")
def test_parallel_matches_sequential_on_real_data():
    # `load_dataset` is the existing idiom for this file: the raw netCDF names
    # its time axis `valid_time` and carries a singleton `pressure_level`, so
    # a bare `open_dataset` would not match the config's labels.
    da = load_dataset().isel(time=slice(0, 4))
    ds = da.to_dataset(name="v")
    config = _config(min_latitude=20, max_latitude=80)

    sequential = Waper.from_config(ds, config)
    sequential.identify_rwps()

    parallel = Waper.from_config(ds, config)
    parallel.identify_rwps(n_jobs=-1)

    assert _paths_signature(parallel) == _paths_signature(sequential)
    for par_ts, seq_ts in zip(
        parallel._time_step_data, sequential._time_step_data, strict=True
    ):
        np.testing.assert_array_equal(par_ts.raster_data, seq_ts.raster_data)
        np.testing.assert_array_equal(par_ts.energy_raster, seq_ts.energy_raster)
