import dataclasses

import pytest
import xarray as xr

from waper.interface.api import Waper, WaperConfig


def _config(**overrides):
    base = {
        "debug": False,
        "scalar_name": "v",
        "latitude_label": "latitude",
        "longitude_label": "longitude",
        "time_label": "time",
        "clip_value": 2,
        "extrema_threshold": 10,
        "max_latitude": 80.1,
        "min_latitude": 20,
        "node_pruning_threshold": 20,
        "edge_pruning_threshold": 3e-5,
        "max_edge_weight": 1,
    }
    base.update(overrides)
    return WaperConfig(**base)


def test_yaml_round_trip_preserves_every_field():
    config = _config(hemisphere="south", lat_gate=12.5, track_weight_threshold=0.4)

    restored = WaperConfig.from_yaml(config.to_yaml())

    # `eq=False` on the dataclass means `==` is identity; compare fields.
    assert dataclasses.asdict(restored) == dataclasses.asdict(config)


def test_to_yaml_writes_the_file_and_returns_the_text(tmp_path):
    config = _config()
    path = tmp_path / "config.yaml"

    text = config.to_yaml(path)

    assert path.read_text() == text
    assert dataclasses.asdict(WaperConfig.from_yaml(path)) == dataclasses.asdict(config)


def test_none_survives_the_round_trip():
    # track_weight_threshold=None disables the overlap gate. YAML must not
    # turn it into the string "None".
    restored = WaperConfig.from_yaml(_config(track_weight_threshold=None).to_yaml())

    assert restored.track_weight_threshold is None


def test_from_yaml_rejects_an_unknown_field(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("scalar_name: v\nnot_a_real_field: 3\n")

    with pytest.raises(TypeError, match="not_a_real_field"):
        WaperConfig.from_yaml(path)


def test_from_config_reaches_fields_the_constructor_cannot(two_timestep_field):
    # `Waper.__init__` has no `hemisphere` parameter, so this is the only way
    # to run the Southern Hemisphere pipeline through the public API.
    config = _config(hemisphere="south", min_latitude=-80, max_latitude=-20)
    ds = xr.Dataset({"v": two_timestep_field})

    waper = Waper.from_config(ds, config)

    assert waper._config.hemisphere == "south"
    assert waper._num_time_steps == 2
    assert waper._time_step_data == []


def test_from_config_runs_the_pipeline(two_timestep_field):
    config = _config()
    waper = Waper.from_config(xr.Dataset({"v": two_timestep_field}), config)

    waper.identify_rwps()

    assert len(waper._time_step_data) == 2
