import pytest, xarray as xr
from waper.interface.api import Waper

@pytest.fixture
def tracked_waper(two_timestep_field):
    ds = xr.Dataset({"v": two_timestep_field})
    w = Waper(data_array=ds, scalar_name="v",
              latitude_label="latitude", longitude_label="longitude", time_label="time",
              clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80.1,
              node_pruning_threshold=15, edge_pruning_threshold=3e-5,
              track_pruning_threshold=0.3, max_edge_weight=1, debug=False)
    w.identify_rwps()
    w.track_rwps()
    return w

@pytest.fixture
def cat(tracked_waper, tmp_path):
    from waper.io.catalogue import save_catalogue, load_catalogue
    p = tmp_path / "cat"
    save_catalogue(tracked_waper, p, meta={"units": "m s**-1", "dt_hours": 6})
    return load_catalogue(p)
