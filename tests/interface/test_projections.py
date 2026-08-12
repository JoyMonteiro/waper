import cartopy.crs as ccrs
import matplotlib
import pytest
import xarray as xr

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from waper.interface.api import Waper, WaperConfig
from waper.interface.projections import POLYGON_CRS, default_extent, default_projection


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_default_projection_follows_the_hemisphere():
    assert default_projection("north").proj4_params["lat_0"] == 90
    assert default_projection("south").proj4_params["lat_0"] == -90


def test_default_extent_follows_the_hemisphere():
    assert default_extent("north") == [-180, 180, 20, 90]
    assert default_extent("south") == [-180, 180, -90, -20]


def test_polygon_crs_is_northern_stereographic_regardless_of_hemisphere():
    # RWP polygons and rasters are built in a fixed stereographic CRS;
    # the display projection is a separate concern.
    assert POLYGON_CRS.proj4_params["lat_0"] == 90
    assert default_projection("south").proj4_params["lat_0"] == -90
    assert default_projection("south") != POLYGON_CRS


def _sh_waper(southern_hemisphere_wave_field):
    config = WaperConfig(
        debug=False, scalar_name="v", latitude_label="latitude",
        longitude_label="longitude", time_label="time", clip_value=2,
        extrema_threshold=10, max_latitude=-20, min_latitude=-80,
        node_pruning_threshold=20, edge_pruning_threshold=3e-5,
        max_edge_weight=1, hemisphere="south",
    )
    # The shared fixture is a single 2-D timestep (latitude, longitude); `Waper`
    # indexes along a time axis, so give it one.
    field = southern_hemisphere_wave_field.expand_dims(time=[0])
    ds = xr.Dataset({"v": field})
    waper = Waper.from_config(ds, config)
    waper.identify_rwps()
    return waper


def test_southern_hemisphere_run_plots_in_the_southern_hemisphere(
    southern_hemisphere_wave_field,
):
    waper = _sh_waper(southern_hemisphere_wave_field)

    ax = waper.plot_rwp_polygons(0)

    assert ax.projection.proj4_params["lat_0"] == -90
    _, _, lat_lo, lat_hi = ax.get_extent(crs=ccrs.PlateCarree())
    assert lat_lo < -20 and lat_hi <= 0


def test_caller_can_override_the_display_projection(southern_hemisphere_wave_field):
    waper = _sh_waper(southern_hemisphere_wave_field)
    orthographic = ccrs.Orthographic(central_longitude=75, central_latitude=25)

    ax = waper.plot_rwp_polygons(0, projection=orthographic)

    assert ax.projection == orthographic


def test_every_plot_method_accepts_projection():
    # Every plot entry point takes the same keyword, so a caller can switch
    # the whole figure over consistently.
    import inspect

    for name in [
        "plot_clusters", "plot_association_graph", "plot_pruned_graph",
        "plot_rwp_graphs", "plot_rwp_polygons", "plot_raster", "plot_tracks",
        "plot_track_polygons", "plot_track_rwps",
    ]:
        params = inspect.signature(getattr(Waper, name)).parameters
        assert "projection" in params, f"{name} has no projection argument"
