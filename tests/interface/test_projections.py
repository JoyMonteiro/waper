import cartopy.crs as ccrs
import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from waper.interface.api import Waper, WaperConfig
from waper.interface.projections import default_extent, default_projection


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


def _drawn_polygon_latitudes(ax):
    """Latitudes of every polygon vertex *as drawn*, read back through its own artist.

    `patch.get_transform()` is the CRS the polygon was handed to matplotlib in,
    chained to the axes' data transform. Subtracting `transData` leaves the
    data-CRS-to-display-projection step, so this measures the `transform=`
    argument the plotting code actually chose — not the vertices it started from,
    which are the same numbers whichever CRS you claim they are in.
    """
    lats = []
    for patch in ax.patches:
        verts = np.asarray(patch.get_xy())
        projected = (patch.get_transform() - ax.transData).transform(verts)
        lon_lat = ccrs.PlateCarree().transform_points(
            ax.projection, projected[:, 0], projected[:, 1]
        )
        lats.append(lon_lat[:, 1])
    return np.concatenate(lats)


def test_sh_polygons_are_drawn_in_the_southern_hemisphere(
    southern_hemisphere_wave_field,
):
    # The crux: polygons of a southern run are built in a south-polar CRS. Naming
    # a north-polar one in `transform=` mirrors them into the *northern*
    # hemisphere, where the southern extent then clips them away — a blank plot,
    # the exact bug this module exists to prevent. Nothing above catches that:
    # the axes projection and extent would be right either way.
    #
    # The override is centred on the synthetic packet (lon 202.5, lat -50)
    # deliberately: an orthographic over South Asia puts this packet on the far
    # side of the globe, where every vertex projects to NaN and the readback
    # measures nothing.
    overrides = [None, ccrs.Orthographic(central_longitude=202.5, central_latitude=-50)]
    measured = []

    for projection in overrides:
        waper = _sh_waper(southern_hemisphere_wave_field)
        ax = waper.plot_rwp_polygons(0, projection=projection)

        lats = _drawn_polygon_latitudes(ax)
        assert len(lats) > 0, f"no polygon was drawn for projection={projection}"
        assert np.all(lats < 0), (
            f"projection={projection} drew polygon vertices at latitudes "
            f"{np.nanmin(lats):.2f}..{np.nanmax(lats):.2f}; they belong in the "
            "southern hemisphere"
        )
        measured.append(lats)
        plt.close("all")

    # Changing the display projection must not move the data.
    assert np.allclose(measured[0], measured[1])


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
