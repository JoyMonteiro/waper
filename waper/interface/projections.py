"""Display projections and map extents shared by the plotting layers.

Two distinct things live here and must not be conflated:

* :func:`polygon_crs` is the coordinate system RWP polygons and rasters are
  *built* in. The identification run's hemisphere fixes it — vertices are in
  ``+proj=stere +lat_0=90`` metres for a northern run and ``lat_0=-90`` for a
  southern one — and the numbers are meaningless read in any other CRS, so this
  is what belongs in matplotlib's ``transform=`` argument.
* :func:`default_projection` returns the CRS a map is *displayed* in. It is a
  presentation choice and callers override it freely.

For a given hemisphere the two return the same projection, which is why one
constant used to serve both. Passing ``projection=`` to a plot changes only the
second; the ``transform=`` arguments must keep following the first, or the
polygons move.
"""

import cartopy.crs as ccrs

PLATE_CARREE = ccrs.PlateCarree(central_longitude=0)


def polygon_crs(hemisphere: str) -> ccrs.Projection:
    """The CRS RWP polygons and rasters are *constructed* in. Not a display choice.

    ``waper.tracking.rwp_polygon.transform_to_stereographic`` projects packet
    vertices to ``+proj=stere +lat_0=90`` for a northern run and ``lat_0=-90``
    for a southern one, and the rasters are burned onto the same grid, so this
    follows the run's hemisphere rather than the map being drawn.

    Args:
        hemisphere: ``"north"`` or ``"south"``.

    Returns:
        The CRS the polygon vertex coordinates are expressed in.
    """
    lat0 = -90 if hemisphere == "south" else 90
    return ccrs.Stereographic(central_longitude=0, central_latitude=lat0)


def default_projection(hemisphere: str) -> ccrs.Projection:
    """Default *display* projection: polar stereographic for the hemisphere.

    Seam-free, so dateline-crossing packets stay contiguous (Web Mercator tore
    them apart). Override per call for other workflows — western disturbances
    over South Asia read better in
    ``ccrs.Orthographic(central_longitude=75, central_latitude=25)``.

    Args:
        hemisphere: ``"north"`` or ``"south"``.

    Returns:
        The projection to draw into.
    """
    lat0 = -90 if hemisphere == "south" else 90
    return ccrs.Stereographic(central_latitude=lat0, central_longitude=0)


def default_extent(hemisphere: str) -> list[float]:
    """Default map extent in PlateCarree degrees, as ``[lon_lo, lon_hi, lat_lo, lat_hi]``.

    Clips to the mid-to-high latitudes of the hemisphere the run identified in;
    the opposite pole projects to infinity in a polar projection and must be
    excluded.

    Args:
        hemisphere: ``"north"`` or ``"south"``.

    Returns:
        A 4-element extent suitable for ``ax.set_extent(..., crs=PLATE_CARREE)``.
    """
    return [-180, 180, -90, -20] if hemisphere == "south" else [-180, 180, 20, 90]
