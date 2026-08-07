import holoviews as hv
import pytest
import xarray as xr

from waper.interface import explorer


def test_layer_builders_return_elements(cat):
    hv.extension("bokeh")
    nodes = explorer.nodes_layer(cat, time=0)
    polys = explorer.polygons_layer(cat, time=0)
    assert isinstance(nodes, hv.Points)
    assert isinstance(polys, hv.Polygons)

def test_explorer_renders(cat):
    hv.extension("bokeh")
    app = explorer.RWPExplorer(cat, n_times=2)
    hv.render(app._map.object)

def test_layer_toggle(cat):
    hv.extension("bokeh")
    app = explorer.RWPExplorer(cat, n_times=2)
    app.layers = ["polygons"]
    hv.render(app._map.object)

def test_explorer_with_field(cat, two_timestep_field):
    # The field layer renders with rasterize=True, which needs the
    # datashader/numba stack. Skip rather than fail where that stack cannot be
    # imported (numba caps numpy at 2.4 — see the numpy pin in pyproject.toml).
    pytest.importorskip("datashader", exc_type=ImportError)
    hv.extension("bokeh")
    field_da = xr.Dataset({"v": two_timestep_field})["v"]
    app = explorer.RWPExplorer(cat, n_times=2, field_da=field_da)
    hv.render(app._map.object)
    # Check that Hovmoller renders too
    hv.render(app._hovmoeller.object)
