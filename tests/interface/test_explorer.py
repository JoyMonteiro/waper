import holoviews as hv
from waper.interface import explorer

def test_layer_builders_return_elements(cat):
    hv.extension("bokeh")
    nodes = explorer.nodes_layer(cat, time=0)
    polys = explorer.polygons_layer(cat, time=0)
    assert isinstance(nodes, hv.Points)
    assert isinstance(polys, hv.Polygons)
