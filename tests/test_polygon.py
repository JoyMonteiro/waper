import numpy as np
import pytest
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

from waper.tracking.rwp_polygon import (
    get_polygon_for_rwp_path,
    transform_to_stereographic,
)
from waper.interface.api import WaperConfig, _identify_rwps


@pytest.fixture
def default_config():
    return WaperConfig(
        debug=False,
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
    )


def test_per_node_hull_not_single_convex(simple_wave_field, default_config):
    """Per-node hull must not be a single convex hull over all points."""
    ts_data = _identify_rwps(simple_wave_field, default_config)
    assert len(ts_data.rwp_info) > 0, "No RWPs identified — check the wave field fixture"

    for path_key, rwp_info in ts_data.rwp_info.items():
        poly = rwp_info["polygon"]
        assert not poly.equals(poly.convex_hull), (
            "Polygon is convex — per-node union was not applied"
        )


def test_per_node_hull_smaller_than_convex(simple_wave_field, default_config):
    """Per-node hull area must be strictly smaller than single convex hull."""
    ts_data = _identify_rwps(simple_wave_field, default_config)
    assert len(ts_data.rwp_info) > 0, "No RWPs identified — check the wave field fixture"

    for path_key, rwp_info in ts_data.rwp_info.items():
        poly = rwp_info["polygon"]
        assert poly.area < poly.convex_hull.area
