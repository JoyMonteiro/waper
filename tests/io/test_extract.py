from waper.io.extract import (
    extract_nodes, extract_edges, extract_rwps, extract_samples,
    extract_track_nodes, extract_track_edges
)
from shapely import wkb

def test_extract_nodes(tracked_waper):
    df = extract_nodes(tracked_waper)
    assert set(["time","rwp_id","node_id","node_type","lon","lat","scalar",
                "cluster_id","region_id"]).issubset(df.columns)
    assert len(df) > 0
    assert set(df["node_type"].unique()).issubset({"max","min"})

def test_extract_edges(tracked_waper):
    df = extract_edges(tracked_waper)
    assert set(["time","rwp_id","src_node_id","dst_node_id"]).issubset(df.columns)
    assert len(df) > 0

def test_extract_rwps(tracked_waper):
    df = extract_rwps(tracked_waper)
    assert set(["time","rwp_id","weighted_lon","weighted_lat","peak_amp",
                "n_nodes","zonal_extent_deg","geometry_wkb"]).issubset(df.columns)
    assert (df["peak_amp"] > 0).all()
    geom = wkb.loads(df["geometry_wkb"].iloc[0])   # round-trips to a shapely geometry
    assert geom.geom_type in ("Polygon","MultiPolygon")

def test_extract_samples(tracked_waper):
    df = extract_samples(tracked_waper)
    assert set(["time","rwp_id","pt_idx","lon","lat"]).issubset(df.columns)
    assert len(df) > 0

def test_extract_track_tables(tracked_waper):
    nodes = extract_track_nodes(tracked_waper)
    edges = extract_track_edges(tracked_waper)
    assert set(["time","feature","lon","lat","key"]).issubset(nodes.columns)
    assert set(["src","dst","time_from","feat_from","time_to","feat_to",
                "weight","distance"]).issubset(edges.columns)
    # keys are the f"{time}:{feature}" strings used to rebuild the DiGraph
    assert edges["src"].iloc[0] == f'{edges["time_from"].iloc[0]}:{edges["feat_from"].iloc[0]}'
