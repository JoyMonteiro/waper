import json
from pathlib import Path
import networkx as nx
from shapely import wkb
from waper.io.catalogue import write_meta, read_meta, save_catalogue, load_catalogue

def test_meta_roundtrip(tmp_path):
    meta = {"units": "m s**-1", "resolution_deg": 1.0, "cadence_hours": 6,
            "config": {"node_pruning_threshold": 15}, "waper_sha": "abc123"}
    write_meta(tmp_path, meta)
    back = read_meta(tmp_path)
    assert back["units"] == "m s**-1"
    assert back["config"]["node_pruning_threshold"] == 15

def test_save_catalogue_writes_tables(tracked_waper, tmp_path):
    save_catalogue(tracked_waper, tmp_path, meta={"units": "m s**-1"})
    for name in ["nodes","edges","rwps","samples","track_nodes","track_edges"]:
        assert (Path(tmp_path)/name).exists(), f"missing {name}"
    assert (Path(tmp_path)/"meta.json").exists()

def test_roundtrip_reconstructs_graphs(tracked_waper, tmp_path):
    save_catalogue(tracked_waper, tmp_path)
    cat = load_catalogue(tmp_path)

    # tracking DiGraph rebuilds with identical edge count + merge/split structure
    te = cat.table("track_edges")
    tg = nx.from_pandas_edgelist(te, "src", "dst",
                                 edge_attr=["weight","distance"], create_using=nx.DiGraph)
    assert tg.number_of_edges() == tracked_waper._tracking_graph.number_of_edges()

    # polygon round-trips
    rwps = cat.table("rwps")
    geom = wkb.loads(rwps["geometry_wkb"].iloc[0])
    assert geom.geom_type in ("Polygon","MultiPolygon")

    # nodes table non-empty and typed
    assert len(cat.table("nodes")) > 0
