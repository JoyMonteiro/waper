import pytest
import numpy as np
import pandas as pd
from shapely import wkb

def test_filter_and_accessors(cat):
    assert len(cat.rwps()) == len(cat.table("rwps"))
    sub = cat.filter(time=0)
    assert (sub.rwps()["time"] == 0).all()
    assert (sub.nodes()["time"] == 0).all()

def test_structural_metrics(cat):
    amp = cat.amplitudes()
    assert {"time", "rwp_id", "peak_amp"}.issubset(amp.columns)
    
    extent = cat.zonal_extent()
    assert {"time", "rwp_id", "zonal_extent_deg"}.issubset(extent.columns)
    
    wn = cat.implied_wavenumber()
    assert "implied_wavenumber" in wn.columns
    # synthetic field is wavenumber-4: implied wavenumber should be in a plausible band
    assert wn["implied_wavenumber"].dropna().between(2, 12).mean() > 0.5

def test_track_metrics(cat):
    dur = cat.track_durations()
    assert {"track_id", "duration_hours"}.issubset(dur.columns)
    
    prop = cat.track_propagation()
    assert {"track_id", "propagation_deg"}.issubset(prop.columns)
    
    gv = cat.group_velocity()
    assert {"track_id", "group_velocity_ms"}.issubset(gv.columns)

def test_graph_topology(cat):
    m = cat.merges()
    s = cat.splits()
    assert set(m.columns) >= {"key", "time", "feature", "in_degree"}
    assert set(s.columns) >= {"key", "time", "feature", "out_degree"}
    
    box = (-180, 180, 20, 81.0)
    tracks = cat.tracks_through(box)
    assert isinstance(tracks, list)
    
    if len(tracks) > 0:
        prov = cat.provenance(tracks[0])
        assert {"track_id", "genesis_lon", "genesis_lat", "genesis_time"}.issubset(prov.keys())

def test_climatology_aggregations(cat):
    apdf = cat.amplitude_pdf(bins=5)
    assert {"bin_left", "bin_right", "density"}.issubset(apdf.columns)
    
    dpdf = cat.duration_pdf(bins=2)
    assert {"bin_left", "bin_right", "density"}.issubset(dpdf.columns)
    
    sc = cat.seasonal_cycle()
    assert {"time", "count"}.issubset(sc.columns)
    
    sf = cat.spatial_frequency(dlon=10, dlat=10)
    assert {"lon_bin", "lat_bin", "count"}.issubset(sf.columns)
    
    cc = cat.cross_stat_correlations()
    assert "amp_vs_extent_r" in cc

def test_region_phase(cat):
    box = (-180, 180, 20, 81.0)
    assert len(cat.rwps_in(box)) > 0
    
    r = cat.rwps()
    point = (r.weighted_lon.iloc[0], r.weighted_lat.iloc[0])
    time = r.time.iloc[0]
    
    pkt = cat.packet_at(point, time)
    assert pkt is not None
    
    res = cat.phase_at(point, time)
    assert {"nearest_node_type", "fractional_position", "nearest_node_lon"}.issubset(res)

def test_match_points(cat):
    r = cat.rwps()
    pts = pd.DataFrame({"lon": [r.weighted_lon.iloc[0]], "lat": [r.weighted_lat.iloc[0]], "time": [r.time.iloc[0]]})
    m = cat.match_points(pts, radius_km=2000)
    assert "matched" in m.columns and bool(m["matched"].iloc[0]) is True
