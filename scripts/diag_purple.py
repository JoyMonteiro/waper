"""Diagnostic: trace the purple (3rd-strongest @ t0) feature step by step.

Caches feature extraction to /tmp/diag_fb.pkl so the tracking logic can be
re-examined instantly. Reports, at each step, what the purple head overlaps,
which fragment the root follows, and what happens to the split children.
"""
import os
import pickle
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr

from waper.interface.api import Waper
from waper.tracking.feature_tracks import extract_features, feature_overlap

CACHE = "/tmp/diag_fb.pkl"

if os.path.exists(CACHE):
    print(f"Loading cached features from {CACHE}")
    with open(CACHE, "rb") as fh:
        fb, thr = pickle.load(fh)
else:
    raw = xr.open_dataset("datasets/forecast_bust_hourly.nc")
    ds = (raw.rename({"valid_time": "time"})
             .squeeze("pressure_level", drop=True)
             .sel(time=slice("2011-04-12", "2011-04-18"))
             .coarsen(latitude=4, longitude=4, boundary="trim").mean()
             .assign_coords(longitude=lambda d: d.longitude % 360)
             .sortby("longitude"))
    av = np.abs(ds["v"].values).ravel()
    thr = float(np.percentile(av, 90))
    w = Waper(data_array=ds, scalar_name="v", latitude_label="latitude",
              longitude_label="longitude", time_label="time", clip_value=2,
              extrema_threshold=10, min_latitude=20, max_latitude=80,
              node_pruning_threshold=20, edge_pruning_threshold=0.02,
              max_edge_weight=1, track_pruning_threshold=8000)
    w.identify_rwps()
    nt = ds.sizes["time"]
    print(f"Extracting features for {nt} timesteps...")
    fb = [extract_features(w._time_step_data[t], t, "v", 10, thr, footprint_fraction=3)
          for t in range(nt)]
    with open(CACHE, "wb") as fh:
        pickle.dump((fb, thr), fh)
    print(f"Cached -> {CACHE}")

# --- Identify the purple feature: 3rd strongest at t0 ---
top3 = sorted(fb[0], key=lambda f: abs(f.scalar), reverse=True)[:3]
purple = top3[2]
print(f"\nthr (strong cutoff) = {thr:.2f} m/s")
print(f"purple seed: {purple.node_type} lon={purple.lon:.1f} lat={purple.lat:.1f} "
      f"scalar={purple.scalar:.1f} footprint_area={purple.footprint.area:.3e}")

# --- Walk forward, tracking the purple head by best-IoU (mirrors primary child) ---
head = purple
for t in range(1, len(fb)):
    curr = fb[t]
    same_type = [f for f in curr if f.node_type == head.node_type]
    overlaps = sorted(
        ((feature_overlap(head, f), f) for f in same_type),
        key=lambda x: -x[0])
    nonzero = [(ov, f) for ov, f in overlaps if ov > 0]
    if not nonzero:
        print(f"\nt={t}: NO OVERLAP with any {head.node_type} feature. "
              f"head was lon={head.lon:.1f} lat={head.lat:.1f} "
              f"area={head.footprint.area:.2e}  -> track would DIE here")
        # show nearest few by centroid distance for context
        def dist(f, head=head):
            dlon = ((f.lon - head.lon + 180) % 360) - 180
            return (dlon**2 + (f.lat - head.lat)**2) ** 0.5
        near = sorted(same_type, key=dist)[:4]
        for f in near:
            print(f"     nearest: lon={f.lon:.1f} lat={f.lat:.1f} "
                  f"scalar={f.scalar:.1f} {f.strength} dist={dist(f):.1f} area={f.footprint.area:.2e}")
        break
    n_signif = sum(1 for ov, f in nonzero if ov >= 0.05)
    primary_ov, primary = nonzero[0]
    east = ((primary.lon - head.lon + 180) % 360) - 180
    tag = "SPLIT" if n_signif >= 2 else ""
    print(f"t={t}: {len(nonzero)} overlaps ({n_signif} w/ IoU>=0.05) {tag}")
    for ov, f in nonzero[:5]:
        de = ((f.lon - head.lon + 180) % 360) - 180
        mark = "<<follows" if f is primary else ""
        print(f"     IoU={ov:.3f} lon={f.lon:.1f} lat={f.lat:.1f} "
              f"d_lon={de:+.1f} {f.strength} area={f.footprint.area:.2e} {mark}")
    head = primary
