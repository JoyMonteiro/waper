"""GIF of feature convex-hull footprints over forecast_bust_hourly (April 12–18 2011).

One patch per extremum per timestep: red = max, blue = min.
Strong features are opaque; weak features are semi-transparent with a dashed edge.

Run: ~/miniconda3/envs/waper/bin/python scripts/feature_tracks_gif.py
"""
import glob
import os
import warnings

import matplotlib

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image

from waper.interface.api import Waper
from waper.interface.colormaps import joy_nl8
from waper.tracking.feature_tracks import extract_features, track_features

# Clear old frames
for f in glob.glob("/tmp/feature_tracks_???.png"):
    os.remove(f)

# Load hourly data; rename valid_time → time and drop singleton pressure_level.
# Coarsen 0.25° → 1° to match the original grid resolution (otherwise Dijkstra
# in cluster_extrema is 16× slower due to the finer VTK mesh).
raw = xr.open_dataset("datasets/forecast_bust_hourly.nc")
ds = (raw.rename({"valid_time": "time"})
         .squeeze("pressure_level", drop=True)
         .sel(time=slice("2011-04-12", "2011-04-18"))
         .coarsen(latitude=4, longitude=4, boundary="trim").mean()
         .assign_coords(longitude=lambda d: d.longitude % 360)
         .sortby("longitude"))

av = np.abs(ds["v"].values).ravel()
vmax = float(np.percentile(av, 99))
thr  = float(np.percentile(av, 90))

w = Waper(data_array=ds, scalar_name="v", latitude_label="latitude",
          longitude_label="longitude", time_label="time", clip_value=2,
          extrema_threshold=10, min_latitude=20, max_latitude=80,
          node_pruning_threshold=20, edge_pruning_threshold=0.02, max_edge_weight=1,
          track_pruning_threshold=0.3)
w.identify_rwps()

nt = ds.sizes["time"]
print(f"Extracting features for {nt} timesteps...")
fb = [extract_features(w._time_step_data[t], t, "v", 10, thr, footprint_fraction=3) for t in range(nt)]

# No displacement cap — rely on IoU alone; min_split_iou guards against
# phantom splits from large convex-hull footprints
tracks = track_features(fb, max_recover_steps=4, lat_bounds=(20.0, 80.0),
                        max_displacement_deg=None, min_split_iou=0.05)

# Top-3 strongest features at t=0
top3_t0 = sorted(fb[0], key=lambda f: abs(f.scalar), reverse=True)[:3]
top3_keys = {(round(f.lon, 4), round(f.lat, 4)) for f in top3_t0}
ROOT_COLORS = ["#ff7f0e", "#2ca02c", "#9467bd"]  # orange, green, purple

tracks_by_id = {tr.track_id: tr for tr in tracks}

top3_tracks = []
for tr in tracks:
    if tr.parent_id is not None:
        continue  # only original seeds, not split descendants
    t0_steps = [s for s in tr.steps if s.time == 0]
    if t0_steps and (round(t0_steps[0].lon, 4), round(t0_steps[0].lat, 4)) in top3_keys:
        top3_tracks.append(tr)
top3_tracks.sort(key=lambda tr: -abs(next(s for s in tr.steps if s.time == 0).scalar))

# Build the full family: top-3 + all split descendants
root_color = {tr.track_id: ROOT_COLORS[i] for i, tr in enumerate(top3_tracks)}

def _root_id(tr):
    while tr.parent_id is not None:
        parent = tracks_by_id.get(tr.parent_id)
        if parent is None:
            break
        tr = parent
    return tr.track_id

family_tracks = []
for tr in tracks:
    rid = _root_id(tr)
    if rid in root_color:
        family_tracks.append(tr)

pc       = ccrs.PlateCarree()
proj     = ccrs.PlateCarree(central_longitude=180)
stereo_crs = ccrs.Stereographic(central_latitude=90, central_longitude=0)
TYPE_COLOR = {"max": "#d62728", "min": "#1f77b4"}

print("Rendering frames...")
frames = []
for t in range(nt):
    fig = plt.figure(figsize=(12, 6))
    ax  = plt.axes(projection=proj)
    ax.set_extent([0, 360, 20, 80], crs=pc)

    cf = ds["v"].isel(time=t).plot.contourf(
        ax=ax, transform=pc, levels=15, cmap=joy_nl8,
        vmin=-vmax, vmax=vmax, add_colorbar=False)
    ax.coastlines(linewidth=0.5, color="0.4")
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.5",
                      alpha=0.6, linestyle="--")
    gl.top_labels  = False
    gl.right_labels = False
    fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.6, aspect=40, pad=0.07
                 ).set_label("v (m s$^{-1}$)")

    for f in fb[t]:
        color  = TYPE_COLOR[f.node_type]
        strong = f.strength == "strong"
        try:
            ax.add_geometries([f.footprint], crs=stereo_crs,
                              facecolor=color, alpha=0.35 if strong else 0.12,
                              edgecolor=color, linewidth=1.2 if strong else 0.6,
                              linestyle="-" if strong else "--", zorder=4)
        except Exception:
            continue
        ax.plot(f.lon, f.lat, transform=pc,
                marker="o", markersize=4,
                markerfacecolor=color, markeredgecolor="k",
                markeredgewidth=0.3, zorder=5)

    # Overlay family tracks (top-3 roots + split descendants)
    for tr in family_tracks:
        rid = _root_id(tr)
        color = root_color[rid]
        is_root = tr.track_id == rid
        past = [(s.lon, s.lat) for s in tr.steps if s.time <= t]
        if len(past) >= 2:
            ax.plot([p[0] for p in past], [p[1] for p in past],
                    transform=ccrs.Geodetic(), color=color,
                    linewidth=2.2 if is_root else 1.2,
                    linestyle="-" if is_root else "--", zorder=8)
        for s in (s for s in tr.steps if s.time == t):
            ax.plot(s.lon, s.lat, transform=pc,
                    marker="*" if is_root else "D",
                    markersize=10 if is_root else 6,
                    color=color, markeredgecolor="k", markeredgewidth=0.4, zorder=9)

    legend = [mpatches.Patch(color=TYPE_COLOR["max"], label="max (strong / weak)"),
              mpatches.Patch(color=TYPE_COLOR["min"], label="min (strong / weak)")]
    for rank, tr in enumerate(top3_tracks):
        s0 = next(s for s in tr.steps if s.time == 0)
        legend.append(mpatches.Patch(color=ROOT_COLORS[rank],
                                     label=f"#{rank+1} {s0.node_type} {abs(s0.scalar):.0f} m/s  (— root  – – split)"))
    # Legend outside axes to the right so it doesn't cover the map
    ax.legend(handles=legend, loc="upper left",
              bbox_to_anchor=(1.02, 1), borderaxespad=0,
              fontsize=7, framealpha=0.85)

    dt_str = str(ds.time.values[t])[:16].replace("T", " ")
    ax.set_title(f"feature footprints + top-3 tracks   {dt_str} UTC   thr={thr:.1f} m/s",
                 fontsize=10)
    fn = f"/tmp/feature_tracks_{t:03d}.png"
    fig.savefig(fn, dpi=110, bbox_inches="tight")
    plt.close(fig)
    frames.append(fn)
    if t % 24 == 0:
        print(f"  t={t}/{nt-1}")

imgs = [Image.open(f).convert("RGB") for f in sorted(glob.glob("/tmp/feature_tracks_???.png"))]
imgs[0].save("/tmp/feature_tracks.gif", save_all=True, append_images=imgs[1:],
             duration=150, loop=0)
print(f"wrote /tmp/feature_tracks.gif  ({len(imgs)} frames, 150 ms/frame)")
