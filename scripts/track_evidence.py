"""Render a sequence of PlateCarree frames following one RWP track through
forecast_bust, with the energy disks (the tracking footprints) outlined and the
energy-maximum disk highlighted. Evidence that the energy-weighted track moves.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from waper import Waper
from waper.tracking import tracking_graph as tg
from waper.tracking.rwp_polygon import energy_disks, transform_to_stereographic
from waper.interface.colormaps import joy_nl8

OUT = "/tmp/track_frames"
os.makedirs(OUT, exist_ok=True)

ds = xr.open_dataset("datasets/forecast_bust.nc")
w = Waper(data_array=ds, scalar_name="v",
          latitude_label="latitude", longitude_label="longitude", time_label="time",
          clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80,
          node_pruning_threshold=20, edge_pruning_threshold=0.02, max_edge_weight=1,
          track_pruning_threshold=0.3)
w.identify_rwps()
w.track_rwps()

def rwp_nodes(time, feature):
    """(lon, lat, scalar) for every extremum of the RWP `feature` at `time`."""
    tsd = w._time_step_data[time]
    for path in tsd.identified_rwp_paths:
        info = tsd.rwp_info[tuple(path)]
        if info["rwp_id"] == feature:
            out = []
            for n in path:
                lon, lat = tsd.pruned_graph.nodes[n]["coords"]
                out.append((lon, lat, tsd.pruned_graph.nodes[n]["scalar"], info))
            return out
    return []


def disk_lonlat(lon, lat, radius_m=500e3):
    (disk, _), = energy_disks([(lon, lat, 1.0)], hemisphere="north", radius_m=radius_m)
    xs, ys = disk.exterior.xy
    lons, lats = transform_to_stereographic(np.asarray(xs), np.asarray(ys),
                                            hemisphere="north", inverse=True)
    return lons, lats


def centroid_traj(path):
    out = []
    for (t, feat) in path:
        nodes = rwp_nodes(t, feat)
        if nodes:
            info = nodes[0][3]
            out.append((t, info["weighted_longitude"], info["weighted_latitude"]))
    return out


def signed_east(lon1, lon2):
    return ((lon2 - lon1 + 180.0) % 360.0) - 180.0


def net_east_no_teleport(traj, max_step=40.0):
    """Net eastward centroid displacement, or -inf if any single step teleports
    (a sign of mis-association rather than a propagating packet)."""
    if len(traj) < 2:
        return float("-inf")
    steps = [signed_east(traj[i][1], traj[i + 1][1]) for i in range(len(traj) - 1)]
    if max(abs(s) for s in steps) > max_step:
        return float("-inf")
    return sum(steps)


# pick the cleanest eastward-propagating track: largest net eastward drift with
# no teleport jumps; prefer length 5-9 so consecutive frames are reviewable
candidates = [p for p in tg.get_track_paths(w._tracking_graph) if 5 <= len(p) <= 9]
if not candidates:
    candidates = [p for p in tg.get_track_paths(w._tracking_graph) if len(p) >= 5]
candidates.sort(key=lambda p: net_east_no_teleport(centroid_traj(p)), reverse=True)
track = candidates[0][:8]           # consecutive frames, capped at 8
trail = centroid_traj(track)

print("chosen track length:", len(track))
print("centroid trajectory (time, lon, lat):")
for t, lo, la in trail:
    print(f"  t={t:2d}  lon={lo:7.1f}  lat={la:5.1f}")

vmax = float(abs(ds["v"]).quantile(0.99))
proj = ccrs.PlateCarree()
pc = ccrs.PlateCarree()

# regional extent around the track so the disk translation is clearly visible
_lons = [lo for _, lo, _ in trail]
_lats = [la for _, _, la in trail]
EXT = [min(_lons) - 28, max(_lons) + 28,
       max(10, min(_lats) - 16), min(86, max(_lats) + 16)]

frames = []
for k, (t, feat) in enumerate(track):
    nodes = rwp_nodes(t, feat)
    if not nodes:
        continue
    fig = plt.figure(figsize=(11, 5.2))
    ax = plt.axes(projection=proj)
    ax.set_extent(EXT, crs=pc)
    da = ds["v"].isel(time=t)
    da.plot.contourf(ax=ax, transform=pc, levels=15, cmap=joy_nl8,
                     vmin=-vmax, vmax=vmax, add_colorbar=True,
                     cbar_kwargs=dict(shrink=0.7, label="v (m s$^{-1}$)"))
    ax.coastlines(linewidth=0.5, color="0.4")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7",
                 xlabel_style={"size": 7}, ylabel_style={"size": 7})

    energies = [abs(s) ** 2 for (_, _, s, _) in nodes]
    emax_idx = int(np.argmax(energies))
    for i, (lon, lat, scalar, info) in enumerate(nodes):
        lons, lats = disk_lonlat(lon, lat)
        is_max = (i == emax_idx)
        ax.plot(lons, lats, transform=pc,
                color="lime" if is_max else "k",
                linewidth=2.6 if is_max else 1.1,
                zorder=6)
        ax.plot(lon, lat, transform=pc, marker="o", markersize=7 if is_max else 4,
                markerfacecolor=("r" if scalar > 0 else "b"),
                markeredgecolor="k", zorder=7)

    # past centroids as faint dots (no line: avoids dateline streaks), current as star
    for (pt, plo, pla) in trail[:k]:
        ax.plot(plo, pla, transform=pc, marker="o", markersize=5,
                markerfacecolor="gold", markeredgecolor="k", alpha=0.45, zorder=5)
    cx, cy = trail[k][1], trail[k][2]
    ax.plot(cx, cy, transform=pc, marker="*", markersize=17,
            markerfacecolor="yellow", markeredgecolor="k", zorder=8)

    ax.set_title(f"forecast_bust — frame {k} (time index {t})   "
                 f"centroid lon={cx:.0f}, lat={cy:.0f}\n"
                 f"lime = energy-max disk · black = other extrema disks · gold ★ = energy-weighted centroid",
                 fontsize=9)
    fn = f"{OUT}/frame_{k:02d}.png"
    fig.savefig(fn, dpi=120, bbox_inches="tight")
    plt.close(fig)
    frames.append(fn)

print("wrote", len(frames), "frames to", OUT)

# montage (grid) for one-look review
n = len(frames)
ncol = 2
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 7, nrow * 3.3))
axes = np.array(axes).reshape(-1)
for ax in axes:
    ax.axis("off")
for ax, fn in zip(axes, frames):
    ax.imshow(plt.imread(fn))
fig.tight_layout()
fig.savefig("/tmp/track_montage.png", dpi=110, bbox_inches="tight")
print("wrote /tmp/track_montage.png")
