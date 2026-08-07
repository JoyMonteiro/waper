"""Full-hemisphere PlateCarree frames of forecast_bust showing ALL the energy
disks used for tracking (every extremum of every identified RWP), with disk
outline thickness scaling with |v| (energy), the single globally strongest disk
highlighted, and each RWP's energy-weighted centroid marked. A sequence so the
disks can be watched moving.
"""
import os
import warnings

warnings.filterwarnings("ignore")
import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from waper import Waper
from waper.interface.colormaps import joy_nl8
from waper.tracking.rwp_polygon import energy_disks

OUT = "/tmp/hemis_frames"
os.makedirs(OUT, exist_ok=True)

ds = xr.open_dataset("datasets/forecast_bust.nc")
w = Waper(data_array=ds, scalar_name="v",
          latitude_label="latitude", longitude_label="longitude", time_label="time",
          clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80,
          node_pruning_threshold=20, edge_pruning_threshold=0.02, max_edge_weight=1,
          track_pruning_threshold=8000)
w.identify_rwps()

vmax = float(abs(ds["v"]).quantile(0.99))
proj = ccrs.PlateCarree(central_longitude=180)
pc = ccrs.PlateCarree()
stereo = ccrs.Stereographic(central_latitude=90, central_longitude=0)
ntimes = ds.sizes["time"]


def rwp_extrema(tsd):
    """list of (rwp_id, [(lon, lat, scalar)...], weighted_lon, weighted_lat)."""
    out = []
    for path in tsd.identified_rwp_paths:
        info = tsd.rwp_info[tuple(path)]
        nodes = [(tsd.pruned_graph.nodes[n]["coords"][0],
                  tsd.pruned_graph.nodes[n]["coords"][1],
                  tsd.pruned_graph.nodes[n]["scalar"]) for n in path]
        out.append((info["rwp_id"], nodes,
                    info["weighted_longitude"], info["weighted_latitude"]))
    return out


def render(t, path=None):
    tsd = w._time_step_data[t]
    rwps = rwp_extrema(tsd)

    fig = plt.figure(figsize=(12, 6.0))
    ax = plt.axes(projection=proj)
    ax.set_extent([-180, 180, 12, 88], crs=pc)
    cf = ds["v"].isel(time=t).plot.contourf(
        ax=ax, transform=pc, levels=15, cmap=joy_nl8,
        vmin=-vmax, vmax=vmax, add_colorbar=False)
    ax.coastlines(linewidth=0.5, color="0.4")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7",
                 xlabel_style={"size": 7}, ylabel_style={"size": 7})

    # horizontal colorbar, 0.6 of figure width, slim
    cbar = fig.colorbar(cf, ax=ax, orientation="horizontal",
                        shrink=0.6, aspect=40, pad=0.07)
    cbar.set_label("v (m s$^{-1}$)")

    # global strongest extremum this timestep (to highlight)
    all_amp = [abs(s) for _, nodes, _, _ in rwps for _, _, s in nodes]
    gmax = max(all_amp) if all_amp else 1.0

    for _rid, nodes, wlon, wlat in rwps:
        for lon, lat, scalar in nodes:
            (disk, _), = energy_disks([(lon, lat, scalar)], hemisphere="north")
            is_global_max = abs(scalar) >= gmax - 1e-9
            edge = "lime" if is_global_max else ("firebrick" if scalar > 0 else "navy")
            lw = 3.2 if is_global_max else (0.8 + 2.0 * abs(scalar) / vmax)
            ax.add_geometries([disk], crs=stereo, facecolor="none",
                              edgecolor=edge, linewidth=lw, zorder=6)
            ax.plot(lon, lat, transform=pc, marker="o",
                    markersize=4, markerfacecolor=("r" if scalar > 0 else "b"),
                    markeredgecolor="k", markeredgewidth=0.4, zorder=7)
        ax.plot(wlon, wlat, transform=pc, marker="*", markersize=12,
                markerfacecolor="gold", markeredgecolor="k", zorder=8)

    ax.set_title(f"forecast_bust  time index {t}/{ntimes-1}   —   "
                 f"all tracking disks (thickness ∝ |v|), lime = strongest, "
                 f"gold ★ = RWP energy centroid", fontsize=10)
    fn = f"{OUT}/hemis_{t:02d}.png"
    fig.savefig(fn, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return fn


frames = [render(t) for t in range(ntimes)]
print("wrote", len(frames), "frames to", OUT)

# montage of 8 evenly-spaced frames for inline review
idx = np.linspace(0, ntimes - 1, 8).round().astype(int)
sel = [f"{OUT}/hemis_{t:02d}.png" for t in idx]
nrow, ncol = 4, 2
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 8.5, nrow * 3.0))
axes = np.array(axes).reshape(-1)
for a in axes:
    a.axis("off")
# strict: the 4x2 grid and the 8 sampled frames are both fixed at 8, so a
# mismatch means the grid shape and the sample count drifted apart.
for a, fn in zip(axes, sel, strict=True):
    a.imshow(plt.imread(fn))
fig.tight_layout()
fig.savefig("/tmp/hemis_montage.png", dpi=110, bbox_inches="tight")
print("wrote /tmp/hemis_montage.png  (frames t =", list(idx), ")")
