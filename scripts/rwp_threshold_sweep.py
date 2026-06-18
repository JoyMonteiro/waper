"""Amplitude-threshold sweep. For each percentile of |v| over the whole dataset,
re-run RWP identification with node_pruning_threshold set to that value (so only
strong crests/troughs survive into the final RWP graph), then render a
full-hemisphere PlateCarree GIF of forecast_bust with:
  - the v field (white-near-zero colormap, horizontal slim colorbar)
  - the final RWP graph: nodes (crests/troughs) + edges of each identified RWP
  - the energy (amplitude^2)-weighted centroid of each RWP (gold star)
  - the 500 km energy disks used for tracking, outlined at each crest/trough
One GIF per threshold so the effect of the threshold can be compared.
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from PIL import Image

from waper import Waper
from waper.tracking.rwp_polygon import energy_disks
from waper.interface.colormaps import joy_nl8

PERCENTILES = [85, 90, 95, 99]
OUTDIR = "/tmp/rwp_sweep"
os.makedirs(OUTDIR, exist_ok=True)

ds = xr.open_dataset("datasets/forecast_bust.nc")
av = np.abs(ds["v"].values).ravel()
vmax = float(np.percentile(av, 99))
proj = ccrs.PlateCarree(central_longitude=180)
pc = ccrs.PlateCarree()
geod = ccrs.Geodetic()
stereo = ccrs.Stereographic(central_latitude=90, central_longitude=0)
ntimes = ds.sizes["time"]


def run_waper(node_thresh):
    w = Waper(data_array=ds, scalar_name="v",
              latitude_label="latitude", longitude_label="longitude", time_label="time",
              clip_value=2, extrema_threshold=10, min_latitude=20, max_latitude=80,
              node_pruning_threshold=node_thresh, edge_pruning_threshold=0.02,
              max_edge_weight=1, track_pruning_threshold=0.3)
    w.identify_rwps()
    return w


def render_frame(w, t, pct, thresh, outdir):
    tsd = w._time_step_data[t]
    fig = plt.figure(figsize=(12, 6.0))
    ax = plt.axes(projection=proj)
    ax.set_extent([-180, 180, 12, 88], crs=pc)
    cf = ds["v"].isel(time=t).plot.contourf(
        ax=ax, transform=pc, levels=15, cmap=joy_nl8,
        vmin=-vmax, vmax=vmax, add_colorbar=False)
    ax.coastlines(linewidth=0.5, color="0.4")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7",
                 xlabel_style={"size": 7}, ylabel_style={"size": 7})
    cbar = fig.colorbar(cf, ax=ax, orientation="horizontal",
                        shrink=0.6, aspect=40, pad=0.07)
    cbar.set_label("v (m s$^{-1}$)")

    n_rwp = 0
    for path in tsd.identified_rwp_paths:
        info = tsd.rwp_info[tuple(path)]
        n_rwp += 1
        nodes = [(tsd.pruned_graph.nodes[n]["coords"][0],
                  tsd.pruned_graph.nodes[n]["coords"][1],
                  tsd.pruned_graph.nodes[n]["scalar"]) for n in path]
        # edges of the RWP graph (great-circle segments between consecutive nodes)
        for (lo0, la0, _), (lo1, la1, _) in zip(nodes[:-1], nodes[1:]):
            ax.plot([lo0, lo1], [la0, la1], transform=geod,
                    color="k", linewidth=1.8, zorder=6)
        # disks + node markers
        for lon, lat, scalar in nodes:
            (disk, _), = energy_disks([(lon, lat, scalar)], hemisphere="north")
            ax.add_geometries([disk], crs=stereo, facecolor="none",
                              edgecolor=("firebrick" if scalar > 0 else "navy"),
                              linewidth=1.0, zorder=5)
            ax.plot(lon, lat, transform=pc, marker="o", markersize=5,
                    markerfacecolor=("r" if scalar > 0 else "b"),
                    markeredgecolor="k", markeredgewidth=0.4, zorder=7)
        # amplitude-weighted centroid
        ax.plot(info["weighted_longitude"], info["weighted_latitude"], transform=pc,
                marker="*", markersize=13, markerfacecolor="gold",
                markeredgecolor="k", zorder=8)

    ax.set_title(f"forecast_bust  t={t}/{ntimes-1}   "
                 f"node_pruning_threshold = {thresh:.1f} m/s  (|v| {pct}th pct)   "
                 f"— {n_rwp} RWPs\nblack = RWP graph edges · disks = tracking footprints · "
                 f"gold ★ = amplitude-weighted centroid", fontsize=9)
    fn = f"{outdir}/f_{t:02d}.png"
    fig.savefig(fn, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return fn


for pct in PERCENTILES:
    thresh = float(np.percentile(av, pct))
    fdir = f"{OUTDIR}/p{pct}"
    os.makedirs(fdir, exist_ok=True)
    for f in glob.glob(f"{fdir}/*.png"):
        os.remove(f)
    print(f"--- percentile {pct} -> node_pruning_threshold {thresh:.2f} m/s ---", flush=True)
    w = run_waper(thresh)
    frames = [render_frame(w, t, pct, thresh, fdir) for t in range(ntimes)]
    imgs = [Image.open(f).convert("RGB") for f in frames]
    gif = f"{OUTDIR}/rwp_sweep_p{pct}.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=550, loop=0)
    print(f"    wrote {gif} ({len(frames)} frames)", flush=True)

print("DONE. GIFs:")
for pct in PERCENTILES:
    print(f"  /tmp/rwp_sweep/rwp_sweep_p{pct}.gif")
