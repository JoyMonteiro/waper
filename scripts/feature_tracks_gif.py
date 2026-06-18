"""GIF of feature tracks over forecast_bust: each tracked crest/trough is one
coloured trajectory, so continuity (and whether neighbours move together) is
visible. Run: ~/miniconda3/envs/waper/bin/python scripts/feature_tracks_gif.py
"""
import glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from PIL import Image
from waper.interface.api import Waper
from waper.tracking.feature_tracks import extract_features, track_features
from waper.interface.colormaps import joy_nl8

ds = xr.open_dataset("datasets/forecast_bust.nc")
av = np.abs(ds["v"].values).ravel(); vmax = float(np.percentile(av, 99))
thr = float(np.percentile(av, 90))
w = Waper(data_array=ds, scalar_name="v", latitude_label="latitude",
          longitude_label="longitude", time_label="time", clip_value=2,
          extrema_threshold=10, min_latitude=20, max_latitude=80,
          node_pruning_threshold=20, edge_pruning_threshold=0.02, max_edge_weight=1,
          track_pruning_threshold=0.3)
w.identify_rwps()
nt = ds.sizes["time"]
fb = [extract_features(w._time_step_data[t], t, "v", 2, thr) for t in range(nt)]
tracks = track_features(fb, max_recover_steps=2, lat_bounds=(20.0, 80.0))

cmap = plt.cm.tab20
colors = {tr.track_id: cmap(tr.track_id % 20) for tr in tracks}
proj = ccrs.PlateCarree(central_longitude=180); pc = ccrs.PlateCarree()
frames = []
for t in range(nt):
    fig = plt.figure(figsize=(12, 6)); ax = plt.axes(projection=proj)
    ax.set_extent([-180, 180, 12, 88], crs=pc)
    cf = ds["v"].isel(time=t).plot.contourf(ax=ax, transform=pc, levels=15, cmap=joy_nl8,
                                            vmin=-vmax, vmax=vmax, add_colorbar=False)
    ax.coastlines(linewidth=0.5, color="0.4")
    fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.6, aspect=40, pad=0.07
                 ).set_label("v (m s$^{-1}$)")
    for tr in tracks:
        pts = [(s.lon, s.lat) for s in tr.steps if s.time <= t]
        if len(pts) >= 2:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], transform=ccrs.Geodetic(),
                    color=colors[tr.track_id], linewidth=1.6, zorder=5)
        here = [s for s in tr.steps if s.time == t]
        for s in here:
            ax.plot(s.lon, s.lat, transform=pc, marker=("s" if s.recovered else "o"),
                    markersize=6, markerfacecolor=colors[tr.track_id],
                    markeredgecolor="k", markeredgewidth=0.4, zorder=7)
    ax.set_title(f"feature tracks  t={t}/{nt-1}  (square = recovered step)", fontsize=10)
    fn = f"/tmp/feature_tracks_{t:02d}.png"; fig.savefig(fn, dpi=110, bbox_inches="tight")
    plt.close(fig); frames.append(fn)

imgs = [Image.open(f).convert("RGB") for f in sorted(glob.glob("/tmp/feature_tracks_*.png"))]
imgs[0].save("/tmp/feature_tracks.gif", save_all=True, append_images=imgs[1:], duration=550, loop=0)
print("wrote /tmp/feature_tracks.gif")
