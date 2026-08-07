"""Render the first N frames from cached features so we can LOOK at the
split. Highlights the purple seed's footprint family in heavy purple."""
import pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from waper.interface.colormaps import joy_nl8

with open("/tmp/diag_fb.pkl", "rb") as fh:
    fb, thr = pickle.load(fh)

raw = xr.open_dataset("datasets/forecast_bust_hourly.nc")
ds = (raw.rename({"valid_time": "time"})
         .squeeze("pressure_level", drop=True)
         .sel(time=slice("2011-04-12", "2011-04-18"))
         .coarsen(latitude=4, longitude=4, boundary="trim").mean()
         .assign_coords(longitude=lambda d: d.longitude % 360)
         .sortby("longitude"))
av = np.abs(ds["v"].values).ravel()
vmax = float(np.percentile(av, 99))

purple = sorted(fb[0], key=lambda f: abs(f.scalar), reverse=True)[2]
print(f"purple seed: {purple.node_type} lon={purple.lon:.1f} lat={purple.lat:.1f} scalar={purple.scalar:.1f}")

pc = ccrs.PlateCarree()
proj = ccrs.PlateCarree(central_longitude=-25)  # Atlantic-centered
stereo = ccrs.Stereographic(central_latitude=90, central_longitude=0)
NSHOW = 8

# Convert ds to -180..180 so the Atlantic is contiguous
ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")

def wrap(lon):
    return ((lon + 180) % 360) - 180

for t in range(NSHOW):
    fig = plt.figure(figsize=(13, 5))
    ax = plt.axes(projection=proj)
    # Zoom on the Atlantic trough region: 90W .. 40E, 25-62N
    ax.set_extent([-90, 40, 25, 62], crs=pc)
    ds["v"].isel(time=t).plot.contourf(ax=ax, transform=pc, levels=15, cmap=joy_nl8,
                                       vmin=-vmax, vmax=vmax, add_colorbar=False)
    ax.coastlines(linewidth=0.5, color="0.4")
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.5", alpha=0.6, linestyle="--")
    gl.top_labels = gl.right_labels = False

    # All min footprints this frame, labelled with lon/scalar
    for f in fb[t]:
        if f.node_type != "min":
            continue
        strong = f.strength == "strong"
        try:
            ax.add_geometries([f.footprint], crs=stereo, facecolor="none",
                              edgecolor="blue", linewidth=1.8 if strong else 0.7,
                              linestyle="-" if strong else "--", zorder=4)
        except Exception:
            pass
        ax.plot(f.lon, f.lat, transform=pc, marker="o", markersize=5,
                color="blue", markeredgecolor="k", zorder=5)
        ax.text(f.lon, f.lat + 0.8, f"{f.lon:.0f}/{f.scalar:.0f}", transform=pc,
                fontsize=7, ha="center", color="k", zorder=6)

    ax.set_title(f"t={t}  {str(ds.time.values[t])[:16]}  (min footprints; blue solid=strong)")
    fn = f"/tmp/diag_frame_{t:02d}.png"
    fig.savefig(fn, dpi=95, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fn}")
