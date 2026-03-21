"""
Visualize WAPER RWP identification and tracking results for all datasets.

Produces per-timestep identification figures and per-dataset tracking figures,
saved under datasets/figures/<dataset_name>/.

Usage:
    python datasets/visualize.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

# Make sure waper is importable when running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from waper import Waper
from waper.tracking import tracking_graph as tg

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(DATASETS_DIR, "figures")

DATASETS = [
    "souders_v_1.nc",
    "souders_v_2.nc",
    "forecast_bust.nc",
]

# WAPER configuration (shared across datasets)
WAPER_KWARGS = dict(
    scalar_name="v",
    latitude_label="latitude",
    longitude_label="longitude",
    time_label="time",
    clip_value=2,
    extrema_threshold=11,
    min_latitude=20,
    max_latitude=80,
    node_pruning_threshold=20,
    edge_pruning_threshold=3e-5,
    max_edge_weight=1,
    track_pruning_threshold=0.3,
)

# Use a generous distance threshold for track display (keep all plausible tracks)
TRACK_DISPLAY_THRESHOLD_KM = 8000


def save_fig(fig, path):
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {os.path.relpath(path)}")


def process_dataset(nc_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoading {nc_path} ...")
    ds = xr.open_dataset(nc_path)
    num_timesteps = len(ds["time"])
    print(f"  {num_timesteps} timesteps: {ds['time'].values}")

    waper = Waper(ds, **WAPER_KWARGS)

    print("  Running identification ...")
    waper.identify_rwps()

    print("  Running tracking ...")
    waper.track_rwps()

    # ------------------------------------------------------------------ #
    #  Per-timestep identification figures                                 #
    # ------------------------------------------------------------------ #
    for t in range(num_timesteps):
        ts_label = str(ds["time"].values[t])[:16].replace(":", "-").replace("T", "_")

        # 1. Cluster plot (raw maxima/minima labelled by region + cluster)
        fig = plt.figure(figsize=(14, 10))
        waper.plot_clusters(t)
        fig.suptitle(f"Clusters  |  t={ts_label}", fontsize=10)
        save_fig(fig, os.path.join(out_dir, f"clusters_t{t:03d}_{ts_label}.png"))

        # 2. Pruned association graph overlaid on scalar field
        fig = plt.figure(figsize=(10, 8))
        waper.plot_pruned_graph(t)
        fig.suptitle(f"Pruned association graph  |  t={ts_label}", fontsize=10)
        save_fig(fig, os.path.join(out_dir, f"pruned_graph_t{t:03d}_{ts_label}.png"))

        # 3. Identified RWP paths (nodes coloured by max/min, edges by path)
        fig = plt.figure(figsize=(10, 8))
        waper.plot_rwp_graphs(t)
        fig.suptitle(f"Identified RWP paths  |  t={ts_label}", fontsize=10)
        save_fig(fig, os.path.join(out_dir, f"rwp_paths_t{t:03d}_{ts_label}.png"))

        # 4. RWP polygons
        fig = plt.figure(figsize=(10, 8))
        waper.plot_rwp_polygons(t)
        fig.suptitle(f"RWP polygons  |  t={ts_label}", fontsize=10)
        save_fig(fig, os.path.join(out_dir, f"polygons_t{t:03d}_{ts_label}.png"))

    # ------------------------------------------------------------------ #
    #  Tracking figures                                                    #
    # ------------------------------------------------------------------ #

    # 5. Track graph (centroid paths across time)
    fig = plt.figure(figsize=(12, 8))
    waper.plot_tracks(threshold=TRACK_DISPLAY_THRESHOLD_KM)
    fig.suptitle("Tracks (centroid paths)", fontsize=10)
    save_fig(fig, os.path.join(out_dir, "tracks_overview.png"))

    # 6. Per-track overlapping polygon sequence
    pruned = tg.prune_tracking_graph(
        waper._tracking_graph, threshold=TRACK_DISPLAY_THRESHOLD_KM
    )
    track_paths = tg.get_track_paths(pruned)
    print(f"  {len(track_paths)} tracks found")

    for i, path in enumerate(track_paths):
        fig = plt.figure(figsize=(10, 8))
        waper.plot_track_polygons(path)
        t_start = path[0][0]
        t_end = path[-1][0]
        ts_start = str(ds["time"].values[t_start])[:16]
        ts_end = str(ds["time"].values[t_end])[:16]
        fig.suptitle(
            f"Track {i+1}  |  t={ts_start} → {ts_end}  ({len(path)} steps)",
            fontsize=10,
        )
        save_fig(fig, os.path.join(out_dir, f"track_polygons_{i+1:02d}.png"))

    print(f"  Done. Figures in {out_dir}")


def main():
    print(f"Output directory: {FIGURES_DIR}")
    for nc_file in DATASETS:
        nc_path = os.path.join(DATASETS_DIR, nc_file)
        if not os.path.exists(nc_path):
            print(f"Skipping {nc_file} (not found)")
            continue
        dataset_name = os.path.splitext(nc_file)[0]
        out_dir = os.path.join(FIGURES_DIR, dataset_name)
        process_dataset(nc_path, out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
