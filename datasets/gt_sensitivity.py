"""
Plot RWP statistics as a function of the gradient threshold (GT = edge_pruning_threshold).

Statistics:
  - Max / Mean / Median edge length (km)
  - Average number of RWPs per timestep
  - Average number of components (nodes) per RWP
  - Mean east-west (longitudinal) extent per RWP (km)

Usage:
    python datasets/gt_sensitivity.py
"""

import os
import sys
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from waper import Waper
from waper.identification.utils import haversine_distance

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(DATASETS_DIR, "figures", "sensitivity")
DATASET_FILE = "forecast_bust.nc"

BASE_KWARGS = dict(
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

GT_VALUES = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

EARTH_RADIUS_KM = 6371.0


def lon_separation_km(lon_min, lon_max, mean_lat_deg):
    """Great-circle east-west span in km at a given latitude."""
    dlon_rad = math.radians(abs(lon_max - lon_min) % 360)
    if dlon_rad > math.pi:
        dlon_rad = 2 * math.pi - dlon_rad
    lat_rad = math.radians(mean_lat_deg)
    return EARTH_RADIUS_KM * dlon_rad * math.cos(lat_rad)


def compute_stats(waper_obj):
    all_edge_lengths = []
    rwps_per_ts = []
    nodes_per_rwp = []
    ew_extents = []

    for tsd in waper_obj._time_step_data:
        graph = tsd.pruned_graph
        paths = tsd.identified_rwp_paths
        rwps_per_ts.append(len(paths))

        for u, v in graph.edges():
            lon_u, lat_u = graph.nodes[u]["coords"]
            lon_v, lat_v = graph.nodes[v]["coords"]
            all_edge_lengths.append(haversine_distance(lat_u, lon_u, lat_v, lon_v))

        for path in paths:
            nodes_per_rwp.append(len(path))

            lons = [graph.nodes[n]["coords"][0] for n in path]
            lats = [graph.nodes[n]["coords"][1] for n in path]
            mean_lat = np.mean(lats)
            lon_min, lon_max = min(lons), max(lons)
            ew_extents.append(lon_separation_km(lon_min, lon_max, mean_lat))

    return {
        "max_edge_km":    max(all_edge_lengths) if all_edge_lengths else 0,
        "mean_edge_km":   np.mean(all_edge_lengths) if all_edge_lengths else 0,
        "median_edge_km": np.median(all_edge_lengths) if all_edge_lengths else 0,
        "rwps_per_ts":    np.mean(rwps_per_ts),
        "nodes_per_rwp":  np.mean(nodes_per_rwp) if nodes_per_rwp else 0,
        "ew_extent_km":   np.mean(ew_extents) if ew_extents else 0,
    }


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    nc_path = os.path.join(DATASETS_DIR, DATASET_FILE)
    print(f"Loading {nc_path} ...")
    ds = xr.open_dataset(nc_path)
    print(f"  {len(ds['time'])} timesteps\n")

    results = {}
    for gt in GT_VALUES:
        kwargs = dict(BASE_KWARGS, edge_pruning_threshold=gt)
        print(f"GT = {gt:.3f} ...", end=" ", flush=True)
        waper = Waper(ds, **kwargs)
        waper.identify_rwps()
        stats = compute_stats(waper)
        results[gt] = stats
        print(
            f"max_edge={stats['max_edge_km']:.0f} km  "
            f"mean_edge={stats['mean_edge_km']:.0f} km  "
            f"median_edge={stats['median_edge_km']:.0f} km  "
            f"RWPs/ts={stats['rwps_per_ts']:.1f}  "
            f"nodes/RWP={stats['nodes_per_rwp']:.1f}  "
            f"EW_extent={stats['ew_extent_km']:.0f} km"
        )

    x = GT_VALUES
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Sensitivity to Gradient Threshold (GT = edge_pruning_threshold)",
                 fontsize=13)

    panels = [
        (axes[0, 0], "max_edge_km",    "Max edge length (km)",           "Max edge length"),
        (axes[0, 1], "mean_edge_km",   "Mean edge length (km)",           "Mean edge length"),
        (axes[0, 2], "median_edge_km", "Median edge length (km)",         "Median edge length"),
        (axes[1, 0], "rwps_per_ts",    "Avg RWPs per timestep",           "Avg RWPs / timestep"),
        (axes[1, 1], "nodes_per_rwp",  "Avg components (nodes) per RWP",  "Avg nodes per RWP"),
        (axes[1, 2], "ew_extent_km",   "Mean E–W extent (km)",            "Mean E–W extent"),
    ]

    for ax, key, ylabel, title in panels:
        y = [results[gt][key] for gt in x]
        ax.plot(x, y, "ko-", markersize=6)
        # Mark the current default
        ax.axvline(3e-5, color="red", ls=":", alpha=0.5, label="current default")
        ax.set_xlabel("GT (edge_pruning_threshold)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "gt_sensitivity.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
