"""
Sensitivity analysis of WAPER tuning parameters.

Sweeps over parameter values and computes per-timestep and aggregate statistics
of the identified RWPs, following the approach of Pandey et al. (2020), Fig. 4.

Statistics computed:
  - Mean number of edges per RWP
  - Mean / median edge length (km)
  - Mean RWP extent (sum of edge lengths per RWP, km)
  - Number of timesteps containing at least one RWP
  - Edge length distribution (histogram)

Usage:
    python datasets/sensitivity.py
"""

import os
import sys
import math
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from waper import Waper
from waper.identification.utils import haversine_distance

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(DATASETS_DIR, "figures", "sensitivity")

# ------------------------------------------------------------------ #
#  Base configuration (same as visualize.py)                          #
# ------------------------------------------------------------------ #
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

# ------------------------------------------------------------------ #
#  Parameter sweeps                                                   #
# ------------------------------------------------------------------ #

# Gradient threshold (edge_pruning_threshold) — "GT" in Pandey et al.
# Baseline edge weights are in the 0.009–0.12 range, so sweep across that.
GT_VALUES = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

# Scalar threshold (node_pruning_threshold) — "ST" in Pandey et al.
ST_VALUES = [10, 15, 20, 25, 30, 35, 40, 45, 50]

# Hill-climbing penalty length scale (new parameter)
PENALTY_VALUES = [500, 1000, 1500, 2000, 3000, 5000]

# Which dataset to use for the sweep
DATASET_FILE = "forecast_bust.nc"


def compute_edge_lengths(waper_obj):
    """Compute haversine edge lengths (km) for all edges in all pruned graphs."""
    edge_lengths_per_timestep = []
    for tsd in waper_obj._time_step_data:
        graph = tsd.pruned_graph
        lengths = []
        for u, v in graph.edges():
            lon_u, lat_u = graph.nodes[u]["coords"]
            lon_v, lat_v = graph.nodes[v]["coords"]
            d = haversine_distance(lat_u, lon_u, lat_v, lon_v)
            lengths.append(d)
        edge_lengths_per_timestep.append(lengths)
    return edge_lengths_per_timestep


def compute_statistics(waper_obj):
    """Compute summary statistics for a single WAPER run."""
    edge_lengths_per_ts = compute_edge_lengths(waper_obj)
    all_edge_lengths = [d for ts in edge_lengths_per_ts for d in ts]

    num_timesteps = len(waper_obj._time_step_data)
    timesteps_with_rwp = 0
    edges_per_rwp = []
    extent_per_rwp = []
    num_rwps_per_timestep = []
    num_nodes_per_timestep = []
    num_edges_per_timestep = []

    for tsd in waper_obj._time_step_data:
        paths = tsd.identified_rwp_paths
        graph = tsd.pruned_graph
        n_rwps = len(paths)
        num_rwps_per_timestep.append(n_rwps)
        num_nodes_per_timestep.append(graph.number_of_nodes())
        num_edges_per_timestep.append(graph.number_of_edges())

        if n_rwps > 0:
            timesteps_with_rwp += 1

        for path in paths:
            n_edges = len(path) - 1
            edges_per_rwp.append(n_edges)
            extent = 0.0
            for k in range(n_edges):
                u, v = path[k], path[k + 1]
                lon_u, lat_u = graph.nodes[u]["coords"]
                lon_v, lat_v = graph.nodes[v]["coords"]
                extent += haversine_distance(lat_u, lon_u, lat_v, lon_v)
            extent_per_rwp.append(extent)

    stats = {
        "num_timesteps": num_timesteps,
        "timesteps_with_rwp": timesteps_with_rwp,
        "num_rwps_per_timestep": num_rwps_per_timestep,
        "num_nodes_per_timestep": num_nodes_per_timestep,
        "num_edges_per_timestep": num_edges_per_timestep,
        "mean_rwps_per_timestep": np.mean(num_rwps_per_timestep) if num_rwps_per_timestep else 0,
        "mean_edges_per_rwp": np.mean(edges_per_rwp) if edges_per_rwp else 0,
        "mean_edge_length_km": np.mean(all_edge_lengths) if all_edge_lengths else 0,
        "median_edge_length_km": np.median(all_edge_lengths) if all_edge_lengths else 0,
        "max_edge_length_km": np.max(all_edge_lengths) if all_edge_lengths else 0,
        "mean_extent_km": np.mean(extent_per_rwp) if extent_per_rwp else 0,
        "all_edge_lengths": all_edge_lengths,
        "edges_per_rwp": edges_per_rwp,
        "extent_per_rwp": extent_per_rwp,
    }
    return stats


def run_sweep(ds, param_name, param_values, label):
    """Run WAPER for each value of a single parameter, collecting statistics."""
    results = {}
    for val in param_values:
        kwargs = dict(BASE_KWARGS)
        kwargs[param_name] = val
        print(f"  {param_name} = {val}")
        waper = Waper(ds, **kwargs)
        waper.identify_rwps()
        stats = compute_statistics(waper)
        results[val] = stats
        print(f"    RWPs/ts={stats['mean_rwps_per_timestep']:.1f}  "
              f"edges/RWP={stats['mean_edges_per_rwp']:.1f}  "
              f"mean_edge={stats['mean_edge_length_km']:.0f} km  "
              f"median_edge={stats['median_edge_length_km']:.0f} km  "
              f"max_edge={stats['max_edge_length_km']:.0f} km  "
              f"extent={stats['mean_extent_km']:.0f} km  "
              f"ts_with_rwp={stats['timesteps_with_rwp']}/{stats['num_timesteps']}")
    return results


def plot_sweep(results, param_values, param_label, filename_prefix):
    """Produce a 4-panel figure like Pandey et al. Fig. 4."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Sensitivity to {param_label}", fontsize=13)

    x = param_values

    # (a) Mean number of edges per RWP
    ax = axes[0, 0]
    y = [results[v]["mean_edges_per_rwp"] for v in x]
    ax.plot(x, y, "ko-")
    ax.set_ylabel("No. Edges")
    ax.set_title("Mean Number of edges per RWP")

    # (b) Mean / Median edge length
    ax = axes[0, 1]
    y_mean = [results[v]["mean_edge_length_km"] for v in x]
    y_median = [results[v]["median_edge_length_km"] for v in x]
    ax.plot(x, y_mean, "ko-", label="mean")
    ax.plot(x, y_median, "k*--", label="median")
    ax.set_ylabel("km")
    ax.set_title("Mean/Median Edge length")
    ax.legend()

    # (c) Mean extent of RWP
    ax = axes[1, 0]
    y = [results[v]["mean_extent_km"] for v in x]
    ax.plot(x, y, "ko-")
    ax.set_ylabel("km")
    ax.set_xlabel(param_label)
    ax.set_title("Mean Extent")

    # (d) Number of timesteps with RWP / Mean RWPs per timestep
    ax = axes[1, 1]
    y_ts = [results[v]["timesteps_with_rwp"] for v in x]
    y_rwps = [results[v]["mean_rwps_per_timestep"] for v in x]
    ax2 = ax.twinx()
    l1 = ax.plot(x, y_ts, "ko-", label="timesteps with RWP")
    l2 = ax2.plot(x, y_rwps, "rs--", label="mean RWPs/timestep")
    ax.set_ylabel("No. Timesteps")
    ax2.set_ylabel("RWPs / timestep", color="r")
    ax.set_xlabel(param_label)
    ax.set_title("No of Timesteps with RWP")
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"{filename_prefix}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.relpath(path)}")


def plot_edge_length_histograms(results, param_values, param_label, filename_prefix):
    """Plot edge length distributions for each parameter value."""
    n = len(param_values)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    fig.suptitle(f"Edge length distributions — varying {param_label}", fontsize=13)

    for idx, val in enumerate(param_values):
        ax = axes[idx // ncols, idx % ncols]
        lengths = results[val]["all_edge_lengths"]
        if lengths:
            ax.hist(lengths, bins=20, color="steelblue", edgecolor="white")
            ax.axvline(np.median(lengths), color="red", ls="--", label="median")
        ax.set_title(f"{param_label}={val}", fontsize=9)
        ax.set_xlabel("km")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"{filename_prefix}_histograms.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.relpath(path)}")


def print_baseline_diagnostics(ds):
    """Print detailed per-timestep diagnostics for the baseline configuration."""
    print("\n=== Baseline diagnostics ===")
    waper = Waper(ds, **BASE_KWARGS)
    waper.identify_rwps()

    for t, tsd in enumerate(waper._time_step_data):
        graph = tsd.pruned_graph
        paths = tsd.identified_rwp_paths
        print(f"\n  Timestep {t}: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges, {len(paths)} RWPs")

        for u, v in graph.edges():
            lon_u, lat_u = graph.nodes[u]["coords"]
            lon_v, lat_v = graph.nodes[v]["coords"]
            d = haversine_distance(lat_u, lon_u, lat_v, lon_v)
            w = graph[u][v].get("weight", None)
            print(f"    edge {u} -> {v}:  dist={d:.0f} km  weight={w}  "
                  f"({lon_u:.1f},{lat_u:.1f}) -> ({lon_v:.1f},{lat_v:.1f})")

        for i, path in enumerate(paths):
            extent = 0.0
            for k in range(len(path) - 1):
                pu, pv = path[k], path[k + 1]
                lon_u, lat_u = graph.nodes[pu]["coords"]
                lon_v, lat_v = graph.nodes[pv]["coords"]
                extent += haversine_distance(lat_u, lon_u, lat_v, lon_v)
            print(f"    RWP {i}: {len(path)} nodes, extent={extent:.0f} km, "
                  f"path={path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    nc_path = os.path.join(DATASETS_DIR, DATASET_FILE)
    if not os.path.exists(nc_path):
        print(f"Dataset not found: {nc_path}")
        sys.exit(1)

    print(f"Loading {nc_path} ...")
    ds = xr.open_dataset(nc_path)
    print(f"  {len(ds['time'])} timesteps")

    # Baseline diagnostics: detailed per-edge info
    print_baseline_diagnostics(ds)

    # Sweep 1: Gradient Threshold (GT)
    print("\n=== Sweep: Gradient Threshold (edge_pruning_threshold) ===")
    gt_results = run_sweep(ds, "edge_pruning_threshold", GT_VALUES,
                           "Gradient Threshold")
    plot_sweep(gt_results, GT_VALUES, "Gradient Threshold", "sweep_gt")
    plot_edge_length_histograms(gt_results, GT_VALUES, "GT", "sweep_gt")

    # Sweep 2: Scalar Threshold (ST)
    print("\n=== Sweep: Scalar Threshold (node_pruning_threshold) ===")
    st_results = run_sweep(ds, "node_pruning_threshold", ST_VALUES,
                           "Scalar Threshold")
    plot_sweep(st_results, ST_VALUES, "Scalar Threshold (m/s)", "sweep_st")
    plot_edge_length_histograms(st_results, ST_VALUES, "ST", "sweep_st")

    # Sweep 3: Penalty Length Scale
    print("\n=== Sweep: Penalty Length Scale (penalty_length_scale_km) ===")
    pl_results = run_sweep(ds, "penalty_length_scale_km", PENALTY_VALUES,
                           "Penalty Length Scale")
    plot_sweep(pl_results, PENALTY_VALUES, "Penalty Length Scale (km)",
               "sweep_penalty")
    plot_edge_length_histograms(pl_results, PENALTY_VALUES, "L_char",
                                "sweep_penalty")

    print(f"\nAll figures saved in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
