"""Driver: preprocess the forecast-bust dataset, cache the Zimin reference masks,
sweep the two WAPER variants, and write the agreement CSV."""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from waper import Waper
from waper.tracking.rwp_polygon import WAPER_IMAGE_SIZE

from .masks import (
    band_mask, compute_rwp_envelope, zimin_mask,
    edge_pruning_mask, node_amplitude_mask,
    t21_truncate, temporal_running_mean,
)
from .metrics import iou, disagreement_decomposition, detection_agreement

DATA_PATH = "datasets/forecast_bust_hourly.nc"
RESULTS_CSV = "results/method_comparison_sweep.csv"
ZIMIN_CACHE = "results/zimin_masks.npy"

GT_GRID = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08]
ST_GRID = [10, 15, 20, 25, 30, 35]
BAND = (20.0, 80.0)
ZIMIN_THRESHOLD = 14.0


def load_dataset(path=DATA_PATH):
    """Load + preprocess to 1-degree, lon 0-360, dims (time, latitude, longitude)."""
    raw = xr.open_dataset(path)
    da = (
        raw["v"]
        .rename({"valid_time": "time"})
        .squeeze("pressure_level", drop=True)
        .coarsen(latitude=4, longitude=4, boundary="trim").mean()
        .assign_coords(longitude=lambda d: d.longitude % 360)
        .sortby("longitude")
    )
    return da


def run_base_waper(v_da, node_pruning_threshold=5, edge_pruning_threshold=0.02):
    """WAPER run used only to harvest per-timestep association graphs."""
    w = Waper(
        data_array=v_da.to_dataset(name="v"),
        scalar_name="v",
        latitude_label="latitude",
        longitude_label="longitude",
        time_label="time",
        clip_value=2,
        # ST values below extrema_threshold will find no nodes (extrema are never
        # created below it), so the node-amplitude ST sweep is effectively
        # floor-bounded at 10.
        extrema_threshold=10,
        min_latitude=20,
        max_latitude=80,
        node_pruning_threshold=node_pruning_threshold,
        edge_pruning_threshold=edge_pruning_threshold,
    )
    w.identify_rwps()
    return w


def compute_zimin_masks(v_da, band, threshold=ZIMIN_THRESHOLD, t21=True, temporal_hours=24):
    """Stacked (ntime,WAPER_IMAGE_SIZE,WAPER_IMAGE_SIZE) bool reference masks.

    With the Souders (2014) defaults (``t21=True``, ``temporal_hours=24``) the
    Zimin envelope is post-processed by a T21 spatial truncation and a 24-h
    temporal running mean before thresholding. Set ``t21=False, temporal_hours=0``
    for the bare Zimin-2003 zonal envelope (no Souders smoothing).
    """
    # Sort latitude ascending for the interpolation path only — older scipy's
    # RegularGridInterpolator requires strictly ascending coordinates.  The
    # original (descending) orientation fed to WAPER is intentionally unchanged.
    v_sorted = v_da.sortby("latitude")
    lon = v_sorted.longitude.values
    lat = v_sorted.latitude.values
    nt = v_da.time.size

    env = np.empty((nt, lat.size, lon.size))
    for t in tqdm(range(nt), desc="Zimin envelope", unit="step"):
        env[t] = compute_rwp_envelope(v_sorted.isel(time=t).values, (3, 11))
    if t21:
        for t in tqdm(range(nt), desc="T21 spatial filter", unit="step"):
            env[t] = t21_truncate(env[t], ntrunc=21)
    if temporal_hours:
        env = temporal_running_mean(env, temporal_hours, v_da.time.values)

    masks = np.empty((nt, WAPER_IMAGE_SIZE, WAPER_IMAGE_SIZE), dtype=bool)
    for t in range(nt):
        masks[t] = zimin_mask(env[t], lon, lat, band, threshold=threshold)
    return masks


def _aggregate(method_masks, zimin_masks, band, method_name, threshold):
    ious, dets, m_onlys, z_onlys = [], [], [], []
    for mm, zm in zip(method_masks, zimin_masks):
        ious.append(iou(mm, zm))
        dets.append(detection_agreement(mm, zm))
        m_only, z_only = disagreement_decomposition(mm, zm, band)
        m_onlys.append(m_only); z_onlys.append(z_only)
    return {
        "method": method_name,
        "threshold": threshold,
        "mean_iou": float(np.mean(ious)),
        "detection_agreement": float(np.mean(dets)),
        "mean_method_only_frac": float(np.mean(m_onlys)),
        "mean_zimin_only_frac": float(np.mean(z_onlys)),
        "n_timesteps": len(ious),
    }


def sweep(v_da, gt_grid=GT_GRID, st_grid=ST_GRID):
    """Full agreement sweep -> tidy DataFrame."""
    band = band_mask(*BAND)

    print("[1/3] Computing Zimin reference masks (14 m/s, wavenumbers 3-11)...")
    zimin_masks = compute_zimin_masks(v_da, band)

    rows = []

    # Node-amplitude: one base run supplies association graphs; re-threshold per ST.
    print(f"[2/3] Node-amplitude sweep: base WAPER run, then {len(st_grid)} ST thresholds...")
    base = run_base_waper(v_da)
    assoc = [tsd.association_graph for tsd in base._time_step_data]
    for st in tqdm(st_grid, desc="Node-amplitude ST sweep", unit="thr"):
        method_masks = [node_amplitude_mask(g, st, band) for g in assoc]
        rows.append(_aggregate(method_masks, zimin_masks, band, "node_amplitude", st))

    # Edge-pruning: a full WAPER run per GT; read raster_data.
    print(f"[3/3] Edge-pruning sweep: {len(gt_grid)} full WAPER runs (one per GT)...")
    for i, gt in enumerate(gt_grid, 1):
        print(f"  edge-pruning run {i}/{len(gt_grid)}  (GT={gt})")
        w = run_base_waper(v_da, node_pruning_threshold=20, edge_pruning_threshold=gt)
        method_masks = [edge_pruning_mask(tsd, band) for tsd in w._time_step_data]
        rows.append(_aggregate(method_masks, zimin_masks, band, "edge_pruning", gt))

    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)
    v = load_dataset()
    df = sweep(v)
    df.to_csv(RESULTS_CSV, index=False)
    print(df.to_string(index=False))
    # report best-agreement thresholds
    for method in ("edge_pruning", "node_amplitude"):
        sub = df[df["method"] == method]
        best = sub.loc[sub["mean_iou"].idxmax()]
        print(f"best {method}: threshold={best['threshold']} mean_iou={best['mean_iou']:.3f}")


if __name__ == "__main__":
    main()
