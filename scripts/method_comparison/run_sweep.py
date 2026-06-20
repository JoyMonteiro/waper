"""Driver: preprocess the forecast-bust dataset, cache the Zimin reference masks,
sweep the two WAPER variants, and write the agreement CSV."""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr

from waper import Waper

from .masks import (
    band_mask, compute_rwp_envelope, zimin_mask,
    edge_pruning_mask, node_amplitude_mask,
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
        extrema_threshold=10,
        min_latitude=20,
        max_latitude=80,
        node_pruning_threshold=node_pruning_threshold,
        edge_pruning_threshold=edge_pruning_threshold,
    )
    w.identify_rwps()
    return w


def compute_zimin_masks(v_da, band, threshold=ZIMIN_THRESHOLD):
    """Stacked (ntime,512,512) bool Zimin reference masks."""
    lon = v_da.longitude.values
    lat = v_da.latitude.values
    masks = np.empty((v_da.time.size, 512, 512), dtype=bool)
    for t in range(v_da.time.size):
        env = compute_rwp_envelope(v_da.isel(time=t).values, (3, 11))
        masks[t] = zimin_mask(env, lon, lat, band, threshold=threshold)
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
    zimin_masks = compute_zimin_masks(v_da, band)

    rows = []

    # Node-amplitude: one base run supplies association graphs; re-threshold per ST.
    base = run_base_waper(v_da)
    assoc = [tsd.association_graph for tsd in base._time_step_data]
    for st in st_grid:
        method_masks = [node_amplitude_mask(g, st, band) for g in assoc]
        rows.append(_aggregate(method_masks, zimin_masks, band, "node_amplitude", st))

    # Edge-pruning: a full WAPER run per GT; read raster_data.
    for gt in gt_grid:
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
