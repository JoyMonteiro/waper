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
