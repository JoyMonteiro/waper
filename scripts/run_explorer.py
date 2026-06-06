#!/usr/bin/env python
"""
Run the WAPER identification and tracking pipeline on a netCDF dataset,
save the results to a temporary Parquet catalogue, and launch the
interactive HoloViz RWPExplorer dashboard.

Usage:
    python scripts/run_explorer.py --dataset datasets/souders_v_1.nc
"""
import argparse
import os
import sys
import tempfile
import xarray as xr
import panel as pn

# Add repo root to path so waper is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from waper import Waper
from waper.io.catalogue import save_catalogue, load_catalogue
from waper.interface.explorer import RWPExplorer

# Standard WAPER kwargs from datasets/visualize.py
WAPER_KWARGS = dict(
    scalar_name="v",
    latitude_label="latitude",
    longitude_label="longitude",
    time_label="time",
    clip_value=2,
    extrema_threshold=10,
    min_latitude=20,
    max_latitude=80,
    node_pruning_threshold=20,
    edge_pruning_threshold=0.02,
    max_edge_weight=1,
    track_pruning_threshold=0.3,
    penalty_length_scale_km=4000,
)

def main():
    parser = argparse.ArgumentParser(description="Run WAPER pipeline and launch RWPExplorer.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/souders_v_1.nc",
        help="Path to the netCDF dataset (default: datasets/souders_v_1.nc)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5006,
        help="Port to serve the dashboard on (default: 5006)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file '{args.dataset}' not found.")
        sys.exit(1)

    print(f"Loading dataset from: {args.dataset} ...")
    ds = xr.open_dataset(args.dataset)
    
    # Check dimensions/vars
    for dim in ["time", "latitude", "longitude"]:
        if dim not in ds.dims:
            print(f"Error: Dimension '{dim}' not found in the dataset.")
            sys.exit(1)
            
    if "v" not in ds.data_vars:
        print("Error: Variable 'v' (meridional wind) not found in the dataset.")
        sys.exit(1)

    print("Running WAPER identification pipeline ...")
    w = Waper(ds, **WAPER_KWARGS)
    w.identify_rwps()

    print("Running WAPER tracking pipeline ...")
    w.track_rwps()

    # Determine time step settings
    n_times = len(ds["time"])
    dt_hours = 6
    if n_times > 1:
        time_diff = ds["time"].values[1] - ds["time"].values[0]
        dt_hours = float(time_diff / np.timedelta64(1, "h")) if "timedelta" in str(type(time_diff)) else 6

    # Create temporary directory for the Parquet catalogue
    tmpdir = tempfile.TemporaryDirectory()
    print(f"Saving temporary Parquet catalogue to: {tmpdir.name} ...")
    save_catalogue(w, tmpdir.name, meta={"units": "m s**-1", "dt_hours": dt_hours})

    print("Loading Parquet catalogue ...")
    cat = load_catalogue(tmpdir.name)

    print("Initializing RWPExplorer dashboard ...")
    # Extract the meridional wind DataArray for background rendering
    field_da = ds["v"]
    
    # Create the explorer
    explorer = RWPExplorer(cat, n_times=n_times, field_da=field_da)
    
    # Construct a beautiful template
    title = f"WAPER RWP Explorer - {os.path.basename(args.dataset)}"
    template = pn.template.MaterialTemplate(
        title=title,
        sidebar=[explorer.__panel__()[0]], # sidebar column
        main=[explorer.__panel__()[1]],    # main map column
    )

    print(f"\nServing RWPExplorer at http://localhost:{args.port} ...")
    print("Press Ctrl+C to stop the server.")
    
    # Serve the app template
    pn.serve(template, port=args.port, show=True)
    
    # Cleanup temp directory on termination
    tmpdir.cleanup()

if __name__ == "__main__":
    import numpy as np # import here to satisfy script run dependencies
    main()
