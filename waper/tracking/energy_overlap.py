"""Energy-weighted overlap between consecutive-timestep RWP rasters.

The association weight focuses on the energetic cores: each feature's "size" is
its summed energy, and the overlap between two features is the summed geometric
mean of their per-pixel energies over the pixels they share. Compared with binary
pixel overlap, the weak periphery contributes ~0, so a moving core changes the
weight even when the broad footprints still overlap.
"""
import numpy as np


def feature_energies(feature_raster, energy_raster):
    """{feature_id: total energy} for every non-zero feature."""
    ids = np.unique(feature_raster)
    ids = ids[ids != 0]
    return {int(i): float(energy_raster[feature_raster == i].sum()) for i in ids}


def overlap_energies(prev_features, prev_energy, curr_features, curr_energy):
    """{(prev_id, curr_id): Σ sqrt(E_prev * E_curr)} over overlapping pixels."""
    out = {}
    prev_ids = np.unique(prev_features)
    prev_ids = prev_ids[prev_ids != 0]
    for a in prev_ids:
        amask = prev_features == a
        curr_here = np.unique(curr_features[amask])
        curr_here = curr_here[curr_here != 0]
        for b in curr_here:
            m = amask & (curr_features == b)
            e = float(np.sqrt(prev_energy[m] * curr_energy[m]).sum())
            if e > 0:
                out[(int(a), int(b))] = e
    return out
