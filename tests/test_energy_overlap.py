import numpy as np
from waper.tracking.energy_overlap import feature_energies, overlap_energies

def test_feature_energies_sums_per_feature():
    F = np.array([[0, 1, 1], [2, 2, 0]])
    E = np.array([[0.0, 3.0, 1.0], [5.0, 5.0, 0.0]])
    assert feature_energies(F, E) == {1: 4.0, 2: 10.0}

def test_overlap_energies_geometric_mean_over_overlap():
    Fp = np.array([[1, 1, 0]]); Ep = np.array([[4.0, 4.0, 0.0]])
    Fc = np.array([[1, 0, 0]]); Ec = np.array([[9.0, 0.0, 0.0]])
    # overlap only at pixel (0,0): sqrt(4*9) = 6
    assert overlap_energies(Fp, Ep, Fc, Ec) == {(1, 1): 6.0}

def test_overlap_energies_no_overlap_empty():
    Fp = np.array([[1, 0]]); Ep = np.array([[4.0, 0.0]])
    Fc = np.array([[0, 2]]); Ec = np.array([[0.0, 9.0]])
    assert overlap_energies(Fp, Ep, Fc, Ec) == {}
