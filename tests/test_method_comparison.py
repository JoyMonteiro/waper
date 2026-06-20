import numpy as np
from scripts.method_comparison.metrics import (
    iou, disagreement_decomposition, detection_agreement,
)


def test_iou_disjoint_is_zero():
    a = np.zeros((4, 4), bool); a[0, 0] = True
    b = np.zeros((4, 4), bool); b[3, 3] = True
    assert iou(a, b) == 0.0


def test_iou_identical_is_one():
    a = np.zeros((4, 4), bool); a[1:3, 1:3] = True
    assert iou(a, a.copy()) == 1.0


def test_iou_both_empty_is_one():
    a = np.zeros((4, 4), bool)
    assert iou(a, a.copy()) == 1.0


def test_iou_contained():
    a = np.zeros((4, 4), bool); a[0:2, 0:2] = True   # 4 cells
    b = np.zeros((4, 4), bool); b[0, 0] = True        # 1 cell, subset of a
    # intersection 1, union 4
    assert iou(a, b) == 0.25


def test_disagreement_decomposition():
    band = np.ones((4, 4), bool)            # 16 cells
    method = np.zeros((4, 4), bool); method[0, :] = True   # 4 cells
    ref = np.zeros((4, 4), bool); ref[0, 0] = True; ref[3, 3] = True  # 2 cells
    # method_only = method & ~ref within band = 3 cells -> 3/16
    # ref_only = ref & ~method within band = 1 cell (3,3) -> 1/16
    m_only, r_only = disagreement_decomposition(method, ref, band)
    assert abs(m_only - 3 / 16) < 1e-9
    assert abs(r_only - 1 / 16) < 1e-9


def test_detection_agreement():
    empty = np.zeros((4, 4), bool)
    nonempty = np.zeros((4, 4), bool); nonempty[0, 0] = True
    assert detection_agreement(nonempty, nonempty.copy()) is True
    assert detection_agreement(empty, empty.copy()) is True
    assert detection_agreement(nonempty, empty) is False
