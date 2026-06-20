"""Set-algebra agreement metrics for boolean RWP masks on the shared grid."""
import numpy as np


def iou(a, b):
    """Intersection-over-union of two boolean masks. 1.0 if both are empty."""
    a = a.astype(bool); b = b.astype(bool)
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0
    return inter / union


def disagreement_decomposition(method, ref, band):
    """Return (method_only_frac, ref_only_frac) as fractions of the band area."""
    method = method.astype(bool) & band
    ref = ref.astype(bool) & band
    denom = np.count_nonzero(band)
    if denom == 0:
        return 0.0, 0.0
    method_only = np.count_nonzero(method & ~ref) / denom
    ref_only = np.count_nonzero(ref & ~method) / denom
    return method_only, ref_only


def detection_agreement(method, ref):
    """True iff both masks detect >=1 cell, or both are empty."""
    return bool(np.any(method)) == bool(np.any(ref))
