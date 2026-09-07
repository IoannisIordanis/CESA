import numpy as np
import pandas as pd
from typing import List, Set
import warnings

# Suppress the specific RuntimeWarning from numpy.corrcoef when dividing by zero
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

def _safe_correlation(a, b):
    """Compute Pearson correlation, return 0.0 if NaN or constant."""
    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        return 0.0
    return corr

def rank_features(
    candidate_window: pd.DataFrame,
    test_window: pd.DataFrame,
    actionable_set: Set[str]
) -> List[str]:
    """
    Rank actionable features by (1 - normalised proximity) and normalised correlation.
    Returns list of features sorted descending by score.
    No warnings, handles constants.
    """
    features = list(actionable_set)
    if not features:
        return []

    prox_vals = []
    corr_vals = []

    for feat in features:
        prox = np.linalg.norm(test_window[feat].values - candidate_window[feat].values)
        corr = abs(_safe_correlation(test_window[feat].values, candidate_window[feat].values))
        prox_vals.append(prox)
        corr_vals.append(corr)

    prox_arr = np.array(prox_vals)
    corr_arr = np.array(corr_vals)

    p_min, p_max = prox_arr.min(), prox_arr.max()
    c_min, c_max = corr_arr.min(), corr_arr.max()
    prox_norm = (prox_arr - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(prox_arr)
    corr_norm = (corr_arr - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(corr_arr)

    scores = ((1 - prox_norm) + corr_norm) / 2.0
    order = np.argsort(scores)[::-1]
    return [features[i] for i in order]