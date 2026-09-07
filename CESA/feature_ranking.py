# feature_ranking.py

import numpy as np
import pandas as pd
from typing import List, Set
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


# ============================================================================
# Helper for correlation (handles NaN)
# ============================================================================

def _safe_correlation(a, b):
    """Compute Pearson correlation, returning 0.0 if the result is NaN."""
    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        return 0.0
    return corr


# ============================================================================
# Feature ranking for a candidate window
# ============================================================================

def rank_features(
    candidate_window: pd.DataFrame,
    test_window: pd.DataFrame,
    actionable_set: Set[str],
    use_corr: bool = True,
    use_prox: bool = True
) -> List[str]:
    """
    Rank actionable features by their suitability for modification when
    transforming the test window toward the candidate window.

    Parameters
    ----------
    candidate_window : pd.DataFrame
        The reference window (e.g., a donor) to which we compare.
    test_window : pd.DataFrame
        The original test window.
    actionable_set : Set[str]
        Features that can be modified.
    use_corr : bool, default=True
        If True, use absolute correlation between feature values in the two windows
        as a ranking criterion (higher correlation is better).
    use_prox : bool, default=True
        If True, use Euclidean distance between feature columns (lower distance is better).

    Returns
    -------
    List[str]
        Features sorted from most to least promising for modification.
        If neither use_corr nor use_prox is True, returns features in alphabetical order.
    """
    # Deterministic ordering for reproducibility
    features = sorted(actionable_set)
    if not features:
        return []

    # If no ranking criteria are used, return the sorted list
    if not use_corr and not use_prox:
        return features

    prox_vals = []
    corr_vals = []

    for feat in features:
        prox = np.linalg.norm(test_window[feat].values - candidate_window[feat].values)
        corr = abs(_safe_correlation(test_window[feat].values, candidate_window[feat].values))
        prox_vals.append(prox)
        corr_vals.append(corr)

    # Ranking based solely on correlation (higher is better)
    if use_corr and not use_prox:
        order = np.argsort(corr_vals)[::-1]
        return [features[i] for i in order]

    # Ranking based solely on proximity (lower is better)
    if use_prox and not use_corr:
        order = np.argsort(prox_vals)
        return [features[i] for i in order]

    # Combined ranking: normalise both metrics and compute a score
    # where lower proximity and higher correlation give higher score.
    prox_arr = np.array(prox_vals)
    corr_arr = np.array(corr_vals)
    p_min, p_max = prox_arr.min(), prox_arr.max()
    c_min, c_max = corr_arr.min(), corr_arr.max()
    prox_norm = (prox_arr - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(prox_arr)
    corr_norm = (corr_arr - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(corr_arr)

    # Score: average of normalised inverse proximity and normalised correlation
    scores = ((1 - prox_norm) + corr_norm) / 2.0
    order = np.argsort(scores)[::-1]
    return [features[i] for i in order]