# feature_grouping.py

import numpy as np
import pandas as pd
from typing import List, Set
from collections import Counter
from utils.survival_utils import robust_corr


# ============================================================================
# Feature Grouping Based on Correlation
# ============================================================================

def group_features(window: pd.DataFrame, actionable_set: Set[str], threshold: float = 0.7) -> List[List[str]]:
    """
    Group actionable features that are mutually correlated with absolute
    correlation >= threshold.

    Parameters
    ----------
    window : pd.DataFrame
        A single time window of data.
    actionable_set : Set[str]
        Set of feature names that can be modified.
    threshold : float, default=0.7
        Correlation threshold for grouping. Features with |correlation| >= threshold
        are considered mutually correlated and placed in the same group.
        If threshold > 1.0, no grouping is performed; each feature becomes its own
        singleton group (useful to maximise the number of groups).

    Returns
    -------
    List[List[str]]
        A list of feature groups, where each group is a list of feature names.
    """
    # Use deterministic ordering for reproducibility
    feat_list = sorted(actionable_set)
    if not feat_list:
        return []

    # Special case: threshold > 1.0 means "no grouping" (each feature alone)
    if threshold > 1.0:
        return [[f] for f in feat_list]

    data = window[feat_list]
    corr_mat = robust_corr(data)
    corr = np.abs(corr_mat)
    remaining = set(range(len(feat_list)))
    groups = []

    # Greedy grouping: start with the smallest remaining index and add all
    # features that are correlated above threshold with every member of the group.
    while remaining:
        start = min(remaining)
        group = {start}
        remaining.remove(start)
        changed = True
        while changed:
            changed = False
            to_add = []
            for i in sorted(remaining):
                # Check if feature i is strongly correlated with all current group members
                if all(corr[i, j] >= threshold for j in group):
                    to_add.append(i)
            if to_add:
                changed = True
                for i in to_add:
                    group.add(i)
                    remaining.remove(i)
        groups.append([feat_list[i] for i in sorted(group)])

    return groups