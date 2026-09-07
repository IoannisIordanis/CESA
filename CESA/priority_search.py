# priority_search.py

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
from utils.survival_utils import get_survival_curve, find_closest_time_index
from swap_evaluation import is_shape_preserved, shape_deviation_score


# ============================================================================
# Helper functions for partitions and combinations
# ============================================================================

def _partitions_of(n: int, max_part: int = None) -> List[List[int]]:
    """
    Generate all integer partitions of n with parts not exceeding max_part.
    Returns a list of lists, each representing a partition.
    """
    if max_part is None:
        max_part = n
    result = []
    if n == 0:
        return [[]]
    for first in range(1, min(n, max_part) + 1):
        for rest in _partitions_of(n - first, first):
            result.append([first] + rest)
    return result


def _feature_score(test_window, candidate_window, feat):
    """
    Compute proximity (Euclidean distance) and absolute correlation for a single feature
    between the test and candidate windows. Returns (proximity, correlation).
    """
    prox = np.linalg.norm(test_window[feat].values - candidate_window[feat].values)
    corr = np.abs(np.corrcoef(test_window[feat].values, candidate_window[feat].values)[0, 1])
    if np.isnan(corr):
        corr = 0.0
    return prox, corr


def bounded_combinations(group_idx, s):
    """
    Given a list of group sizes (group_idx) and a target total size s,
    return all combinations of group sizes (each from the list, with repetition
    limited by the multiplicity in group_idx) that sum to s.
    Used to combine partial groups when no single group achieves the required gain.
    """
    counts = Counter(group_idx)
    items = list(counts.items())
    res = []

    def dfs(i, remaining, path):
        if remaining == 0:
            res.append(path.copy())
            return
        if i == len(items) or remaining < 0:
            return
        value, max_count = items[i]
        for k in range(max_count + 1):
            new_sum = remaining - k * value
            if new_sum < 0:
                break
            if k > 0:
                path.extend([value] * k)
            dfs(i + 1, new_sum, path)
            if k > 0:
                for _ in range(k):
                    path.pop()

    dfs(0, s, [])
    return res


# ============================================================================
# Core priority search algorithm
# ============================================================================

def priority_search(
    candidates: List[pd.DataFrame],
    test_window: pd.DataFrame,
    S0: np.ndarray,
    delta_1: float,
    alpha: float,
    tau: float,
    time_points: np.ndarray,
    dt: float,
    model,
    actionable_set: Set[str],
    rankings: Dict[int, List[str]],
    groups: Dict[int, List[List[str]]],
    stats: Optional[Dict] = None,
    use_shape: bool = True,
    use_corr: bool = True,
    use_prox: bool = True,
    shape_test: str = 'derivative',
    shape_delta: float = 0.05
) -> Tuple[pd.DataFrame, List[str], List[Dict[frozenset, float]], Optional[Tuple[pd.DataFrame, List[str], float]]]:
    """
    Priority search over candidates, testing feature groups in order of increasing size.
    It attempts to find a small set of features (within a group) that, when modified,
    achieves the required survival gain while preserving shape.

    Parameters
    ----------
    candidates : list of pd.DataFrame
        List of candidate (donor) windows, already aligned to the test window.
    test_window : pd.DataFrame
        The original test window.
    S0 : np.ndarray
        Survival curve of the test window.
    delta_1 : float
        Required gain at tau.
    alpha : float
        Significance level for shape tests.
    tau : float
        Time point of interest.
    time_points : np.ndarray
        Grid of time points.
    dt : float
        Time step.
    model : survival model
        For predicting survival curves.
    actionable_set : Set[str]
        Features that can be modified.
    rankings : Dict[int, List[str]]
        Ranking of features for each candidate (not used directly in current implementation).
    groups : Dict[int, List[List[str]]]
        Feature groups for each candidate (indexed by candidate index).
    stats : dict, optional
        Dictionary to collect statistics (group sizes, chosen group count, etc.).
    use_shape : bool, default=True
        Whether to enforce shape preservation.
    use_corr : bool, default=True
        Whether correlation is considered when ranking features.
    use_prox : bool, default=True
        Whether proximity is considered when ranking features.
    shape_test : str, default='derivative'
        Which shape preservation test to use.
    shape_delta : float, default=0.05
        Tolerance for shape test.

    Returns
    -------
    Tuple:
        - best_window (pd.DataFrame): the modified window, or test_window if none found.
        - explanation (List[str]): the set of features modified.
        - beneficial (List[Dict[frozenset, float]]): per‑candidate dictionary of beneficial partial groups.
        - fallback (Optional[Tuple]): if a non‑shape‑preserving solution with gain was found,
          returns (win, expl, deviation_score); else None.
    """
    # Record group statistics if requested
    if stats is not None:
        group_counts = [len(g_list) for g_list in groups.values() if g_list is not None]
        if group_counts:
            stats['group_min'] = int(min(group_counts))
            stats['group_max'] = int(max(group_counts))
            stats['group_mean'] = float(np.mean(group_counts))
        else:
            stats['group_min'] = 0
            stats['group_max'] = 0
            stats['group_mean'] = 0.0
        stats['chosen_group_count'] = 0

    j = find_closest_time_index(time_points, tau)

    # beneficial[idx] = { frozenset(group): gain } for groups that gave positive gain but not enough
    beneficial = [dict() for _ in range(len(candidates))]

    # Pre‑compute per‑feature scores for each candidate
    candidate_scores = []
    for idx, cand in enumerate(candidates):
        feat_list = list(actionable_set)
        prox_vals = []
        corr_vals = []
        for feat in feat_list:
            p, c = _feature_score(test_window, cand, feat)
            prox_vals.append(p)
            corr_vals.append(c)
        prox_arr = np.array(prox_vals)
        corr_arr = np.array(corr_vals)
        if use_prox:
            p_min, p_max = prox_arr.min(), prox_arr.max()
            prox_norm = (prox_arr - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(prox_arr)
        else:
            prox_norm = np.zeros_like(prox_arr)
        if use_corr:
            c_min, c_max = corr_arr.min(), corr_arr.max()
            corr_norm = (corr_arr - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(corr_arr)
        else:
            corr_norm = np.zeros_like(corr_arr)

        if use_prox and use_corr:
            scores = ((1 - prox_norm) + corr_norm) / 2.0
        elif use_prox:
            scores = 1 - prox_norm
        elif use_corr:
            scores = corr_norm
        else:
            scores = np.ones(len(feat_list))
        feat_score_dict = {feat: scores[i] for i, feat in enumerate(feat_list)}
        candidate_scores.append(feat_score_dict)

    # Track best non‑shape‑preserving solution as fallback
    best_gain_win = None
    best_gain_expl = None
    best_gain_deviation = float('inf')
    best_gain_idx = -1

    max_group_size = max((len(g) for gs in groups.values() for g in gs), default=0)

    # Search over increasing group sizes
    for s in range(1, max_group_size + 1):
        # --- Single groups of size s ---
        for idx, cand in enumerate(candidates):
            s_groups = [g for g in groups[idx] if len(g) == s]
            # Sort groups by average feature score (descending)
            def group_avg_score(group):
                return np.mean([candidate_scores[idx][feat] for feat in group])
            s_groups.sort(key=group_avg_score, reverse=True)

            for group in s_groups:
                win = test_window.copy()
                for feat in group:
                    if feat in actionable_set:
                        win[feat] = cand[feat].values
                surv = get_survival_curve(model, win, time_points, stats=stats)
                gain = surv[j] - S0[j]

                if gain >= delta_1:
                    if use_shape:
                        shape_ok = is_shape_preserved(S0, surv, dt, alpha,
                                                      shape_test, shape_delta)
                    else:
                        shape_ok = True
                    if shape_ok:
                        if stats is not None:
                            stats['chosen_group_count'] = len(groups[idx])
                        return win, group, beneficial, None
                    else:
                        # Keep this as a fallback candidate
                        dev = shape_deviation_score(S0, surv, dt)
                        if dev < best_gain_deviation:
                            best_gain_deviation = dev
                            best_gain_win = win.copy()
                            best_gain_expl = group.copy()
                            best_gain_idx = idx
                elif gain > 0:
                    # Store beneficial partial group for later combination
                    key = frozenset(group)
                    if key not in beneficial[idx] or beneficial[idx][key] < gain:
                        beneficial[idx][key] = gain

        # --- Combinations of groups (for s > 1) ---
        if s > 1:
            for idx, cand in enumerate(candidates):
                # groups with size < s can be combined to reach size s
                groups_idx = [len(gg) for gg in groups[idx] if len(gg) < s]
                partitions = bounded_combinations(groups_idx, s)

                for part in partitions:
                    best_groups = []
                    total_gain = 0.0
                    feasible = True
                    used_features = set()

                    # For each required size, pick the best stored group of that size
                    for size in part:
                        best = None
                        best_gain = -np.inf
                        for key, gain_val in beneficial[idx].items():
                            if len(key) == size and not (set(key) & used_features):
                                if gain_val > best_gain:
                                    best = key
                                    best_gain = gain_val
                        if best is None:
                            feasible = False
                            break
                        best_groups.append(best)
                        used_features.update(best)
                        total_gain += best_gain

                    if not feasible:
                        continue

                    combined = set()
                    for grp in best_groups:
                        combined.update(grp)
                    win = test_window.copy()
                    for feat in combined:
                        if feat in actionable_set:
                            win[feat] = cand[feat].values
                    surv = get_survival_curve(model, win, time_points, stats=stats)
                    gain = surv[j] - S0[j]

                    if gain >= delta_1:
                        if use_shape:
                            shape_ok = is_shape_preserved(S0, surv, dt, alpha,
                                                          shape_test, shape_delta)
                        else:
                            shape_ok = True
                        if shape_ok:
                            if stats is not None:
                                stats['chosen_group_count'] = len(groups[idx])
                            return win, list(combined), beneficial, None
                        else:
                            dev = shape_deviation_score(S0, surv, dt)
                            if dev < best_gain_deviation:
                                best_gain_deviation = dev
                                best_gain_win = win.copy()
                                best_gain_expl = list(combined)
                                best_gain_idx = idx

    # If no shape‑preserving solution found, return the best fallback (if any)
    if best_gain_win is not None:
        if stats is not None and best_gain_idx != -1:
            stats['chosen_group_count'] = len(groups[best_gain_idx])
        return best_gain_win, best_gain_expl, beneficial, (best_gain_win, best_gain_expl, best_gain_deviation)

    # No solution at all
    if stats is not None:
        stats['chosen_group_count'] = 0
    return test_window, [], beneficial, None