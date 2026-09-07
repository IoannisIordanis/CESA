import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from utils.survival_utils import get_survival_curve, find_closest_time_index
from swap_evaluation import _first_derivatives, _second_derivatives, _shape_preserved, shape_deviation

def _partitions_of(n: int, max_part: int = None) -> List[List[int]]:
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
    prox = np.linalg.norm(test_window[feat].values - candidate_window[feat].values)
    corr = np.abs(np.corrcoef(test_window[feat].values, candidate_window[feat].values)[0, 1])
    if np.isnan(corr):
        corr = 0.0
    return prox, corr

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
    stats: Optional[Dict] = None
) -> Tuple[pd.DataFrame, List[str], List[Dict[frozenset, float]], Optional[Tuple[pd.DataFrame, List[str], float]]]:
    j = find_closest_time_index(time_points, tau)
    S0_d1 = _first_derivatives(S0, dt)
    S0_d2 = _second_derivatives(S0, dt)

    beneficial = [dict() for _ in range(len(candidates))]

    # Precompute feature scores
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
        p_min, p_max = prox_arr.min(), prox_arr.max()
        c_min, c_max = corr_arr.min(), corr_arr.max()
        prox_norm = (prox_arr - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(prox_arr)
        corr_norm = (corr_arr - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(corr_arr)
        scores = ((1 - prox_norm) + corr_norm) / 2.0
        feat_score_dict = {feat: scores[i] for i, feat in enumerate(feat_list)}
        candidate_scores.append(feat_score_dict)

    # Track best gain‑achieving solution (shape failed)
    best_gain_win = None
    best_gain_expl = None
    best_gain_deviation = float('inf')

    max_group_size = max((len(g) for gs in groups.values() for g in gs), default=0)

    for s in range(1, max_group_size + 1):
        # Real groups of size s
        for idx, cand in enumerate(candidates):
            s_groups = [g for g in groups[idx] if len(g) == s]
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
                    if _shape_preserved(S0_d1, S0_d2, surv, dt, alpha):
                        return win, group, beneficial, None
                    else:
                        dev = shape_deviation(S0_d1, S0_d2, surv, dt)
                        if dev < best_gain_deviation:
                            best_gain_deviation = dev
                            best_gain_win = win.copy()
                            best_gain_expl = group.copy()
                elif gain > 0:
                    key = frozenset(group)
                    if key not in beneficial[idx] or beneficial[idx][key] < gain:
                        beneficial[idx][key] = gain

        # Combinations
        if s > 1:
            for idx, cand in enumerate(candidates):
                groups_idx= [len(gg) for gg in groups[idx] if len(gg) < s]
                partitions = bounded_combinations(groups_idx, s)
                for part in partitions:
                    best_groups = []
                    total_gain = 0.0
                    feasible = True
                    used_features = set()
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
                        if _shape_preserved(S0_d1, S0_d2, surv, dt, alpha):
                            return win, list(combined), beneficial, None
                        else:
                            dev = shape_deviation(S0_d1, S0_d2, surv, dt)
                            if dev < best_gain_deviation:
                                best_gain_deviation = dev
                                best_gain_win = win.copy()
                                best_gain_expl = list(combined)

    if best_gain_win is not None:
        return best_gain_win, best_gain_expl, beneficial, (best_gain_win, best_gain_expl, best_gain_deviation)
    return test_window, [], beneficial, None


from collections import Counter


def bounded_combinations(group_idx, s):
    counts = Counter(group_idx)
    items = list(counts.items())  # (value, max_count)

    res = []

    def dfs(i, remaining, path):
        if remaining == 0:
            res.append(path.copy())
            return
        if i == len(items) or remaining < 0:
            return

        value, max_count = items[i]

        # try taking k copies of this value
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