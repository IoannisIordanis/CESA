import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from utils.survival_utils import get_survival_curve, find_closest_time_index
from swap_evaluation import _first_derivatives, _second_derivatives, _shape_preserved, shape_deviation

def _all_combinations_of_beneficial(
    beneficial_dict: Dict[frozenset, float],
    max_total_size: int = 10
) -> List[Tuple[Set[str], float]]:
    groups = list(beneficial_dict.items())
    combos = []
    def backtrack(start_idx, current_set, current_gain, current_features):
        if current_set and len(current_features) <= max_total_size:
            combos.append((set(current_set), current_gain))
        for i in range(start_idx, len(groups)):
            grp, gain = groups[i]
            if not (set(grp) & current_features):
                backtrack(i+1, current_set + [grp], current_gain + gain, current_features.union(grp))
    backtrack(0, [], 0.0, set())
    combos.sort(key=lambda x: x[1], reverse=True)
    return combos

def full_search(
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
    beneficial_records: List[Dict[frozenset, float]],
    stats: Optional[Dict] = None,
    max_total_size: int = 10
) -> Tuple[pd.DataFrame, List[str], Optional[Tuple[pd.DataFrame, List[str], float]]]:
    j = find_closest_time_index(time_points, tau)
    S0_d1 = _first_derivatives(S0, dt)
    S0_d2 = _second_derivatives(S0, dt)

    best_gain_win = None
    best_gain_expl = None
    best_gain_deviation = float('inf')

    for idx, cand in enumerate(candidates):
        if not beneficial_records[idx]:
            continue
        combos = _all_combinations_of_beneficial(beneficial_records[idx], max_total_size)
        for feat_set, _ in combos:
            if not feat_set:
                continue
            win = test_window.copy()
            for feat in feat_set:
                if feat in actionable_set:
                    win[feat] = cand[feat].values
            surv = get_survival_curve(model, win, time_points, stats=stats)
            gain = surv[j] - S0[j]
            if gain >= delta_1:
                if _shape_preserved(S0_d1, S0_d2, surv, dt, alpha):
                    return win, list(feat_set), None
                else:
                    dev = shape_deviation(S0_d1, S0_d2, surv, dt)
                    if dev < best_gain_deviation:
                        best_gain_deviation = dev
                        best_gain_win = win.copy()
                        best_gain_expl = list(feat_set)
    if best_gain_win is not None:
        return best_gain_win, best_gain_expl, (best_gain_win, best_gain_expl, best_gain_deviation)
    return test_window, [], None