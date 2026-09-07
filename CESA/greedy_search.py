import numpy as np
import pandas as pd
from typing import Tuple, List, Set, Optional, Any, Dict
from swap_evaluation import check_shape_only
from survival_utils import find_closest_time_index

def greedy_search(
    actionable_set: Set[str],
    current_window: pd.DataFrame,
    candidate_window: pd.DataFrame,
    original_survival: np.ndarray,
    time_points: np.ndarray,
    dt: float,
    model,
    delta_1: float,
    alpha: float,
    tau: float,
    f_max: Optional[int],
    explain: List[str],
    current_survival: np.ndarray,
    preserve_shape: bool = True,
    fallback_list: Optional[List[Any]] = None,
    stats: Optional[Dict] = None
) -> Tuple[bool, np.ndarray, List[str], pd.DataFrame]:
    j = find_closest_time_index(time_points, tau)
    used = set(explain)

    if preserve_shape:
        from swap_evaluation import _first_derivatives, _second_derivatives
        orig_d1 = _first_derivatives(original_survival, dt)
        orig_d2 = _second_derivatives(original_survival, dt)
    else:
        orig_d1 = orig_d2 = None

    while current_survival[j] - original_survival[j] <= delta_1 and len(explain) < (f_max or float('inf')):
        best_gain = -np.inf
        best_swap = None
        target_reached = False

        for feat in actionable_set - used:
            res = check_shape_only(
                feat, current_window, candidate_window,
                original_survival, time_points, dt, model, alpha,
                actionable_set, used,
                orig_d1, orig_d2,
                preserve_shape=preserve_shape,
                stats=stats
            )
            if res is not None:
                new_surv, feat_name, shape_ok, dev = res
                gain = new_surv[j] - original_survival[j]

                if gain > delta_1:
                    if shape_ok:
                        current_window[feat_name] = candidate_window[feat_name].values
                        used.add(feat_name)
                        explain.append(feat_name)
                        current_survival = new_surv
                        target_reached = True
                        break
                    else:
                        if fallback_list is not None:
                            donor_vals = candidate_window[feat_name].values.copy()
                            fallback_list.append((dev, feat_name, donor_vals, new_surv))
                else:
                    if shape_ok and gain > best_gain:
                        best_gain = gain
                        best_swap = (feat_name, new_surv)

        if target_reached:
            break
        if best_swap is None:
            break

        feat_name, new_surv = best_swap
        current_window[feat_name] = candidate_window[feat_name].values
        used.add(feat_name)
        explain.append(feat_name)
        current_survival = new_surv

    reached = (current_survival[j] - original_survival[j] > delta_1)
    return reached, current_survival, explain, current_window