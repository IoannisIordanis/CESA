import numpy as np
import pandas as pd
from typing import Optional, Tuple, Set
from scipy import stats
from utils.survival_utils import get_survival_curve, find_closest_time_index

def _first_derivatives(surv: np.ndarray, dt: float) -> np.ndarray:
    if len(surv) < 3:
        return np.array([])
    return (surv[2:] - surv[:-2]) / (2 * dt)

def _second_derivatives(surv: np.ndarray, dt: float) -> np.ndarray:
    if len(surv) < 3:
        return np.array([])
    return (surv[2:] - 2 * surv[1:-1] + surv[:-2]) / (dt ** 2)

_CRIT_CACHE = {}

def _shape_preserved(orig_d1: np.ndarray, orig_d2: np.ndarray,
                     new_surv: np.ndarray, dt: float, alpha: float) -> bool:
    d1_new = _first_derivatives(new_surv, dt)
    if len(d1_new) < 2:
        return False
    diff1 = orig_d1 - d1_new
    n1 = len(diff1)
    var1 = np.var(diff1, ddof=1)
    if var1 == 0:
        return False
    key1 = (n1, alpha)
    if key1 not in _CRIT_CACHE:
        _CRIT_CACHE[key1] = stats.t.ppf(1 - alpha/2, n1 - 1)
    t1 = np.mean(diff1) / np.sqrt(var1 / n1)
    if np.abs(t1) >= _CRIT_CACHE[key1]:
        return False

    d2_new = _second_derivatives(new_surv, dt)
    if len(d2_new) < 2:
        return False
    diff2 = orig_d2 - d2_new
    n2 = len(diff2)
    var2 = np.var(diff2, ddof=1)
    if var2 == 0:
        return False
    key2 = (n2, alpha)
    if key2 not in _CRIT_CACHE:
        _CRIT_CACHE[key2] = stats.t.ppf(1 - alpha/2, n2 - 1)
    t2 = np.mean(diff2) / np.sqrt(var2 / n2)
    return np.abs(t2) < _CRIT_CACHE[key2]

def shape_deviation(orig_d1: np.ndarray, orig_d2: np.ndarray,
                    new_surv: np.ndarray, dt: float) -> float:
    d1_new = _first_derivatives(new_surv, dt)
    d2_new = _second_derivatives(new_surv, dt)
    dev = 0.0
    if len(d1_new) >= 2 and len(orig_d1) == len(d1_new):
        diff1 = orig_d1 - d1_new
        n1 = len(diff1)
        var1 = np.var(diff1, ddof=1)
        if var1 > 0:
            t1 = np.mean(diff1) / np.sqrt(var1 / n1)
            dev += abs(t1)
    if len(d2_new) >= 2 and len(orig_d2) == len(d2_new):
        diff2 = orig_d2 - d2_new
        n2 = len(diff2)
        var2 = np.var(diff2, ddof=1)
        if var2 > 0:
            t2 = np.mean(diff2) / np.sqrt(var2 / n2)
            dev += abs(t2)
    return dev

def check_shape_only(
    feature: str,
    current_window: pd.DataFrame,
    candidate_window: pd.DataFrame,
    original_survival: np.ndarray,
    time_points: np.ndarray,
    dt: float,
    model,
    alpha: float,
    actionable_set: Set[str],
    used_features: Set[str],
    orig_d1: np.ndarray,
    orig_d2: np.ndarray,
    preserve_shape: bool = True
) -> Optional[Tuple[np.ndarray, str, bool, float]]:
    """
    Returns (new_surv, feature_name, shape_preserved, deviation)
    """
    if feature not in actionable_set or feature in used_features:
        return None

    cur_vals = current_window[feature].to_numpy()
    cand_vals = candidate_window[feature].to_numpy()
    if np.array_equal(cur_vals, cand_vals):
        return None

    temp = current_window.copy(deep=False)
    temp[feature] = cand_vals
    try:
        new_surv = get_survival_curve(model, temp, time_points)
    except Exception:
        return None

    if not preserve_shape:
        return new_surv, feature, True, 0.0

    dev = shape_deviation(orig_d1, orig_d2, new_surv, dt)
    preserved = _shape_preserved(orig_d1, orig_d2, new_surv, dt, alpha)
    return new_surv, feature, preserved, dev

def evaluate_swap(
    feature: str,
    current_window: pd.DataFrame,
    candidate_window: pd.DataFrame,
    original_survival: np.ndarray,
    time_points: np.ndarray,
    dt: float,
    model,
    delta_1: float,
    alpha: float,
    tau: float,
    actionable_set: Set[str],
    used_features: Set[str],
    orig_d1: np.ndarray,
    orig_d2: np.ndarray,
    preserve_shape: bool = True
) -> Optional[Tuple[np.ndarray, str, bool, float]]:
    """
    Similar to check_shape_only but enforces pointwise gain > delta_1.
    """
    if feature not in actionable_set or feature in used_features:
        return None

    cur_vals = current_window[feature].to_numpy()
    cand_vals = candidate_window[feature].to_numpy()
    if np.array_equal(cur_vals, cand_vals):
        return None

    temp = current_window.copy(deep=False)
    temp[feature] = cand_vals
    try:
        new_surv = get_survival_curve(model, temp, time_points)
    except Exception:
        return None

    j = find_closest_time_index(time_points, tau)
    if new_surv[j] <= original_survival[j] + delta_1:
        return None

    if not preserve_shape:
        return new_surv, feature, True, 0.0

    dev = shape_deviation(orig_d1, orig_d2, new_surv, dt)
    preserved = _shape_preserved(orig_d1, orig_d2, new_surv, dt, alpha)
    return new_surv, feature, preserved, dev