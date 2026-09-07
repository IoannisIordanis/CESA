import numpy as np
from scipy import stats as scipy_stats

_CRIT_CACHE = {}

def _first_derivatives(surv: np.ndarray, dt: float) -> np.ndarray:
    if len(surv) < 3:
        return np.array([])
    return (surv[2:] - surv[:-2]) / (2 * dt)

def _second_derivatives(surv: np.ndarray, dt: float) -> np.ndarray:
    if len(surv) < 3:
        return np.array([])
    return (surv[2:] - 2 * surv[1:-1] + surv[:-2]) / (dt ** 2)

def _shape_preserved(orig_d1: np.ndarray, orig_d2: np.ndarray,
                     new_surv: np.ndarray, dt: float, alpha: float) -> bool:
    d1_new = _first_derivatives(new_surv, dt)
    if len(d1_new) < 2:
        return False
    diff1 = orig_d1 - d1_new
    n1 = len(diff1)
    var1 = np.var(diff1, ddof=1)
    if var1 == 0:
        if np.mean(diff1) != 0:
            return False
    else:
        key1 = (n1, alpha)
        if key1 not in _CRIT_CACHE:
            _CRIT_CACHE[key1] = scipy_stats.t.ppf(1 - alpha/2, n1 - 1)
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
        if np.mean(diff2) != 0:
            return False
        return True
    else:
        key2 = (n2, alpha)
        if key2 not in _CRIT_CACHE:
            _CRIT_CACHE[key2] = scipy_stats.t.ppf(1 - alpha/2, n2 - 1)
        t2 = np.mean(diff2) / np.sqrt(var2 / n2)
        return np.abs(t2) < _CRIT_CACHE[key2]

def shape_deviation(orig_d1, orig_d2, new_surv, dt):
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
        elif np.mean(diff1) != 0:
            dev += float('inf')
    else:
        dev += float('inf')
        
    if len(d2_new) >= 2 and len(orig_d2) == len(d2_new):
        diff2 = orig_d2 - d2_new
        n2 = len(diff2)
        var2 = np.var(diff2, ddof=1)
        if var2 > 0:
            t2 = np.mean(diff2) / np.sqrt(var2 / n2)
            dev += abs(t2)
        elif np.mean(diff2) != 0:
            dev += float('inf')
    else:
        dev += float('inf')
        
    return dev