# swap_evaluation.py

import numpy as np
from scipy import stats as scipy_stats

_CRIT_CACHE = {}


# ============================================================================
# Derivative computation helpers
# ============================================================================

def _first_derivatives(surv: np.ndarray, dt: float) -> np.ndarray:
    """Central difference approximation of first derivative."""
    if len(surv) < 3:
        return np.array([])
    return (surv[2:] - surv[:-2]) / (2 * dt)


def _second_derivatives(surv: np.ndarray, dt: float) -> np.ndarray:
    """Central difference approximation of second derivative."""
    if len(surv) < 3:
        return np.array([])
    return (surv[2:] - 2 * surv[1:-1] + surv[:-2]) / (dt ** 2)


# ============================================================================
# Shape preservation tests
# ============================================================================

def _shape_preserved_derivative(orig_d1: np.ndarray, orig_d2: np.ndarray,
                                new_surv: np.ndarray, dt: float, alpha: float) -> bool:
    """
    Original two-tailed t‑test on mean differences of first and second derivatives.
    H₀: mean difference = 0; reject if |t| ≥ critical value.
    """
    # Test first derivative
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
        _CRIT_CACHE[key1] = scipy_stats.t.ppf(1 - alpha/2, n1 - 1)
    t1 = np.mean(diff1) / np.sqrt(var1 / n1)
    if np.abs(t1) >= _CRIT_CACHE[key1]:
        return False

    # Test second derivative
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
        _CRIT_CACHE[key2] = scipy_stats.t.ppf(1 - alpha/2, n2 - 1)
    t2 = np.mean(diff2) / np.sqrt(var2 / n2)
    return np.abs(t2) < _CRIT_CACHE[key2]


def _shape_preserved_tost_pct_max(orig_d1: np.ndarray, orig_d2: np.ndarray,
                                  new_surv: np.ndarray, dt: float,
                                  alpha: float, epsilon_pct: float) -> bool:
    """
    TOST equivalence test on derivative differences.
    H₀: |mean difference| ≥ epsilon, where epsilon = epsilon_pct * (1/dt).
    Reject H₀ if both one‑sided t‑tests are significant.
    """
    eps = epsilon_pct * (1.0 / dt)

    # First derivative
    d1_new = _first_derivatives(new_surv, dt)
    if len(d1_new) < 2 or len(orig_d1) != len(d1_new):
        return False

    diff1 = orig_d1 - d1_new
    n1 = len(diff1)
    mean1 = np.mean(diff1)
    var1 = np.var(diff1, ddof=1)

    if var1 == 0:
        if abs(mean1) >= eps:
            return False
    else:
        se1 = np.sqrt(var1 / n1)
        t_lower = (mean1 + eps) / se1
        t_upper = (mean1 - eps) / se1
        crit = scipy_stats.t.ppf(1 - alpha, n1 - 1)
        if not (t_lower > crit and t_upper < -crit):
            return False

    # Second derivative
    d2_new = _second_derivatives(new_surv, dt)
    if len(d2_new) < 2 or len(orig_d2) != len(d2_new):
        return False

    diff2 = orig_d2 - d2_new
    n2 = len(diff2)
    mean2 = np.mean(diff2)
    var2 = np.var(diff2, ddof=1)

    if var2 == 0:
        if abs(mean2) >= eps:
            return False
    else:
        se2 = np.sqrt(var2 / n2)
        t_lower = (mean2 + eps) / se2
        t_upper = (mean2 - eps) / se2
        crit = scipy_stats.t.ppf(1 - alpha, n2 - 1)
        if not (t_lower > crit and t_upper < -crit):
            return False

    return True


def _shape_preserved_mad_sad(orig_surv: np.ndarray, new_surv: np.ndarray,
                             delta: float = 0.05) -> bool:
    """
    MAD (maximum absolute deviation) and SAD (sum of absolute deviations)
    thresholds on survival values.
    Requires: MAD < delta and SAD < delta * len(orig_surv).
    """
    if len(orig_surv) != len(new_surv):
        return False
    diffs = np.abs(orig_surv - new_surv)
    mad = np.max(diffs)
    sad = np.sum(diffs)
    return mad < delta and sad < delta * len(orig_surv)


# ============================================================================
# Public dispatcher for shape preservation
# ============================================================================

def is_shape_preserved(orig_surv: np.ndarray, new_surv: np.ndarray,
                       dt: float, alpha: float,
                       test_type: str = 'derivative',
                       delta_shape: float = 0.05) -> bool:
    """
    Dispatch to the selected shape preservation test.

    Parameters
    ----------
    test_type : {'derivative', 'derivative_tost_pct_max', 'mad_sad'}
        - 'derivative'               : two-tailed t‑test on derivative differences
        - 'derivative_tost_pct_max'  : TOST equivalence with epsilon = delta_shape/dt
        - 'mad_sad'                  : absolute deviation thresholds on survival values
    delta_shape : float
        Tolerance parameter. For 'derivative_tost_pct_max', it is the percentage
        of the maximum possible slope (1/dt). For 'mad_sad', it is the absolute
        threshold for MAD and SAD.
    """
    if test_type == 'derivative':
        orig_d1 = _first_derivatives(orig_surv, dt)
        orig_d2 = _second_derivatives(orig_surv, dt)
        return _shape_preserved_derivative(orig_d1, orig_d2, new_surv, dt, alpha)

    elif test_type == 'mad_sad':
        return _shape_preserved_mad_sad(orig_surv, new_surv, delta_shape)

    elif test_type == 'derivative_tost_pct_max':
        orig_d1 = _first_derivatives(orig_surv, dt)
        orig_d2 = _second_derivatives(orig_surv, dt)
        return _shape_preserved_tost_pct_max(orig_d1, orig_d2, new_surv, dt, alpha, delta_shape)

    else:
        raise ValueError(f"Unknown shape test: {test_type}. "
                         f"Options: 'derivative', 'mad_sad', 'derivative_tost_pct_max'")


# ============================================================================
# Continuous shape deviation scores (fallback)
# ============================================================================

def shape_deviation(orig_d1: np.ndarray, orig_d2: np.ndarray,
                    new_surv: np.ndarray, dt: float) -> float:
    """Sum of absolute t‑statistics for first and second derivative differences."""
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


def shape_deviation_from_surv(orig_surv: np.ndarray, new_surv: np.ndarray, dt: float) -> float:
    """Convenience wrapper for shape_deviation using survival curves directly."""
    orig_d1 = _first_derivatives(orig_surv, dt)
    orig_d2 = _second_derivatives(orig_surv, dt)
    return shape_deviation(orig_d1, orig_d2, new_surv, dt)


# Alias for backward compatibility
shape_deviation_score = shape_deviation_from_surv


# ============================================================================
# Absolute deviation metrics
# ============================================================================

def compute_absolute_deviation_metrics(orig_surv: np.ndarray, new_surv: np.ndarray) -> dict:
    """
    Compute MAD (maximum absolute deviation), SAD (sum of absolute deviations),
    and MAE (mean absolute error) between two survival curves.
    """
    if len(orig_surv) != len(new_surv):
        return {'mad': np.nan, 'sad': np.nan, 'mae': np.nan}
    diffs = np.abs(orig_surv - new_surv)
    k = len(diffs)
    return {
        'mad': float(np.max(diffs)),
        'sad': float(np.sum(diffs)),
        'mae': float(np.sum(diffs) / k)
    }


# ============================================================================
# Legacy alias (kept for compatibility)
# ============================================================================

def _shape_preserved(orig_d1, orig_d2, new_surv, dt, alpha):
    """Legacy alias for _shape_preserved_derivative."""
    return _shape_preserved_derivative(orig_d1, orig_d2, new_surv, dt, alpha)