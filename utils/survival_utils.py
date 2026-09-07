# survival_utils.py

import numpy as np
import pandas as pd
from typing import List, Tuple
import time


# ============================================================================
# Survival curve prediction
# ============================================================================

def get_survival_curve(model, window: pd.DataFrame, time_points: np.ndarray,
                       stats=None) -> np.ndarray:
    """
    Predict the survival curve for a given window at the specified time points.

    Parameters
    ----------
    model : survival model
        A fitted survival model with a `.predict()` method.
    window : pd.DataFrame
        The input window (time series slice).
    time_points : np.ndarray
        The time grid at which survival probabilities are evaluated.
    stats : dict, optional
        If provided, collects the number of model calls and total inference time.

    Returns
    -------
    np.ndarray
        Survival probabilities of shape (len(time_points),).
    """
    window_num = window.select_dtypes(include=[np.number]).copy()
    window_num['label'] = 0
    window_num['event'] = 0
    window_num['vehicle_id'] = 0
    t_start = time.perf_counter()
    try:
        result = model.predict(window_num, "train", event_data=None)
    except Exception as e:
        print(e)
        result = model.predict(window_num.values, "train", event_data=None)
    elapsed = time.perf_counter() - t_start
    if stats is not None:
        stats['num_model_calls'] = stats.get('num_model_calls', 0) + 1
        stats['total_model_time'] = stats.get('total_model_time', 0.0) + elapsed
    return result[0, :]


# ============================================================================
# Window extraction from dataset
# ============================================================================

def extract_windows_from_dataset(dataset: dict, seq_length: int) -> Tuple[List[pd.DataFrame], List[float], List[int]]:
    """
    Extract sliding windows of length seq_length from the historic data of a dataset.

    Parameters
    ----------
    dataset : dict
        Dictionary containing 'historic_data' (list of DataFrames) and 'anomaly_labels' (list of label arrays).
    seq_length : int
        Window length.

    Returns
    -------
    windows : list of pd.DataFrame
        List of extracted windows.
    ruls : list of float
        RUL values (last element of each window's label).
    events : list of int
        Event indicators (last element of each window's label).
    """
    df = dataset['historic_data'][0].select_dtypes(include=[np.number]).copy()
    labels = dataset['anomaly_labels'][0]
    windows, ruls, events = [], [], []
    for i in range(len(df) - seq_length + 1):
        windows.append(df.iloc[i:i + seq_length])
        rul_end, event_end = labels[i + seq_length - 1][0], labels[i + seq_length - 1][1]
        ruls.append(rul_end)
        events.append(event_end)
    return windows, ruls, events


# ============================================================================
# Correlation vector and flattening utilities
# ============================================================================

def compute_correlation_vector(window: pd.DataFrame) -> np.ndarray:
    """
    Compute the upper triangular part of the correlation matrix of a window,
    handling constant columns robustly.

    - If both columns are constant: correlation is +1 if they are equal, else -1.
    - If only one is constant: correlation is 0.
    - Otherwise: Pearson correlation (NaN becomes 0).

    Returns
    -------
    np.ndarray
        Flattened upper triangular part (excluding diagonal).
    """
    n = window.shape[1]
    corr_mat = np.ones((n, n))
    variances = window.var().values
    is_constant = variances < 1e-12
    constant_values = window.iloc[0].values

    for i in range(n):
        for j in range(i + 1, n):
            if is_constant[i] and is_constant[j]:
                corr_mat[i, j] = 1.0 if constant_values[i] == constant_values[j] else -1.0
            elif is_constant[i] or is_constant[j]:
                corr_mat[i, j] = 0.0
            else:
                col_i = window.iloc[:, i].values
                col_j = window.iloc[:, j].values
                corr = np.corrcoef(col_i, col_j)[0, 1]
                corr_mat[i, j] = corr if not np.isnan(corr) else 0.0
    return corr_mat[np.triu_indices_from(corr_mat, k=1)]


def flatten_window(window: pd.DataFrame) -> np.ndarray:
    """Flatten a window DataFrame into a 1D numpy array (row‑major order)."""
    return window.values.ravel()


# ============================================================================
# Time point utilities
# ============================================================================

def find_closest_time_index(time_points: np.ndarray, target: float) -> int:
    """Return the index of the time point closest to the target value."""
    return np.argmin(np.abs(time_points - target))


def calculate_tau(rho, time_points: np.ndarray, orig_surv: np.ndarray):
    """
    Determine the time point tau based on the given rho.

    If rho is the string "change_point", tau is set to the time point where
    the survival curve has its steepest drop (minimum first derivative).
    Otherwise, rho is interpreted as a fraction of the total time horizon,
    and tau = time_points[int(rho * len(time_points))].
    """
    if rho == "change_point":
        diff_1 = np.diff(orig_surv)
        p_min = np.argmin(diff_1)
        tau = time_points[p_min]
    else:
        tau = time_points[int(rho * len(time_points))]
    return tau


def calculate_delta_1(tau, orig_surv: np.ndarray, d1, time_points: np.ndarray):
    """
    Compute the required survival gain delta_1 at tau.

    delta_1 = (1 - S0[tau]) * d1, where S0[tau] is the original survival at tau.
    d1 is a percentage of the remaining survival gap.
    """
    j = find_closest_time_index(time_points, tau)
    delta_1 = (1 - orig_surv[j]) * d1
    return delta_1


# ============================================================================
# RUL estimation from survival curve
# ============================================================================

def get_rul(model, window: pd.DataFrame, time_points: np.ndarray, source: str = "test") -> float:
    """
    Estimate the Remaining Useful Life (RUL) as the earliest time when
    the survival probability drops below the model's threshold (th_to_rul).
    """
    survival_curve = get_survival_curve(model, window, time_points)
    result = [x for x, s in zip(model.time_points, survival_curve) if s < model.th_to_rul]
    if len(result) > 0:
        rul = result[0]
    else:
        rul = model.time_points[-1]
    return rul


# ============================================================================
# Robust correlation matrix (NaN‑free)
# ============================================================================

def robust_corr(df: pd.DataFrame) -> np.ndarray:
    """
    Compute a correlation matrix that never contains NaN.

    - If both columns are constant: correlation = +1 if equal, else -1.
    - If one column is constant: correlation = 0.
    - Otherwise: Pearson correlation, with NaN replaced by 0.
    """
    n = df.shape[1]
    corr_mat = np.ones((n, n))
    variances = df.var().values
    is_constant = variances < 1e-12
    constant_values = df.iloc[0].values

    for i in range(n):
        for j in range(i + 1, n):
            if is_constant[i] and is_constant[j]:
                corr_mat[i, j] = 1.0 if constant_values[i] == constant_values[j] else -1.0
            elif is_constant[i] or is_constant[j]:
                corr_mat[i, j] = 0.0
            else:
                col_i = df.iloc[:, i].values
                col_j = df.iloc[:, j].values
                corr = np.corrcoef(col_i, col_j)[0, 1]
                corr_mat[i, j] = corr if not np.isnan(corr) else 0.0
            corr_mat[j, i] = corr_mat[i, j]
    return corr_mat


# ============================================================================
# MLflow wrapper for survival models
# ============================================================================

class mlflow_rapper:
    """
    Wrapper class to adapt an MLflow‑loaded pipeline to the expected interface.
    It exposes attributes like seq_length, time_points, th_to_rul, etc.,
    and provides a `.predict()` method that delegates to the underlying model.
    """
    def __init__(self, loaded_pipeline):
        train_source = "train"
        self.seq_length = loaded_pipeline._model_impl.python_model.method.seq_length
        self.time_points = loaded_pipeline._model_impl.python_model.method.avail_times_per_source[train_source]
        self.th_to_rul = loaded_pipeline._model_impl.python_model.thresholder.threshold_value
        self.avail_times_per_source = loaded_pipeline._model_impl.python_model.method.avail_times_per_source
        self.model_per_source = loaded_pipeline._model_impl.python_model.method.model_per_source
        self.model = loaded_pipeline._model_impl.python_model.method

    def predict(self, target_data, source="train", event_data=None):
        preds = self.model.predict(target_data, source, event_data)
        return preds[-1]