import numpy as np
import pandas as pd
from typing import List, Tuple
import time

def get_survival_curve(model, window: pd.DataFrame, time_points: np.ndarray,
                       stats = None) -> np.ndarray:
    """Return survival probabilities for a window at given time points."""
    # Ensure only numeric columns (RSF may choke on strings)
    window_num = window.select_dtypes(include=[np.number]).copy()
    # Add required columns with numeric values (avoid strings)
    window_num['label'] = 0
    window_num['event'] = 0
    window_num['vehicle_id'] = 0
    t_start = time.perf_counter()
    try:
        result = model.predict(window_num, "train", event_data=None)
    except Exception as e:
        print(e)
        # Fallback: try converting to numpy array
        result = model.predict(window_num.values, "train", event_data=None)
    elapsed = time.perf_counter() - t_start
    if stats is not None:
        stats['num_model_calls'] = stats.get('num_model_calls', 0) + 1
        stats['total_model_time'] = stats.get('total_model_time', 0.0) + elapsed
    return result[0, :]

def extract_windows_from_dataset(dataset: dict, seq_length: int) -> Tuple[List[pd.DataFrame], List[float], List[int]]:
    """Create windows of length seq_length from historic_data, returning windows, RUL, and event indicators."""
    df = dataset['historic_data'][0].select_dtypes(include=[np.number]).copy()
    labels = dataset['anomaly_labels'][0]
    windows, ruls, events = [], [], []
    for i in range(len(df) - seq_length + 1):
        windows.append(df.iloc[i:i + seq_length])
        rul_end, event_end = labels[i + seq_length - 1][0], labels[i + seq_length - 1][1]
        ruls.append(rul_end)
        events.append(event_end)
    return windows, ruls, events

def compute_correlation_vector(window: pd.DataFrame) -> np.ndarray:
    """
    Upper triangular part of the correlation matrix, handling constant columns.
    For pairs where both columns are constant:
        - returns 1 if constants equal
        - returns -1 if constants opposite
    For pairs where only one column is constant: returns 0.
    """
    n = window.shape[1]
    corr_mat = np.ones((n, n))  # diagonal = 1
    variances = window.var().values
    is_constant = variances < 1e-12
    constant_values = window.iloc[0].values  # any row, all same if constant

    for i in range(n):
        for j in range(i+1, n):
            if is_constant[i] and is_constant[j]:
                # Both constant: +1 if same value, -1 otherwise
                corr_mat[i, j] = 1.0 if constant_values[i] == constant_values[j] else -1.0
            elif is_constant[i] or is_constant[j]:
                # One constant, one varying → no correlation
                corr_mat[i, j] = 0.0
            else:
                # Normal case: Pearson correlation
                col_i = window.iloc[:, i].values
                col_j = window.iloc[:, j].values
                corr = np.corrcoef(col_i, col_j)[0, 1]
                corr_mat[i, j] = corr if not np.isnan(corr) else 0.0

    # Return upper triangular part (excluding diagonal)
    return corr_mat[np.triu_indices_from(corr_mat, k=1)]

def flatten_window(window: pd.DataFrame) -> np.ndarray:
    """Flatten a window to a 1D array."""
    return window.values.ravel()

def find_closest_time_index(time_points: np.ndarray, target: float) -> int:
    """Index of the time point closest to target."""
    return np.argmin(np.abs(time_points - target))

def calculate_tau(rho, time_points: np.ndarray, orig_surv: np.ndarray):
    if rho == "change_point":
        diff_1 = np.diff(orig_surv)
        p_min = np.argmin(diff_1)
        tau = time_points[p_min]
    else:
        tau = time_points[int(rho*len(time_points))]
    return tau

def calculate_delta_1(tau, orig_surv: np.ndarray, d1, time_points: np.ndarray):
    j = find_closest_time_index(time_points, tau)
    delta_1 = (1 - orig_surv[j]) * d1
    return delta_1

def get_rul(model, window: pd.DataFrame, time_points: np.ndarray, source: str = "test") -> float:
    """Return RUL as first time when survival drops below threshold."""
    survival_curve = get_survival_curve(model, window, time_points)
    result = [x for x, s in zip(model.time_points, survival_curve) if s < model.th_to_rul]
    if len(result) > 0:
        rul = result[0]
    else:
        rul = model.time_points[-1]
    return rul

def robust_corr(df: pd.DataFrame) -> np.ndarray:
    """
    Compute a correlation matrix that never contains NaN.
    For constant columns, correlation with another constant column is:
        +1 if constants equal,
        -1 if constants opposite.
    For constant vs non-constant, correlation is 0.
    """
    n = df.shape[1]
    corr_mat = np.ones((n, n))
    variances = df.var().values
    is_constant = variances < 1e-12
    constant_values = df.iloc[0].values

    for i in range(n):
        for j in range(i+1, n):
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

class mlflow_rapper:
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