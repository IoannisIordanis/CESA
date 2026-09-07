# global_shap.py
import numpy as np
import pandas as pd
import shap
from typing import List, Dict, Optional

def _flatten_window(window: pd.DataFrame) -> np.ndarray:
    """Flatten a window to a 1D array."""
    return window.values.ravel()

def _is_rsf_wrapper(model) -> bool:
    """Heuristic: model has 'model_per_source' and its first model is a RandomSurvivalForest."""
    if hasattr(model, 'model_per_source') and model.model_per_source:
        first_model = next(iter(model.model_per_source.values()))
        return 'RandomSurvivalForest' in str(type(first_model))
    return False

def _build_shap_predict_fn(model, time_points, feature_names, seq_len):
    """
    Build a prediction function for SHAP that returns a scalar output
    (survival probability at the median time point) for each flattened window.
    Works for standard survival models and the RSF wrapper.
    """
    median_idx = len(time_points) // 2
    fixed_time = time_points[median_idx]

    # Detect RSF wrapper
    if _is_rsf_wrapper(model):
        # Get the underlying RandomSurvivalForest model (any source, they are identical)
        first_model = next(iter(model.model_per_source.values()))
        # RSF expects input of shape (n_samples, n_features * seq_len)
        def predict_fn(X_2d):
            # X_2d shape: (n_samples, D) where D = seq_len * n_features
            # Call underlying predict_survival_function with return_array=True
            surv_probs = first_model.predict_survival_function(X_2d, return_array=True)  # shape (n_samples, n_times)
            # Extract survival probability at fixed_time for each sample
            # We need the column index closest to fixed_time
            time_col = np.argmin(np.abs(time_points - fixed_time))
            return surv_probs[:, time_col]
        return predict_fn

    # Standard models: use the wrapper's predict method
    def predict_fn(X_2d):
        survs = []
        for row in X_2d:
            # Reconstruct DataFrame
            win_df = pd.DataFrame(row.reshape(seq_len, len(feature_names)), columns=feature_names)
            win_df = win_df.select_dtypes(include=[np.number]).copy()
            win_df['label'] = 0
            win_df['event'] = 0
            win_df['vehicle_id'] = 'test'

            # Determine source for prediction
            if hasattr(model, 'model_per_source'):
                src = next(iter(model.model_per_source))
            else:
                src = 'test'

            try:
                result = model.predict(win_df, src, event_data=None)
                # Handle different output shapes
                if result.ndim == 3:
                    # shape (n_samples, 2, n_times) – first dimension is samples
                    surv = result[0, 0, :]   # survival probabilities of first sample
                elif result.ndim == 2:
                    surv = result[0, :]
                else:
                    surv = result.flatten()
                prob_at_fixed = surv[median_idx]
                survs.append(prob_at_fixed)
            except Exception as e:
                # If prediction fails, return 0.0 (fallback)
                print(f"Warning: SHAP prediction failed: {e}")
                survs.append(0.0)
        return np.array(survs)
    return predict_fn

def compute_global_shap(
    model,
    train_windows: List[pd.DataFrame],
    time_points: np.ndarray,
    background_size: int = 100,
    nsamples: int = 200
) -> Dict[str, float]:
    """
    Compute global SHAP feature importance by averaging local SHAP values
    over a random subset of training windows.

    Works with any survival model (CoxPH, RSF wrapper, DeepHIT, GradientBoostingSurvival)
    as long as a prediction wrapper can be built (see _build_shap_predict_fn).

    Parameters
    ----------
    model : object
        Survival model. Must have either .predict() or be an RSF wrapper.
    train_windows : List[pd.DataFrame]
        List of training windows (each of shape (seq_len, n_features)).
    time_points : np.ndarray
        Array of evaluation times.
    background_size : int, default 100
        Number of windows to use as background for KernelExplainer.
    nsamples : int, default 200
        Number of SHAP kernel evaluations.

    Returns
    -------
    importance : Dict[str, float]
        Dictionary mapping feature name to importance score (normalised to [0,1]).
    """
    if not train_windows:
        raise ValueError("Need at least one training window.")

    feature_names = list(train_windows[0].columns)
    seq_len = train_windows[0].shape[0]
    predict_fn = _build_shap_predict_fn(model, time_points, feature_names, seq_len)

    # Background: random subset of training windows (flattened)
    bg_size = min(background_size, len(train_windows))
    bg_indices = np.random.choice(len(train_windows), bg_size, replace=False)
    background = np.array([_flatten_window(train_windows[i]) for i in bg_indices])

    # Test prediction on background to catch errors early
    try:
        _ = predict_fn(background[:1])
    except Exception as e:
        raise RuntimeError(f"SHAP prediction function test failed: {e}")

    explainer = shap.KernelExplainer(predict_fn, background)

    # Sample a subset of training windows to explain (to keep runtime reasonable)
    sample_size = min(200, len(train_windows))
    sample_indices = np.random.choice(len(train_windows), sample_size, replace=False)
    X_sample = np.array([_flatten_window(train_windows[i]) for i in sample_indices])

    shap_values = explainer.shap_values(X_sample, nsamples=nsamples)

    # shap_values can be a list of arrays (one per output) or a 2D array.
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)                 # (outputs, samples, D)
        shap_values = np.mean(np.abs(shap_values), axis=(0,1))  # average over outputs and samples
    else:
        shap_values = np.mean(np.abs(shap_values), axis=0)  # (D,)

    # Aggregate over time steps: each feature appears in seq_len consecutive positions
    importance = {}
    for i, name in enumerate(feature_names):
        start = i * seq_len
        end = start + seq_len
        importance[name] = float(np.mean(shap_values[start:end]))

    # Normalise to [0,1]
    maxv = max(importance.values()) if importance else 1.0
    if maxv > 0:
        for k in importance:
            importance[k] /= maxv
    return importance