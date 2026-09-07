# local_shap.py
import numpy as np
import pandas as pd
import shap
from typing import Dict, List

def compute_local_shap_importance(
    model,
    window: pd.DataFrame,
    time_points: np.ndarray,
    train_windows: List[pd.DataFrame] = None,
    feature_names: List[str] = None,
    nsamples: int = 200
) -> Dict[str, float]:
    """
    Compute local feature importance for a single window using SHAP KernelExplainer.
    
    Parameters
    ----------
    model : object with a predict(DataFrame, source, event_data) method
    window : pd.DataFrame of shape (seq_len, n_features)
    time_points : np.ndarray of evaluation times
    train_windows : list of training windows to build background (if None, uses window with noise)
    feature_names : list of feature names (optional)
    nsamples : number of SHAP kernel evaluations
    
    Returns
    -------
    importance : dict {feature_name: score (0-1)}
    """
    if feature_names is None:
        feature_names = window.columns.tolist()

    n_features = len(feature_names)
    seq_len = window.shape[0]

    # Flatten the window to a 1D vector
    x_flat = window.values.ravel().astype(np.float32)

    # ------------------------------------------------------------------
    # Build prediction function for SHAP
    def predict_fn(X_2d):
        """
        X_2d : np.ndarray of shape (N, n_features * seq_len)
        returns : np.ndarray of shape (N, T)  -- survival curves at the model's time points
        """
        survs = []
        for row in X_2d:
            # Reconstruct DataFrame
            win_df = pd.DataFrame(row.reshape(seq_len, n_features), columns=feature_names)
            win_df = win_df.select_dtypes(include=[np.number]).copy()
            win_df['label'] = 0
            win_df['event'] = 0
            win_df['vehicle_id'] = 'test'

            src = next(iter(model.model_per_source)) if hasattr(model, 'model_per_source') else 'test'
            result = model.predict(win_df, src, event_data=None)

            # Handle different output shapes: (1, 1, T) or (1, T)
            if result.ndim == 3:
                surv = result[0, 0, :]
            elif result.ndim == 2:
                surv = result[0, :]
            else:
                surv = result.flatten()
            survs.append(surv)
        return np.array(survs)

    # ------------------------------------------------------------------
    # Background dataset: if training windows provided, use a random subset
    if train_windows is not None and len(train_windows) > 0:
        n_bg = min(nsamples, len(train_windows))
        idx = np.random.choice(len(train_windows), n_bg, replace=False)
        bg = np.array([w.values.ravel().astype(np.float32) for w in [train_windows[i] for i in idx]])
    else:
        # fallback: noisy copies of the instance itself (less representative)
        np.random.seed(42)
        bg = np.tile(x_flat, (nsamples, 1)) + np.random.normal(0, 0.01, (nsamples, len(x_flat)))
        bg = np.clip(bg, 0, 1)

    # ------------------------------------------------------------------
    # Kernel SHAP explainer
    explainer = shap.KernelExplainer(predict_fn, bg)
    shap_values = explainer.shap_values(x_flat.reshape(1, -1), nsamples=nsamples)

    # shap_values can be a list of arrays (one per time point) or a 2D array (n_times, n_features_flat)
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)          # (T, D)
    else:
        shap_values = np.atleast_2d(shap_values)     # ensure 2D

    # Aggregate across time: mean absolute SHAP per original feature
    feat_imp = np.mean(np.abs(shap_values), axis=0)  # (D,) or (T, D) -> mean over T
    if feat_imp.ndim > 1:
        feat_imp = np.mean(feat_imp, axis=0)

    # Map back to features (each feature has seq_len flattened entries)
    importance = {}
    for i, name in enumerate(feature_names):
        start = i * seq_len
        end = start + seq_len
        importance[name] = float(np.mean(feat_imp[start:end]))

    # Normalise to [0,1]
    maxv = max(importance.values()) if importance else 1.0
    if maxv > 0:
        for k in importance:
            importance[k] /= maxv

    return importance