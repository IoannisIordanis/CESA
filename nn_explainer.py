# nn_explainer.py

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional
from utils.survival_utils import get_survival_curve, find_closest_time_index, flatten_window
from swap_evaluation import (
    is_shape_preserved,
    shape_deviation_from_surv,
    compute_absolute_deviation_metrics
)


# ============================================================================
# NNExplainer: nearest‑neighbour donor‑based counterfactual explainer
# ============================================================================

class NNExplainer:
    """
    Generates counterfactual explanations by finding the nearest training window
    that achieves the required survival gain at tau, then replacing only the
    actionable features of the test window with those from the donor.

    Non‑actionable features are kept unchanged from the test window.
    Shape preservation is evaluated on the final modified window.
    """

    def __init__(
        self,
        model,
        training_windows: List[pd.DataFrame],
        time_points: np.ndarray,
        dt: float,
        actionable_set: Optional[Set[str]] = None,
        alpha: float = 0.05,
        shape_test: str = 'derivative',
        shape_delta: float = 0.05
    ):
        """
        Parameters
        ----------
        model : survival model
            Fitted survival model.
        training_windows : list of pd.DataFrame
            Training windows used as donor pool.
        time_points : np.ndarray
            Time grid for survival evaluation.
        dt : float
            Time step between consecutive time points.
        actionable_set : set of str, optional
            Features that can be modified. If None, all features are actionable.
        alpha : float, default=0.05
            Significance level for shape preservation tests.
        shape_test : str, default='derivative'
            Which shape preservation test to use (see swap_evaluation).
        shape_delta : float, default=0.05
            Tolerance parameter for shape tests.
        """
        self.model = model
        self.training_windows = training_windows
        self.time_points = time_points
        self.dt = dt
        self.actionable_set = actionable_set if actionable_set is not None else set(training_windows[0].columns)
        self.columns = training_windows[0].columns
        self.seq_len = training_windows[0].shape[0]
        self.alpha = alpha
        self.shape_test = shape_test
        self.shape_delta = shape_delta

        # Pre‑compute flattened windows and survival curves for all training windows
        self.train_flat = [flatten_window(w) for w in self.training_windows]
        self.train_surv = [get_survival_curve(self.model, w, self.time_points) for w in self.training_windows]

    def _reconstruct_window(self, flat_array: np.ndarray, template_window: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct a window from a flattened array using the column order of the template.
        """
        new_win = template_window.copy(deep=False)
        reshaped = flat_array.reshape(self.seq_len, len(self.columns))
        for i, col in enumerate(self.columns):
            new_win[col] = reshaped[:, i]
        return new_win

    def explain(
        self,
        test_window: pd.DataFrame,
        test_rul: float,
        tau,
        orig_surv,
        delta_1,
        vehicle_id: Optional[str] = None,
        window_start_idx: Optional[int] = None
    ) -> Dict:
        """
        Generate a counterfactual explanation.

        Parameters
        ----------
        test_window : pd.DataFrame
            The original window to explain.
        test_rul : float
            RUL at the end of the window (not used in this explainer but kept for interface).
        tau : float or str
            Time point of interest; can be a float or the string 'change_point'.
        orig_surv : np.ndarray
            Survival curve of the test window.
        delta_1 : float
            Required gain in survival at tau.
        vehicle_id : str, optional
            Not used in NNExplainer but kept for interface compatibility.
        window_start_idx : int, optional
            Not used in NNExplainer but kept for interface compatibility.

        Returns
        -------
        dict
            Dictionary containing explanation details: changed features, new window,
            survival curves, gain, shape preservation status, and deviation metrics.
        """
        j = find_closest_time_index(self.time_points, tau)
        target_surv = orig_surv[j] + delta_1
        x_flat = flatten_window(test_window)

        # Boolean mask for actionable features in the flattened array
        actionable_mask = np.zeros(self.seq_len * len(self.columns), dtype=bool)
        for i, col in enumerate(self.columns):
            if col in self.actionable_set:
                actionable_mask[i::len(self.columns)] = True

        # Find the nearest training window (by Euclidean distance) that meets the survival target
        best_cand_flat = None
        min_dist = np.inf
        for i in range(len(self.training_windows)):
            if self.train_surv[i][j] >= target_surv:
                w_flat = self.train_flat[i]
                dist = np.linalg.norm(w_flat - x_flat)
                if dist < min_dist:
                    min_dist = dist
                    best_cand_flat = w_flat

        if best_cand_flat is None:
            # No donor found: return original window
            best_window = test_window
            best_surv = orig_surv
            target_reached = False
            changed_features = []
        else:
            # Replace only actionable features with donor values
            hybrid_flat = x_flat.copy()
            hybrid_flat[actionable_mask] = best_cand_flat[actionable_mask]
            best_window = self._reconstruct_window(hybrid_flat, test_window)
            try:
                best_surv = get_survival_curve(self.model, best_window, self.time_points)
                target_reached = bool(best_surv[j] >= target_surv)
            except Exception:
                # Fallback in case of prediction error
                best_surv = orig_surv
                best_window = test_window
                target_reached = False

            # Identify which actionable features actually changed
            changed_features = []
            for col in self.actionable_set:
                if not np.allclose(test_window[col].values, best_window[col].values):
                    changed_features.append(col)

        # Evaluate shape preservation and compute deviation metrics
        shape_ok = is_shape_preserved(orig_surv, best_surv, self.dt, self.alpha,
                                      self.shape_test, self.shape_delta)
        derivative_dev = shape_deviation_from_surv(orig_surv, best_surv, self.dt)
        abs_metrics = compute_absolute_deviation_metrics(orig_surv, best_surv)

        return {
            'explanation': changed_features,
            'position tau': int(j),
            'new_window': best_window,
            'original_survival': orig_surv,
            'new_survival': best_surv,
            'gain_at_tau': float(best_surv[j] - orig_surv[j]),
            'target_reached': target_reached,
            'donor_window': None,                    # Not stored in this simple version
            'shape_preserved': shape_ok,
            'shape_deviation': float(derivative_dev),
            'mad': abs_metrics['mad'],
            'sad': abs_metrics['sad'],
            'mae': abs_metrics['mae'],
        }