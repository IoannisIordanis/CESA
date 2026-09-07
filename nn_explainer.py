import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional
from utils.survival_utils import get_survival_curve, find_closest_time_index, flatten_window
from swap_evaluation import _first_derivatives, _second_derivatives, shape_deviation, _shape_preserved

class NNExplainer:
    def __init__(
        self,
        model,
        training_windows: List[pd.DataFrame],
        time_points: np.ndarray,
        dt: float,
        actionable_set: Optional[Set[str]] = None,
        alpha: float = 0.05
    ):
        self.model = model
        self.training_windows = training_windows
        self.time_points = time_points
        self.dt = dt
        self.actionable_set = actionable_set if actionable_set is not None else set(training_windows[0].columns)
        self.columns = training_windows[0].columns
        self.seq_len = training_windows[0].shape[0]
        self.alpha = alpha

        self.train_flat = [flatten_window(w) for w in self.training_windows]
        self.train_surv = [get_survival_curve(self.model, w, self.time_points) for w in self.training_windows]

    def _reconstruct_window(self, flat_array: np.ndarray, template_window: pd.DataFrame) -> pd.DataFrame:
        new_win = template_window.copy(deep=False)
        reshaped = flat_array.reshape(self.seq_len, len(self.columns))
        for i, col in enumerate(self.columns):
            new_win[col] = reshaped[:, i]
        return new_win

    def explain(self, test_window: pd.DataFrame, test_rul: float, tau, orig_surv, delta_1) -> Dict:
        j = find_closest_time_index(self.time_points, tau)
        target_surv = orig_surv[j] + delta_1
        x_flat = flatten_window(test_window)

        orig_d1 = _first_derivatives(orig_surv, self.dt)
        orig_d2 = _second_derivatives(orig_surv, self.dt)

        actionable_mask = np.zeros(self.seq_len * len(self.columns), dtype=bool)
        for i, col in enumerate(self.columns):
            if col in self.actionable_set:
                actionable_mask[i::len(self.columns)] = True

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
            best_window = test_window
            best_surv = orig_surv
            target_reached = False
            changed_features = []
        else:
            hybrid_flat = x_flat.copy()
            hybrid_flat[actionable_mask] = best_cand_flat[actionable_mask]
            best_window = self._reconstruct_window(hybrid_flat, test_window)
            try:
                best_surv = get_survival_curve(self.model, best_window, self.time_points)
                target_reached = bool(best_surv[j] >= target_surv)
            except Exception:
                best_surv = orig_surv
                best_window = test_window
                target_reached = False

            changed_features = []
            for col in self.actionable_set:
                if not np.allclose(test_window[col].values, best_window[col].values):
                    changed_features.append(col)

        dev = shape_deviation(orig_d1, orig_d2, best_surv, self.dt)
        shape_ok = bool(_shape_preserved(orig_d1, orig_d2, best_surv, self.dt, self.alpha))

        return {
            'explanation': changed_features,
            'position tau': int(j),
            'new_window': best_window,
            'original_survival': orig_surv,
            'new_survival': best_surv,
            'gain_at_tau': float(best_surv[j] - orig_surv[j]),
            'target_reached': target_reached,
            'donor_window': None,
            'shape_preserved': shape_ok,
            'shape_deviation': float(dev)
        }
