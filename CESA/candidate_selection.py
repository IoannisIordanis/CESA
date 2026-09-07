import numpy as np
import pandas as pd
import time
from typing import List, Optional, Dict, Set
from scipy.spatial import cKDTree
from utils.survival_utils import compute_correlation_vector, flatten_window, get_survival_curve
from swap_evaluation import _first_derivatives, _second_derivatives, _shape_preserved

class CandidateIndex:
    def __init__(self, training_windows: List[pd.DataFrame], M: int = 200):
        self.M = M
        self.training_windows = training_windows
        self.N = len(training_windows)
        self.flat_vectors = np.array([flatten_window(w) for w in training_windows]).astype(np.float32)
        self.corr_vectors = np.array([compute_correlation_vector(w) for w in training_windows]).astype(np.float32)
        self.kdtree = cKDTree(self.flat_vectors)

    def get_candidates(
        self,
        test_window: pd.DataFrame,
        actionable_set: Set[str],
        model,
        time_points: np.ndarray,
        tau: float,
        delta_1: float,
        alpha: float,
        dt: float,
        beta: float = 0.5,
        k: int = 10,
        stats: Optional[Dict] = None,
        shape_required: bool = True
    ) -> List[pd.DataFrame]:
        test_flat = flatten_window(test_window).astype(np.float32)
        test_corr = compute_correlation_vector(test_window).astype(np.float32)
        test_surv = get_survival_curve(model, test_window, time_points, stats=stats)
        j = np.argmin(np.abs(time_points - tau))
        test_surv_at_tau = test_surv[j]
        test_d1 = _first_derivatives(test_surv, dt)
        test_d2 = _second_derivatives(test_surv, dt)

        k_query = min(self.M, self.N)
        dists, indices = self.kdtree.query(test_flat.reshape(1, -1), k=k_query)
        neighbour_indices = indices[0]

        candidates = []
        for idx in neighbour_indices:
            if idx < 0 or idx >= self.N:
                continue
            win = self.training_windows[idx]
            aligned = win.copy()
            for col in aligned.columns:
                if col not in actionable_set:
                    aligned[col] = test_window[col].values
            surv_aligned = get_survival_curve(model, aligned, time_points, stats=stats)
            if surv_aligned[j] >= test_surv_at_tau + delta_1:
                shape_ok = True
                if shape_required:
                    shape_ok = _shape_preserved(test_d1, test_d2, surv_aligned, dt, alpha)
                if shape_ok:
                    flat_aligned = flatten_window(aligned).astype(np.float32)
                    corr_aligned = compute_correlation_vector(aligned).astype(np.float32)
                    d_euc = np.linalg.norm(flat_aligned - test_flat)
                    d_corr = np.linalg.norm(corr_aligned - test_corr)
                    candidates.append((win, d_euc, d_corr))

        if not candidates:
            return []

        d_euc_vals = np.array([c[1] for c in candidates])
        d_corr_vals = np.array([c[2] for c in candidates])
        min_e, max_e = d_euc_vals.min(), d_euc_vals.max()
        min_c, max_c = d_corr_vals.min(), d_corr_vals.max()
        d_euc_norm = (d_euc_vals - min_e) / (max_e - min_e) if max_e > min_e else np.zeros_like(d_euc_vals)
        d_corr_norm = (d_corr_vals - min_c) / (max_c - min_c) if max_c > min_c else np.zeros_like(d_corr_vals)
        scores = beta * d_corr_norm + (1 - beta) * d_euc_norm

        if len(scores) <= k:
            idx_sorted = np.argsort(scores)
        else:
            idx_sorted = np.argpartition(scores, k)[:k]
            idx_sorted = idx_sorted[np.argsort(scores[idx_sorted])]
        return [candidates[i][0] for i in idx_sorted[:k]]