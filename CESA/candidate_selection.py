# candidate_selection.py

import numpy as np
import pandas as pd
import time
from typing import List, Optional, Dict, Set
from scipy.spatial import cKDTree
from utils.survival_utils import compute_correlation_vector, flatten_window, get_survival_curve
from swap_evaluation import is_shape_preserved


# ============================================================================
# Candidate Index
# ============================================================================

class CandidateIndex:
    """
    Index over training windows using a k‑d tree for fast nearest neighbour search.
    Stores both flattened feature vectors and correlation vectors.
    """

    def __init__(self, training_windows: List[pd.DataFrame], M: int = 200):
        """
        Parameters
        ----------
        training_windows : List[pd.DataFrame]
            All training windows used as the candidate pool.
        M : int
            Maximum number of neighbours to query during candidate retrieval.
        """
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
        shape_required: bool = True,
        use_corr: bool = True,
        use_prox: bool = True,
        extra_candidates: Optional[List[pd.DataFrame]] = None,
        orig_surv: Optional[np.ndarray] = None,
        shape_test: str = 'derivative',
        shape_delta: float = 0.05
    ) -> List[pd.DataFrame]:
        """
        Retrieve candidate windows that, after aligning non‑actionable features to the test window,
        achieve the required survival gain and optionally pass the shape preservation test.
        Candidates are ranked by a combination of Euclidean and correlation distances.

        Parameters
        ----------
        actionable_set : Set[str]
            Features that can be modified.
        model : survival model
            Used to predict survival curves.
        time_points : np.ndarray
            Time grid for survival evaluation.
        tau : float
            Time point of interest.
        delta_1 : float
            Required gain in survival at tau.
        alpha : float
            Significance level for shape tests.
        dt : float
            Time step between consecutive time points.
        beta : float, default=0.5
            Weight for correlation distance in the combined score; (1‑beta) weights Euclidean distance.
        k : int, default=10
            Number of candidates to return.
        stats : dict, optional
            Dictionary to collect timing and model call statistics.
        shape_required : bool, default=True
            If True, only candidates that preserve shape are kept.
        use_corr : bool, default=True
            Whether to use correlation distance in ranking.
        use_prox : bool, default=True
            Whether to use Euclidean distance in ranking.
        extra_candidates : list of pd.DataFrame, optional
            Additional windows to consider (e.g., from the same vehicle's past).
        orig_surv : np.ndarray, optional
            Pre‑computed survival curve of the test window.
        shape_test : str, default='derivative'
            Shape preservation test to apply (see swap_evaluation.is_shape_preserved).
        shape_delta : float, default=0.05
            Tolerance parameter for the shape test.

        Returns
        -------
        List[pd.DataFrame]
            Ranked list of candidate windows (aligned with test_window).
        """
        test_flat = flatten_window(test_window).astype(np.float32)
        test_corr = compute_correlation_vector(test_window).astype(np.float32)
        if orig_surv is None:
            orig_surv = get_survival_curve(model, test_window, time_points, stats=stats)
        test_surv = orig_surv
        j = np.argmin(np.abs(time_points - tau))
        test_surv_at_tau = test_surv[j]

        # Query k‑d tree for the M nearest neighbours
        k_query = min(self.M, self.N)
        dists, indices = self.kdtree.query(test_flat.reshape(1, -1), k=k_query)
        neighbour_indices = indices[0]

        raw_candidates = []
        for idx in neighbour_indices:
            if idx < 0 or idx >= self.N:
                continue
            win = self.training_windows[idx]
            raw_candidates.append(win)

        if extra_candidates:
            raw_candidates.extend(extra_candidates)

        candidates = []
        for win in raw_candidates:
            # Align non‑actionable features to the test window
            aligned = win.copy()
            for col in aligned.columns:
                if col not in actionable_set:
                    aligned[col] = test_window[col].values
            surv_aligned = get_survival_curve(model, aligned, time_points, stats=stats)

            # Check if survival gain is sufficient
            if surv_aligned[j] >= test_surv_at_tau + delta_1:
                # Optionally enforce shape preservation
                shape_ok = True
                if shape_required:
                    shape_ok = is_shape_preserved(test_surv, surv_aligned, dt, alpha,
                                                  shape_test, shape_delta)
                if shape_ok:
                    flat_aligned = flatten_window(aligned).astype(np.float32)
                    corr_aligned = compute_correlation_vector(aligned).astype(np.float32)
                    d_euc = np.linalg.norm(flat_aligned - test_flat)
                    d_corr = np.linalg.norm(corr_aligned - test_corr)
                    candidates.append((aligned, d_euc, d_corr))

        if not candidates:
            return []

        # If neither distance metric is used, return all in original order
        if not use_corr and not use_prox:
            return [c[0] for c in candidates]

        d_euc_vals = np.array([c[1] for c in candidates])
        d_corr_vals = np.array([c[2] for c in candidates])

        # Ranking logic based on selected metrics
        if use_corr and not use_prox:
            idx_sorted = np.argsort(d_corr_vals)
            return [candidates[i][0] for i in idx_sorted]

        if use_prox and not use_corr:
            idx_sorted = np.argsort(d_euc_vals)
            return [candidates[i][0] for i in idx_sorted]

        # Combined score: beta * corr_norm + (1-beta) * euc_norm
        min_e, max_e = d_euc_vals.min(), d_euc_vals.max()
        min_c, max_c = d_corr_vals.min(), d_corr_vals.max()
        d_euc_norm = (d_euc_vals - min_e) / (max_e - min_e) if max_e > min_e else np.zeros_like(d_euc_vals)
        d_corr_norm = (d_corr_vals - min_c) / (max_c - min_c) if max_c > min_c else np.zeros_like(d_corr_vals)
        scores = beta * d_corr_norm + (1 - beta) * d_euc_norm

        # Select top k candidates
        if len(scores) <= k:
            idx_sorted = np.argsort(scores)
        else:
            idx_sorted = np.argpartition(scores, k)[:k]
            idx_sorted = idx_sorted[np.argsort(scores[idx_sorted])]
        return [candidates[i][0] for i in idx_sorted[:k]]

    def get_fixed_candidates(
        self,
        test_window: pd.DataFrame,
        actionable_set: Set[str],
        k: int,
        extra_candidates: Optional[List[pd.DataFrame]] = None
    ) -> List[pd.DataFrame]:
        """
        Retrieve the k nearest neighbours from the training set (by Euclidean distance)
        without any survival or shape filtering. Aligns non‑actionable features to the test window.

        Parameters
        ----------
        test_window : pd.DataFrame
            The query window.
        actionable_set : Set[str]
            Features that can be modified.
        k : int
            Number of candidates to return.
        extra_candidates : list of pd.DataFrame, optional
            Additional windows to include before final selection.

        Returns
        -------
        List[pd.DataFrame]
            Aligned candidate windows, sorted by proximity to the test window.
        """
        test_flat = flatten_window(test_window).astype(np.float32)

        k_query = min(k, self.N)
        dists, indices = self.kdtree.query(test_flat.reshape(1, -1), k=k_query)

        candidates = []
        for idx in indices[0]:
            win = self.training_windows[idx].copy()
            for col in win.columns:
                if col not in actionable_set:
                    win[col] = test_window[col].values
            candidates.append(win)

        if extra_candidates:
            for extra_win in extra_candidates:
                aligned = extra_win.copy()
                for col in aligned.columns:
                    if col not in actionable_set:
                        aligned[col] = test_window[col].values
                candidates.append(aligned)

        # Ensure we return exactly k candidates by proximity
        if len(candidates) > k:
            flat_test = test_flat
            dists = [np.linalg.norm(flatten_window(w) - flat_test) for w in candidates]
            sorted_idx = np.argsort(dists)
            candidates = [candidates[i] for i in sorted_idx[:k]]

        return candidates