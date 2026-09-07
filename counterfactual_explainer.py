# counterfactual_explainer.py

import numpy as np
import pandas as pd
import time
from typing import List, Dict, Set, Optional, Any, Tuple
from CESA.feature_ranking import rank_features
from CESA.feature_grouping import group_features
from CESA.priority_search import priority_search
from utils.survival_utils import get_survival_curve, find_closest_time_index, flatten_window
from swap_evaluation import (
    is_shape_preserved,
    shape_deviation_from_surv,
    compute_absolute_deviation_metrics
)


# ============================================================================
# CounterfactualExplainer class
# ============================================================================

class CounterfactualExplainer:
    """
    Main class for generating counterfactual explanations using candidate selection,
    feature grouping, and priority search. Supports shape preservation and multiple
    ranking criteria (correlation, proximity).
    """

    def __init__(
        self,
        model,
        training_windows: List[pd.DataFrame],
        time_points: np.ndarray,
        dt: float,
        alpha: float = 0.05,
        beta: float = 0.5,
        k: int = 10,
        M: int = 200,
        actionable_set: Optional[Set[str]] = None,
        candidate_index: Optional[Any] = None,
        use_shape: bool = True,
        use_corr: bool = True,
        use_prox: bool = True,
        fixed_pool: bool = False,
        vehicle_past_windows: Optional[Dict[str, List[Tuple[pd.DataFrame, int]]]] = None,
        grouping_threshold: float = 0.7,
        shape_test: str = 'derivative',
        shape_delta: float = 0.05
    ):
        """
        Parameters
        ----------
        model : survival model
            Fitted survival model.
        training_windows : list of pd.DataFrame
            Training windows used as candidate pool.
        time_points : np.ndarray
            Time grid for survival evaluation.
        dt : float
            Time step between consecutive time points.
        alpha : float, default=0.05
            Significance level for shape preservation tests.
        beta : float, default=0.5
            Weight for correlation distance in combined ranking (1-beta weights Euclidean).
        k : int, default=10
            Number of candidates to retrieve.
        M : int, default=200
            Number of nearest neighbours to query from the k‑d tree.
        actionable_set : set of str, optional
            Features that can be modified. If None, all features are actionable.
        candidate_index : CandidateIndex, optional
            Pre‑built index for candidate retrieval. If None, one is built internally.
        use_shape : bool, default=True
            Whether to enforce shape preservation during search.
        use_corr : bool, default=True
            Whether to use correlation distance in candidate ranking.
        use_prox : bool, default=True
            Whether to use Euclidean distance in candidate ranking.
        fixed_pool : bool, default=False
            If True, use a fixed number of nearest neighbours (k) without survival filtering.
        vehicle_past_windows : dict, optional
            Mapping from vehicle ID to list of (window, index) for past windows.
        grouping_threshold : float, default=0.7
            Correlation threshold for feature grouping.
        shape_test : str, default='derivative'
            Which shape preservation test to use (see swap_evaluation).
        shape_delta : float, default=0.05
            Tolerance parameter for shape tests.
        """
        self.model = model
        self.training_windows = training_windows
        self.time_points = time_points
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.M = M
        self.actionable_set = actionable_set if actionable_set is not None else set(training_windows[0].columns)
        self.candidate_index = candidate_index
        self.use_shape = use_shape
        self.use_corr = use_corr
        self.use_prox = use_prox
        self.fixed_pool = fixed_pool
        self.vehicle_past_windows = vehicle_past_windows
        self.grouping_threshold = grouping_threshold
        self.shape_test = shape_test
        self.shape_delta = shape_delta

        # Internal storage for group statistics (for debugging/analysis)
        self._last_group_min = 0
        self._last_group_max = 0
        self._last_group_mean = 0.0
        self._last_chosen_group_count = 0

    def _get_changed_features(self, test_window, candidate_window):
        """Return the list of actionable features that differ between two windows."""
        changed = []
        for col in self.actionable_set:
            if not np.allclose(test_window[col].values, candidate_window[col].values):
                changed.append(col)
        return changed

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
        Generate a counterfactual explanation for the given test window.

        Parameters
        ----------
        test_window : pd.DataFrame
            The original window to explain.
        test_rul : float
            RUL at the end of the window (used for context, not directly in search).
        tau : float or str
            Time point of interest; can be a float or the string 'change_point'.
        orig_surv : np.ndarray
            Survival curve of the test window.
        delta_1 : float
            Required gain in survival at tau.
        vehicle_id : str, optional
            Identifier of the vehicle (used to retrieve past windows).
        window_start_idx : int, optional
            Start index of the window within the vehicle's history.

        Returns
        -------
        dict
            Dictionary containing explanation details: changed features, new window,
            survival curves, gain, shape preservation status, timing statistics, etc.
        """
        j = find_closest_time_index(self.time_points, tau)

        # Statistics collector (model calls, timing, groups)
        stats = {
            'num_model_calls': 0,
            'total_model_time': 0.0,
            'num_shape_tests': 0,
            'time_shape_tests': 0.0,
            'time_ann_queries': 0.0,
            'time_exact_distances': 0.0,
            'group_min': 0,
            'group_max': 0,
            'group_mean': 0.0,
            'chosen_group_count': 0
        }

        # Retrieve past windows from the same vehicle (if available)
        extra_candidates = []
        if self.vehicle_past_windows is not None and vehicle_id is not None and window_start_idx is not None:
            if vehicle_id in self.vehicle_past_windows:
                past = [(win, idx) for win, idx in self.vehicle_past_windows[vehicle_id] if idx < window_start_idx]
                if past:
                    test_flat = flatten_window(test_window)
                    past_windows = [win for win, _ in past]
                    dists = [np.linalg.norm(flatten_window(w) - test_flat) for w in past_windows]
                    sorted_pairs = sorted(zip(past_windows, dists), key=lambda x: x[1])
                    extra_candidates = [w for w, _ in sorted_pairs[:5]]

        # Candidate selection
        t_cand_start = time.perf_counter()
        shape_relaxed = False
        initial_pool_size = 0
        final_pool_size = 0

        if self.fixed_pool:
            # Fixed pool: nearest neighbours without survival/shape filtering
            candidates = self.candidate_index.get_fixed_candidates(
                test_window, self.actionable_set, self.k, extra_candidates=extra_candidates
            )
            had_shape_candidates = False
            shape_relaxed = False
            initial_pool_size = len(candidates)
            final_pool_size = len(candidates)
        else:
            # Adaptive pool: filter by gain and optionally shape
            shape_required = self.use_shape
            candidates = self.candidate_index.get_candidates(
                test_window, self.actionable_set, self.model, self.time_points,
                tau, delta_1, self.alpha, self.dt, beta=self.beta, k=self.k, stats=stats,
                shape_required=shape_required,
                use_corr=self.use_corr, use_prox=self.use_prox,
                extra_candidates=extra_candidates,
                orig_surv=orig_surv,
                shape_test=self.shape_test,
                shape_delta=self.shape_delta
            )
            had_shape_candidates = (len(candidates) > 0) if self.use_shape else False
            initial_pool_size = len(candidates)

            # If no shape‑preserving candidates exist, relax the shape constraint
            if not candidates and self.use_shape:
                shape_relaxed = True
                candidates = self.candidate_index.get_candidates(
                    test_window, self.actionable_set, self.model, self.time_points,
                    tau, delta_1, self.alpha, self.dt, beta=self.beta, k=self.k, stats=stats,
                    shape_required=False,
                    use_corr=self.use_corr, use_prox=self.use_prox,
                    extra_candidates=extra_candidates,
                    orig_surv=orig_surv,
                    shape_test=self.shape_test,
                    shape_delta=self.shape_delta
                )
            final_pool_size = len(candidates)

        time_candidate_selection = time.perf_counter() - t_cand_start
        no_candidates = (len(candidates) == 0)

        # Helper to build the result dictionary
        def make_result(win, new_surv, expl, reached, phase, t_prio, t_full, t_fb, shape_ok):
            derivative_dev = shape_deviation_from_surv(orig_surv, new_surv, self.dt)
            abs_metrics = compute_absolute_deviation_metrics(orig_surv, new_surv)

            result = {
                'explanation': expl,
                'position tau': j,
                'new_window': win,
                'original_survival': orig_surv,
                'new_survival': new_surv,
                'gain_at_tau': float(new_surv[j] - orig_surv[j]),
                'target_reached': bool(reached),
                'phase': phase,
                'time_candidate_selection': float(time_candidate_selection),
                'time_priority': float(t_prio),
                'time_full': float(t_full),
                'time_fallback': float(t_fb),
                'num_model_calls': stats['num_model_calls'],
                'total_model_time': float(stats['total_model_time']),
                'num_shape_tests': stats['num_shape_tests'],
                'time_shape_tests': float(stats['time_shape_tests']),
                'shape_preserved': bool(shape_ok),
                'shape_deviation': float(derivative_dev),
                'mad': abs_metrics['mad'],
                'sad': abs_metrics['sad'],
                'mae': abs_metrics['mae'],
                'no_candidates_found': no_candidates,
                'had_shape_candidates': had_shape_candidates,
                'sparsity': len(expl),
                'group_min': stats.get('group_min', 0),
                'group_max': stats.get('group_max', 0),
                'group_mean': stats.get('group_mean', 0.0),
                'chosen_group_count': stats.get('chosen_group_count', 0),
                'shape_relaxed': shape_relaxed,
                'initial_pool_size': initial_pool_size,
                'final_pool_size': final_pool_size,
            }
            self._last_group_min = result['group_min']
            self._last_group_max = result['group_max']
            self._last_group_mean = result['group_mean']
            self._last_chosen_group_count = result['chosen_group_count']
            return result

        # No candidates: return the original window
        if not candidates:
            stats['group_min'] = 0
            stats['group_max'] = 0
            stats['group_mean'] = 0.0
            stats['chosen_group_count'] = 0
            return make_result(test_window, orig_surv, [], False, 'none',
                               0.0, 0.0, 0.0, True)

        # Compute feature rankings and groups for each candidate
        rankings = {}
        groups = {}
        for idx, cand in enumerate(candidates):
            rankings[idx] = rank_features(cand, test_window, self.actionable_set,
                                          use_corr=self.use_corr, use_prox=self.use_prox)
            groups[idx] = group_features(cand, self.actionable_set, threshold=self.grouping_threshold)

        # Priority search over candidates and feature groups
        t_prio_start = time.perf_counter()
        win, expl, beneficial, priority_best = priority_search(
            candidates, test_window, orig_surv, delta_1, self.alpha, tau,
            self.time_points, self.dt, self.model, self.actionable_set,
            rankings, groups, stats,
            use_shape=self.use_shape,
            use_corr=self.use_corr,
            use_prox=self.use_prox,
            shape_test=self.shape_test,
            shape_delta=self.shape_delta
        )
        time_priority = time.perf_counter() - t_prio_start

        # If priority search found a valid modification, return it
        if not win.equals(test_window):
            surv = get_survival_curve(self.model, win, self.time_points, stats=stats)
            shape_ok = is_shape_preserved(orig_surv, surv, self.dt, self.alpha,
                                          self.shape_test, self.shape_delta) if self.use_shape else False
            return make_result(win, surv, expl, True, 'priority',
                               time_priority, 0.0, 0.0, shape_ok)

        # Fallback: try to use the best (shape‑wise) candidate that reaches the gain
        best_win = None
        best_expl = None
        best_dev = float('inf')
        best_phase = None

        for cand in candidates:
            surv_cand = get_survival_curve(self.model, cand, self.time_points, stats=stats)
            if surv_cand[j] >= orig_surv[j] + delta_1:
                dev = shape_deviation_from_surv(orig_surv, surv_cand, self.dt)
                if dev < best_dev:
                    best_dev = dev
                    best_win = cand
                    best_expl = self._get_changed_features(test_window, cand)
                    best_phase = 'fallback'

        # If priority search had a fallback candidate (from beneficial groups), use it if better
        if priority_best is not None:
            prio_win, prio_expl, prio_dev = priority_best
            if prio_dev < best_dev:
                best_dev = prio_dev
                best_win = prio_win
                best_expl = prio_expl
                best_phase = 'priority_best'

        if best_win is not None:
            surv = get_survival_curve(self.model, best_win, self.time_points, stats=stats)
            shape_ok = is_shape_preserved(orig_surv, surv, self.dt, self.alpha,
                                          self.shape_test, self.shape_delta) if self.use_shape else False
            return make_result(best_win, surv, best_expl, True, best_phase,
                               time_priority, 0.0, 0.0, shape_ok)
        else:
            # No fallback solution: return original
            stats['chosen_group_count'] = 0
            return make_result(test_window, orig_surv, [], False, 'none',
                               0.0, 0.0, 0.0, False)