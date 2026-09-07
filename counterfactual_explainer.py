import numpy as np
import pandas as pd
import time
from typing import List, Dict, Set, Optional, Any
from CESA.feature_ranking import rank_features
from CESA.feature_grouping import group_features
from CESA.priority_search import priority_search
from CESA.full_search import full_search
from utils.survival_utils import get_survival_curve, find_closest_time_index
from swap_evaluation import _first_derivatives, _second_derivatives, shape_deviation, _shape_preserved

class CounterfactualExplainer:
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
        candidate_index: Optional[Any] = None
    ):
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

    def _get_changed_features(self, test_window, candidate_window):
        changed = []
        for col in self.actionable_set:
            if not np.allclose(test_window[col].values, candidate_window[col].values):
                changed.append(col)
        return changed

    def explain(self, test_window: pd.DataFrame, test_rul: float, tau, orig_surv, delta_1) -> Dict:
        j = find_closest_time_index(self.time_points, tau)
        orig_d1 = _first_derivatives(orig_surv, self.dt)
        orig_d2 = _second_derivatives(orig_surv, self.dt)

        stats = {
            'num_model_calls': 0,
            'total_model_time': 0.0,
            'num_shape_tests': 0,
            'time_shape_tests': 0.0,
            'time_ann_queries': 0.0,
            'time_exact_distances': 0.0
        }

        # ---- Candidate selection: first with shape ----
        t_cand_start = time.perf_counter()
        if self.candidate_index is not None:
            candidates = self.candidate_index.get_candidates(
                test_window, self.actionable_set, self.model, self.time_points,
                tau, delta_1, self.alpha, self.dt, beta=self.beta, k=self.k, stats=stats,
                shape_required=True
            )
        else:
            candidates = []
        had_shape_candidates = (len(candidates) > 0)
        time_candidate_selection = time.perf_counter() - t_cand_start

        shape_relaxed = False
        if not candidates:
            #print("  No shape‑preserving candidates. Relaxing shape constraint (gain only).")
            shape_relaxed = True
            t_cand_start = time.perf_counter()
            if self.candidate_index is not None:
                candidates = self.candidate_index.get_candidates(
                    test_window, self.actionable_set, self.model, self.time_points,
                    tau, delta_1, self.alpha, self.dt, beta=self.beta, k=self.k, stats=stats,
                    shape_required=False
                )
            time_candidate_selection = time.perf_counter() - t_cand_start

        no_candidates = (len(candidates) == 0)

        def make_result(win, new_surv, expl, reached, phase, t_cand, t_prio, t_full, t_fb, shape_ok):
            dev = shape_deviation(orig_d1, orig_d2, new_surv, self.dt)
            return {
                'explanation': expl,
                'position tau': j,
                'new_window': win,
                'original_survival': orig_surv,
                'new_survival': new_surv,
                'gain_at_tau': float(new_surv[j] - orig_surv[j]),
                'target_reached': bool(reached),
                'phase': phase,
                'time_candidate_selection': float(t_cand),
                'time_priority': float(t_prio),
                'time_full': float(t_full),
                'time_fallback': float(t_fb),
                'num_model_calls': stats['num_model_calls'],
                'total_model_time': float(stats['total_model_time']),
                'num_shape_tests': stats['num_shape_tests'],
                'time_shape_tests': float(stats['time_shape_tests']),
                'shape_preserved': bool(shape_ok),
                'shape_deviation': float(dev),
                'no_candidates_found': no_candidates,
                'had_shape_candidates': had_shape_candidates,
                'sparsity': len(expl)
            }

        if not candidates:
            return make_result(test_window, orig_surv, [], False, 'none',
                               time_candidate_selection, 0.0, 0.0, 0.0, True)

        # Precompute rankings and groups
        rankings = {}
        groups = {}
        for idx, cand in enumerate(candidates):
            rankings[idx] = rank_features(cand, test_window, self.actionable_set)
            groups[idx] = group_features(cand, self.actionable_set, threshold=0.7)

        # Priority search
        t_prio_start = time.perf_counter()
        win, expl, beneficial, priority_best = priority_search(
            candidates, test_window, orig_surv, delta_1, self.alpha, tau,
            self.time_points, self.dt, self.model, self.actionable_set,
            rankings, groups, stats
        )
        time_priority = time.perf_counter() - t_prio_start

        if not win.equals(test_window):
            surv = get_survival_curve(self.model, win, self.time_points, stats=stats)
            shape_ok = _shape_preserved(orig_d1, orig_d2, surv, self.dt, self.alpha)
            return make_result(win, surv, expl, True, 'priority',
                               time_candidate_selection, time_priority, 0.0, 0.0, shape_ok)

        # Full search
        # t_full_start = time.perf_counter()
        # win, expl, full_best = full_search(
        #     candidates, test_window, orig_surv, delta_1, self.alpha, tau,
        #     self.time_points, self.dt, self.model, self.actionable_set,
        #     beneficial, stats, max_total_size=10
        # )
        # time_full = time.perf_counter() - t_full_start
        #
        # if not win.equals(test_window):
        #     surv = get_survival_curve(self.model, win, self.time_points, stats=stats)
        #     shape_ok = _shape_preserved(orig_d1, orig_d2, surv, self.dt, self.alpha)
        #     return make_result(win, surv, expl, True, 'full',
        #                        time_candidate_selection, time_priority, time_full, 0.0, shape_ok)
        full_best =None

        # ---- No perfect shape‑preserving solution found. Collect best from all sources ----
        best_win = None
        best_expl = None
        best_dev = float('inf')
        best_phase = None

        # Candidate list (each whole window)
        for cand in candidates:
            surv_cand = get_survival_curve(self.model, cand, self.time_points, stats=stats)
            if surv_cand[j] >= orig_surv[j] + delta_1:
                dev = shape_deviation(orig_d1, orig_d2, surv_cand, self.dt)
                if dev < best_dev:
                    best_dev = dev
                    best_win = cand
                    best_expl = self._get_changed_features(test_window, cand)
                    best_phase = 'fallback'

        # Best from priority search (if any)
        if priority_best is not None:
            prio_win, prio_expl, prio_dev = priority_best
            if prio_dev < best_dev:
                best_dev = prio_dev
                best_win = prio_win
                best_expl = prio_expl
                best_phase = 'priority_best'

        # Best from full search (if any)
        if full_best is not None:
            full_win, full_expl, full_dev = full_best
            if full_dev < best_dev:
                best_dev = full_dev
                best_win = full_win
                best_expl = full_expl
                best_phase = 'full_best'

        if best_win is not None:
            surv = get_survival_curve(self.model, best_win, self.time_points, stats=stats)
            shape_ok = _shape_preserved(orig_d1, orig_d2, surv, self.dt, self.alpha)
            return make_result(best_win, surv, best_expl, True, best_phase,
                               time_candidate_selection, time_priority, 0.0, 0.0, shape_ok)
        else:
            return make_result(test_window, orig_surv, [], False, 'none',
                               time_candidate_selection, 0.0, 0.0, 0.0, True)