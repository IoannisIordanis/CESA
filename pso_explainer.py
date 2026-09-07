# pso_explainer.py

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
# PSOExplainer: Particle Swarm Optimisation for counterfactual generation
# ============================================================================

class PSOExplainer:
    """
    Uses Particle Swarm Optimisation to search for a modified window that achieves
    a specified survival gain at a given time point, while optionally preserving
    the shape of the survival curve.

    The objective is to minimise the Euclidean distance from the original window,
    subject to a penalty for insufficient survival gain and optionally a shape
    preservation penalty.
    """

    def __init__(
        self,
        model,
        training_windows: List[pd.DataFrame],
        time_points: np.ndarray,
        dt: float,
        actionable_set: Optional[Set[str]] = None,
        N: int = 50,
        N_iter: int = 100,
        preserve_shape: bool = False,
        alpha: float = 0.01,
        C: float = 1e6,
        shape_test: str = 'derivative',
        shape_delta: float = 0.05
    ):
        """
        Parameters
        ----------
        model : survival model
            Fitted survival model.
        training_windows : list of pd.DataFrame
            Training windows used to initialise bounds and the donor z_ct.
        time_points : np.ndarray
            Time grid for survival evaluation.
        dt : float
            Time step between consecutive time points.
        actionable_set : set of str, optional
            Features that can be modified. If None, all features are actionable.
        N : int, default=50
            Number of particles in the swarm.
        N_iter : int, default=100
            Number of optimisation iterations.
        preserve_shape : bool, default=False
            If True, adds a penalty for violating shape preservation.
        alpha : float, default=0.01
            Significance level for shape tests (when preserve_shape=True).
        C : float, default=1e6
            Penalty coefficient for constraint violations.
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
        self.N = N
        self.N_iter = N_iter
        self.preserve_shape = preserve_shape
        self.alpha = alpha
        self.C = C
        self.shape_test = shape_test
        self.shape_delta = shape_delta

        # Compute global bounds from all training windows
        all_data = np.stack([w.values for w in training_windows])
        self.x_min = np.min(all_data, axis=0).ravel()
        self.x_max = np.max(all_data, axis=0).ravel()
        self.columns = training_windows[0].columns
        self.seq_len = training_windows[0].shape[0]

        # Boolean mask for actionable features in flattened array
        self.actionable_mask = np.zeros(self.seq_len * len(self.columns), dtype=bool)
        for i, col in enumerate(self.columns):
            if col in self.actionable_set:
                self.actionable_mask[i::len(self.columns)] = True

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
        Generate a counterfactual explanation using PSO.

        Parameters
        ----------
        test_window : pd.DataFrame
            The original window to explain.
        test_rul : float
            RUL at the end of the window (not used but kept for interface).
        tau : float or str
            Time point of interest; can be a float or the string 'change_point'.
        orig_surv : np.ndarray
            Survival curve of the test window.
        delta_1 : float
            Required gain in survival at tau.
        vehicle_id : str, optional
            Not used but kept for interface compatibility.
        window_start_idx : int, optional
            Not used but kept for interface compatibility.

        Returns
        -------
        dict
            Dictionary containing explanation details: changed features, new window,
            survival curves, gain, shape preservation status, and deviation metrics.
        """
        j = find_closest_time_index(self.time_points, tau)
        target_surv = orig_surv[j] + delta_1
        x_flat = flatten_window(test_window)

        # ---- Find the closest training window (z_ct) that meets the target ----
        z_ct = None
        min_dist = np.inf
        for w in self.training_windows:
            s_w = get_survival_curve(self.model, w, self.time_points)
            if s_w[j] >= target_surv:
                w_flat = flatten_window(w)
                dist = np.linalg.norm(w_flat - x_flat)
                if dist < min_dist:
                    min_dist = dist
                    z_ct = w_flat
        R_ct = min_dist if z_ct is not None else np.inf

        # ---- Initialise particles ----
        particles = np.zeros((self.N, len(x_flat)))
        for i in range(self.N):
            if i == 0 and z_ct is not None:
                particles[i] = z_ct
            else:
                particles[i] = np.random.uniform(self.x_min, self.x_max)
                # Keep non‑actionable features fixed to the original
                particles[i][~self.actionable_mask] = x_flat[~self.actionable_mask]
                # Constrain within the hyper‑sphere of radius R_ct if possible
                if z_ct is not None:
                    dist = np.linalg.norm(particles[i] - x_flat)
                    if dist > R_ct and dist > 0:
                        direction = (particles[i] - x_flat) / dist
                        particles[i] = x_flat + direction * R_ct * np.random.uniform(0, 1)
                        particles[i][~self.actionable_mask] = x_flat[~self.actionable_mask]
                        particles[i] = np.clip(particles[i], self.x_min, self.x_max)

        velocities = np.zeros_like(particles)
        personal_bests = particles.copy()
        personal_best_scores = np.full(self.N, np.inf)
        global_best = None
        global_best_score = np.inf
        global_best_surv = None

        # PSO hyperparameters (standard values)
        w_pso = 0.729
        c1 = 1.4945
        c2 = 1.4945

        def evaluate_particle(u):
            """
            Evaluate a particle: objective = distance + penalty for insufficient gain
            and optional shape violation.
            """
            u_win = self._reconstruct_window(u, test_window)
            try:
                surv = get_survival_curve(self.model, u_win, self.time_points)
            except Exception:
                return np.inf, None
            dist = np.linalg.norm(u - x_flat)
            penalty = self.C * max(0, target_surv - surv[j])
            if self.preserve_shape:
                if not is_shape_preserved(orig_surv, surv, self.dt, self.alpha,
                                          self.shape_test, self.shape_delta):
                    penalty += self.C
            return dist + penalty, surv

        # ---- PSO main loop ----
        for _ in range(self.N_iter):
            for i in range(self.N):
                score, surv = evaluate_particle(particles[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_bests[i] = particles[i].copy()
                    if score < global_best_score:
                        global_best_score = score
                        global_best = particles[i].copy()
                        global_best_surv = surv

            # Update velocities and positions
            r1 = np.random.uniform(0, 1, size=particles.shape)
            r2 = np.random.uniform(0, 1, size=particles.shape)
            velocities = (w_pso * velocities +
                          r1 * c1 * (personal_bests - particles) +
                          r2 * c2 * (global_best - particles))
            particles = particles + velocities
            particles = np.clip(particles, self.x_min, self.x_max)
            # Re‑apply non‑actionable mask after clipping
            for i in range(self.N):
                particles[i][~self.actionable_mask] = x_flat[~self.actionable_mask]

        # ---- Determine if target was reached ----
        target_reached = False
        if global_best_surv is not None and global_best_surv[j] >= target_surv:
            if not self.preserve_shape or is_shape_preserved(orig_surv, global_best_surv, self.dt,
                                                            self.alpha, self.shape_test, self.shape_delta):
                target_reached = True
        target_reached = bool(target_reached)

        # ---- Build the final window ----
        if global_best is None:
            best_window = test_window
            best_surv = orig_surv
        else:
            best_window = self._reconstruct_window(global_best, test_window)
            best_surv = global_best_surv if global_best_surv is not None else orig_surv

        # ---- Compute shape preservation and deviation metrics ----
        shape_ok = is_shape_preserved(orig_surv, best_surv, self.dt, self.alpha,
                                      self.shape_test, self.shape_delta)
        derivative_dev = shape_deviation_from_surv(orig_surv, best_surv, self.dt)
        abs_metrics = compute_absolute_deviation_metrics(orig_surv, best_surv)

        # ---- Identify changed features ----
        changed_features = []
        for col in self.actionable_set:
            if not np.allclose(test_window[col].values, best_window[col].values):
                changed_features.append(col)

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
            'shape_deviation': float(derivative_dev),
            'mad': abs_metrics['mad'],
            'sad': abs_metrics['sad'],
            'mae': abs_metrics['mae'],
        }