# surv_cf_explainer.py

import numpy as np
import pandas as pd
import copy
import random
import math
from typing import List, Dict, Set, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import dual_annealing
from utils.survival_utils import get_survival_curve, find_closest_time_index, flatten_window
from swap_evaluation import (
    shape_deviation_from_surv,
    compute_absolute_deviation_metrics,
    is_shape_preserved
)


# ============================================================================
# Optimizer wrappers (Simulated Annealing, PSO, Dual Annealing)
# ============================================================================

class SimulatedAnnealing:
    """
    Simulated Annealing optimizer with linear cooling and optional step decay.
    """
    def __init__(self, T_start: float, T_stop: float, iterations: int,
                 step_decrease_rate=0.0, step=0.01):
        if T_start < T_stop:
            raise ValueError("T_start must be greater than T_stop")
        self.T_start = T_start
        self.T_stop = T_stop
        self.iterations = iterations
        self.alpha = (T_start - T_stop) / iterations
        self.step_decrease_rate = step_decrease_rate
        self.step = step

    def minimize(self, func, bounds, x_original, **kwargs):
        """
        Run the simulated annealing loop.

        Parameters
        ----------
        func : callable
            Objective function to minimise.
        bounds : list of tuples
            (lower, upper) bounds for each dimension.
        x_original : np.ndarray
            Starting point.
        **kwargs : additional arguments passed to func.

        Returns
        -------
        np.ndarray
            Best found solution.
        """
        T = self.T_start
        x = copy.deepcopy(x_original)
        x_best = copy.deepcopy(x_original)

        E = func(x, **kwargs)
        E_best = E

        bounds_min = np.array([b[0] for b in bounds])
        bounds_max = np.array([b[1] for b in bounds])

        step_val = self.step
        for _ in range(self.iterations):
            if T <= self.T_stop:
                break

            dx = np.random.uniform(-step_val, step_val, size=x.shape[-1])
            x_next = np.clip(x + dx, bounds_min, bounds_max)

            E_next = func(x_next, **kwargs)
            dE = E_next - E

            if dE < 0:
                x = x_next
                E = E_next
            else:
                p = math.exp(-dE / T) if dE / T < 700 else 0.0
                if random.uniform(0, 1) < p:
                    x = x_next
                    E = E_next

            if E < E_best:
                x_best = copy.deepcopy(x)
                E_best = E

            T = T - self.alpha
            step_val = step_val * (1.0 - self.step_decrease_rate)

        return x_best


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimizer with optional use of pyswarms library and a custom fallback.
    """
    def __init__(self, n_particles: int, iterations: int, patience: int = 10,
                 options: Optional[dict] = None, ftol: float = 1e-3):
        self.n_particles = n_particles
        self.iterations = iterations
        self.patience = patience
        self.options = options if options is not None else {'c1': 1.49618, 'c2': 1.49618, 'w': 0.7298}
        self.ftol = ftol

    def minimize(self, func, bounds, x_original, **kwargs):
        """
        Run PSO. Tries to use pyswarms if available; otherwise uses a custom implementation.

        Parameters
        ----------
        func : callable
            Objective function. Can accept a batch of particles (2D array) or a single particle.
        bounds : list of tuples
            (lower, upper) for each dimension.
        x_original : np.ndarray
            Initial point (used as first particle).
        **kwargs : additional arguments passed to func.

        Returns
        -------
        np.ndarray
            Best found solution.
        """
        try:
            import pyswarms
            min_bounds = np.array([b[0] for b in bounds])
            max_bounds = np.array([b[1] for b in bounds])
            pso_bounds = (min_bounds, max_bounds)

            pso_opt = pyswarms.single.GlobalBestPSO(
                n_particles=self.n_particles,
                dimensions=len(min_bounds),
                bounds=pso_bounds,
                options=self.options,
                ftol_iter=self.patience,
                ftol=self.ftol
            )

            def pso_obj(u):
                return func(u, **kwargs)

            best_cost, best = pso_opt.optimize(pso_obj, iters=self.iterations, verbose=False)
            return best
        except (ImportError, Exception):
            return self._custom_minimize(func, bounds, x_original, **kwargs)

    def _custom_minimize(self, func, bounds, x_original, **kwargs):
        """Fallback PSO implementation when pyswarms is not available."""
        min_bounds = np.array([b[0] for b in bounds])
        max_bounds = np.array([b[1] for b in bounds])
        dimensions = len(min_bounds)

        c1 = self.options.get('c1', 1.49618)
        c2 = self.options.get('c2', 1.49618)
        w = self.options.get('w', 0.7298)

        particles = np.zeros((self.n_particles, dimensions))
        for i in range(self.n_particles):
            if i == 0:
                particles[i] = copy.deepcopy(x_original)
            else:
                particles[i] = np.random.uniform(min_bounds, max_bounds)

        velocities = np.zeros_like(particles)
        personal_bests = particles.copy()

        personal_best_scores = func(particles, **kwargs)

        best_idx = np.argmin(personal_best_scores)
        global_best = personal_bests[best_idx].copy()
        global_best_score = personal_best_scores[best_idx]

        no_improvement_count = 0
        for t in range(self.iterations):
            r1 = np.random.uniform(0, 1, size=particles.shape)
            r2 = np.random.uniform(0, 1, size=particles.shape)

            velocities = (w * velocities +
                          c1 * r1 * (personal_bests - particles) +
                          c2 * r2 * (global_best - particles))

            particles = particles + velocities
            particles = np.clip(particles, min_bounds, max_bounds)

            scores = func(particles, **kwargs)

            improved_mask = scores < personal_best_scores
            personal_bests[improved_mask] = particles[improved_mask]
            personal_best_scores[improved_mask] = scores[improved_mask]

            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < global_best_score - self.ftol:
                global_best_score = scores[min_score_idx]
                global_best = particles[min_score_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.patience:
                break

        return global_best


class DualAnnealing:
    """
    Wrapper for scipy.optimize.dual_annealing.
    """
    def __init__(self, maxiter: int = 100):
        self.maxiter = maxiter

    def minimize(self, func, bounds, x_original, **kwargs):
        """
        Run dual annealing optimisation.

        Parameters
        ----------
        func : callable
            Objective function.
        bounds : list of tuples
            (lower, upper) for each dimension.
        x_original : np.ndarray
            Starting point.
        **kwargs : additional arguments passed to func.

        Returns
        -------
        np.ndarray
            Best found solution.
        """
        def da_obj(u):
            return func(u, **kwargs)

        res = dual_annealing(da_obj, bounds, x0=x_original, maxiter=self.maxiter)
        return res.x


# ============================================================================
# SurvCFExplainer: generic optimisation‑based counterfactual explainer
# ============================================================================

class SurvCFExplainer:
    """
    Counterfactual explainer that uses a general-purpose optimisation algorithm
    (PSO, Simulated Annealing, or Dual Annealing) to search for a modified window.

    The loss function combines:
        - Distance from the original window (scaled)
        - Target survival loss (hinge or absolute)
        - Optional one‑hot encoding mutual exclusion penalty
        - Optional anomaly score penalty
    Shape preservation is evaluated after optimisation (not during, except via final check).
    """

    def __init__(
        self,
        model,
        training_windows: List[pd.DataFrame],
        time_points: np.ndarray,
        dt: float,
        actionable_set: Optional[Set[str]] = None,
        optimizer_name: str = 'pso',
        norm_distance: int = 1,
        norm_target: int = 2,
        weight_distance: float = 1.0,
        weight_target: float = 1e2,
        weight_anomaly: float = 1.0,
        weight_mutual_exclusion: float = 0.0,
        anomaly_model=None,
        anomaly_threshold=None,
        feature_types: Optional[List[str]] = None,
        ohe_features: Optional[List[List[int]]] = None,
        hinge_target: bool = True,
        n_particles: int = 100,
        n_iterations: int = 100000,
        patience: int = 200,
        sa_T_start: float = 10.0,
        sa_T_stop: float = 0.001,
        sa_step: float = 0.01,
        sa_step_decrease_rate: float = 0.0,
        da_maxiter: int = 100,
        shape_test: str = 'derivative',
        shape_delta: float = 0.05
    ):
        """
        Parameters
        ----------
        model : survival model
            Fitted survival model.
        training_windows : list of pd.DataFrame
            Training windows used to fit the scaler and determine bounds.
        time_points : np.ndarray
            Time grid for survival evaluation.
        dt : float
            Time step between consecutive time points.
        actionable_set : set of str, optional
            Features that can be modified. If None, all features are actionable.
        optimizer_name : str, default='pso'
            Optimizer to use: 'pso', 'sa' (Simulated Annealing), or 'da' (Dual Annealing).
        norm_distance : int, default=1
            Norm order for distance loss (e.g., 1 for L1, 2 for L2).
        norm_target : int, default=2
            Exponent for target loss.
        weight_distance : float, default=1.0
            Weight for the distance loss term.
        weight_target : float, default=1e2
            Weight for the target survival loss.
        weight_anomaly : float, default=1.0
            Weight for the anomaly penalty.
        weight_mutual_exclusion : float, default=0.0
            Weight for the one‑hot mutual exclusion penalty.
        anomaly_model : optional
            Model that can compute anomaly scores; must have `.anomaly_score()` method.
        anomaly_threshold : float, optional
            Threshold above which anomaly penalty is applied.
        feature_types : list of str, optional
            For each column, either 'bool' or 'float'. If not provided, auto‑detected.
        ohe_features : list of lists, optional
            Groups of feature indices that form one‑hot encoded sets. Penalty enforces exactly one 1.
        hinge_target : bool, default=True
            If True, target loss is max(0, target_surv - surv[j])^norm_target.
            If False, uses absolute difference.
        n_particles : int, default=100
            Number of particles for PSO.
        n_iterations : int, default=100000
            Number of iterations for PSO and SA.
        patience : int, default=200
            Early stopping patience for PSO.
        sa_T_start : float, default=10.0
            Starting temperature for SA.
        sa_T_stop : float, default=0.001
            Stopping temperature for SA.
        sa_step : float, default=0.01
            Initial step size for SA.
        sa_step_decrease_rate : float, default=0.0
            Decay rate for SA step size (per iteration).
        da_maxiter : int, default=100
            Maximum iterations for Dual Annealing.
        shape_test : str, default='derivative'
            Shape preservation test to use (see swap_evaluation).
        shape_delta : float, default=0.05
            Tolerance for shape tests.
        """
        self.model = model
        self.training_windows = training_windows
        self.time_points = time_points
        self.dt = dt
        self.actionable_set = actionable_set if actionable_set is not None else set(training_windows[0].columns)

        self.norm_distance = norm_distance
        self.norm_target = norm_target
        self.weight_distance = weight_distance
        self.weight_target = weight_target
        self.weight_anomaly = weight_anomaly
        self.weight_mutual_exclusion = weight_mutual_exclusion
        self.anomaly_model = anomaly_model
        self.anomaly_threshold = anomaly_threshold
        self.hinge_target = hinge_target
        self.shape_test = shape_test
        self.shape_delta = shape_delta

        self.columns = list(training_windows[0].columns)
        self.seq_len = training_windows[0].shape[0]
        self.num_features = len(self.columns)

        # Fit a MinMax scaler on flattened training windows
        self.train_flat = [flatten_window(w) for w in self.training_windows]
        self.X_flat = np.array(self.train_flat)
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.X_flat)

        # Bounds for each flattened feature (from training data)
        self.x_min = np.min(self.X_flat, axis=0)
        self.x_max = np.max(self.X_flat, axis=0)

        # Actionable mask in flattened form
        self.actionable_mask = np.zeros(self.seq_len * self.num_features, dtype=bool)
        for i, col in enumerate(self.columns):
            if col in self.actionable_set:
                self.actionable_mask[i::self.num_features] = True

        # Auto‑detect feature types if not provided
        if feature_types is None:
            feature_types = []
            for col in self.columns:
                unique_vals = set()
                for w in self.training_windows:
                    unique_vals.update(w[col].unique())
                    if len(unique_vals) > 2:
                        break
                if unique_vals.issubset({0.0, 1.0, 0, 1}):
                    feature_types.append('bool')
                else:
                    feature_types.append('float')
        self.feature_types = feature_types

        # Map feature types to flattened array (repeated for each time step)
        self.flat_feature_types = []
        for _ in range(self.seq_len):
            self.flat_feature_types.extend(self.feature_types)

        # Map OHE groups to flattened indices
        self.ohe_features = ohe_features
        if self.ohe_features:
            resolved_ohe = []
            for ohe_group in self.ohe_features:
                resolved_group = []
                for item in ohe_group:
                    if isinstance(item, str):
                        resolved_group.append(self.columns.index(item))
                    else:
                        resolved_group.append(item)
                resolved_ohe.append(resolved_group)
            self.ohe_features = resolved_ohe

        self.flat_ohe_features = []
        if self.ohe_features:
            for ohe_group in self.ohe_features:
                for t in range(self.seq_len):
                    flat_ohe_group = [t * self.num_features + idx for idx in ohe_group]
                    self.flat_ohe_features.append(flat_ohe_group)

        # Instantiate the requested optimizer
        self.optimizer_name = optimizer_name.lower()
        if self.optimizer_name == 'pso':
            self.optimizer = ParticleSwarmOptimization(
                n_particles=n_particles,
                iterations=n_iterations,
                patience=patience
            )
        elif self.optimizer_name == 'sa':
            self.optimizer = SimulatedAnnealing(
                T_start=sa_T_start,
                T_stop=sa_T_stop,
                iterations=n_iterations,
                step_decrease_rate=sa_step_decrease_rate,
                step=sa_step
            )
        elif self.optimizer_name == 'da':
            self.optimizer = DualAnnealing(
                maxiter=da_maxiter
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _reconstruct_window(self, flat_array: np.ndarray, template_window: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct a window from a flattened array using the template's column order."""
        new_win = template_window.copy(deep=False)
        reshaped = flat_array.reshape(self.seq_len, self.num_features)
        for i, col in enumerate(self.columns):
            new_win[col] = reshaped[:, i]
        return new_win

    def _loss(self, u, x_flat, target_surv, j, test_window):
        """
        Loss function for optimisation.

        Parameters
        ----------
        u : np.ndarray (1D or 2D)
            Flattened candidate window(s). If 2D, each row is a particle.
        x_flat : np.ndarray
            Flattened original window.
        target_surv : float
            Desired survival probability at time index j.
        j : int
            Time index for target.
        test_window : pd.DataFrame
            Original window (used for reconstruction).

        Returns
        -------
        float or np.ndarray
            Loss value(s). If u is 2D, returns a 1D array of losses per particle.
        """
        is_batch = u.ndim == 2
        u_mapped = u.copy()

        # Round boolean features to 0/1 based on threshold 0.5
        for i, t in enumerate(self.flat_feature_types):
            if t == 'bool':
                u_mapped[..., i] = (u[..., i] > 0.5).astype(float)

        # Keep non‑actionable features fixed to original values
        if is_batch:
            u_mapped[:, ~self.actionable_mask] = x_flat[~self.actionable_mask]
        else:
            u_mapped[~self.actionable_mask] = x_flat[~self.actionable_mask]

        # ---- Target loss ----
        if is_batch:
            loss_targets = []
            for p_idx in range(u_mapped.shape[0]):
                p_flat = u_mapped[p_idx]
                p_win = self._reconstruct_window(p_flat, test_window)
                try:
                    surv = get_survival_curve(self.model, p_win, self.time_points)
                    if self.hinge_target:
                        val = max(0.0, target_surv - surv[j])
                    else:
                        val = abs(surv[j] - target_surv)
                    loss_targets.append(val ** self.norm_target)
                except Exception:
                    loss_targets.append(1e9)
            loss_target = self.weight_target * np.array(loss_targets)
        else:
            p_win = self._reconstruct_window(u_mapped, test_window)
            try:
                surv = get_survival_curve(self.model, p_win, self.time_points)
                if self.hinge_target:
                    val = max(0.0, target_surv - surv[j])
                else:
                    val = abs(surv[j] - target_surv)
                loss_target = self.weight_target * (val ** self.norm_target)
            except Exception:
                loss_target = 1e9

        # ---- Distance loss (scaled) ----
        if is_batch:
            u_scaled = self.scaler.transform(u_mapped)
            x_scaled = self.scaler.transform(x_flat.reshape(1, -1))
            loss_distance = self.weight_distance * np.linalg.norm(u_scaled - x_scaled,
                                                                  ord=self.norm_distance, axis=1)
        else:
            u_scaled = self.scaler.transform(u_mapped.reshape(1, -1))[0]
            x_scaled = self.scaler.transform(x_flat.reshape(1, -1))[0]
            loss_distance = self.weight_distance * np.linalg.norm(u_scaled - x_scaled,
                                                                  ord=self.norm_distance)

        # ---- One‑hot mutual exclusion penalty ----
        if self.flat_ohe_features:
            if is_batch:
                loss_ohes = np.zeros(u_mapped.shape[0])
                for ohe_indices in self.flat_ohe_features:
                    sum_bits = np.sum(u_mapped[:, ohe_indices], axis=1)
                    is_invalid = sum_bits != 1
                    loss_ohes += is_invalid.astype(float)
                loss_ohe = self.weight_mutual_exclusion * loss_ohes
            else:
                loss_ohes = 0.0
                for ohe_indices in self.flat_ohe_features:
                    sum_bits = np.sum(u_mapped[ohe_indices])
                    if sum_bits != 1:
                        loss_ohes += 1.0
                loss_ohe = self.weight_mutual_exclusion * loss_ohes
        else:
            loss_ohe = 0.0

        # ---- Anomaly penalty ----
        if self.anomaly_model is not None:
            try:
                if is_batch:
                    scores = self.anomaly_model.anomaly_score(u_mapped)
                    loss_anomaly = self.weight_anomaly * np.maximum(0.0, scores - self.anomaly_threshold)
                else:
                    score = self.anomaly_model.anomaly_score(u_mapped.reshape(1, -1))[0]
                    loss_anomaly = self.weight_anomaly * max(0.0, score - self.anomaly_threshold)
            except Exception:
                loss_anomaly = 1e9
        else:
            loss_anomaly = 0.0

        total_loss = loss_distance + loss_target + loss_ohe + loss_anomaly
        return total_loss

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
        Generate a counterfactual explanation via optimisation.

        Parameters
        ----------
        test_window : pd.DataFrame
            The original window to explain.
        test_rul : float
            RUL at the end of the window (not used but kept for interface).
        tau : float or str
            Time point of interest.
        orig_surv : np.ndarray
            Survival curve of the test window.
        delta_1 : float
            Required gain in survival at tau.
        vehicle_id : str, optional
            Not used.
        window_start_idx : int, optional
            Not used.

        Returns
        -------
        dict
            Explanation details: changed features, new window, survival curves,
            gain, shape preservation status, deviation metrics, etc.
        """
        j = find_closest_time_index(self.time_points, tau)
        target_surv = orig_surv[j] + delta_1

        x_flat = flatten_window(test_window)

        # Build bounds for each element in the flattened array
        bounds = []
        for i in range(len(self.flat_feature_types)):
            if self.flat_feature_types[i] == 'bool':
                bounds.append((0.0, 1.0))
            else:
                bounds.append((self.x_min[i], self.x_max[i]))

        kwargs = {
            'x_flat': x_flat,
            'target_surv': target_surv,
            'j': j,
            'test_window': test_window
        }

        # Run optimisation
        best_flat = self.optimizer.minimize(
            func=self._loss,
            bounds=bounds,
            x_original=x_flat,
            **kwargs
        )

        # Post‑process: round booleans and enforce non‑actionable mask
        best_mapped = best_flat.copy()
        for i, t in enumerate(self.flat_feature_types):
            if t == 'bool':
                best_mapped[i] = 1.0 if best_flat[i] > 0.5 else 0.0
        best_mapped[~self.actionable_mask] = x_flat[~self.actionable_mask]

        best_window = self._reconstruct_window(best_mapped, test_window)

        # Compute survival curve for the best window
        try:
            best_surv = get_survival_curve(self.model, best_window, self.time_points)
            target_reached = best_surv[j] >= target_surv
        except Exception:
            best_surv = orig_surv
            best_window = test_window
            target_reached = False

        # Identify changed features
        changed_features = []
        for col in self.actionable_set:
            if not np.allclose(test_window[col].values, best_window[col].values):
                changed_features.append(col)

        # Evaluate shape preservation and compute deviation metrics
        shape_ok = is_shape_preserved(orig_surv, best_surv, self.dt, 0.05,
                                      self.shape_test, self.shape_delta)
        derivative_dev = shape_deviation_from_surv(orig_surv, best_surv, self.dt)
        abs_metrics = compute_absolute_deviation_metrics(orig_surv, best_surv)

        return {
            'explanation': changed_features,
            'position tau': j,
            'new_window': best_window,
            'original_survival': orig_surv,
            'new_survival': best_surv,
            'gain_at_tau': best_surv[j] - orig_surv[j],
            'target_reached': target_reached,
            'donor_window': None,
            'shape_preserved': shape_ok,
            'shape_deviation': float(derivative_dev),
            'mad': abs_metrics['mad'],
            'sad': abs_metrics['sad'],
            'mae': abs_metrics['mae'],
        }
