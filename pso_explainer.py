import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional
from utils.survival_utils import get_survival_curve, find_closest_time_index, flatten_window
from swap_evaluation import _first_derivatives, _second_derivatives, shape_deviation, _shape_preserved

class PSOExplainer:
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
        C: float = 1e6
    ):
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

        all_data = np.stack([w.values for w in training_windows])
        self.x_min = np.min(all_data, axis=0).ravel()
        self.x_max = np.max(all_data, axis=0).ravel()
        self.columns = training_windows[0].columns
        self.seq_len = training_windows[0].shape[0]

        self.actionable_mask = np.zeros(self.seq_len * len(self.columns), dtype=bool)
        for i, col in enumerate(self.columns):
            if col in self.actionable_set:
                self.actionable_mask[i::len(self.columns)] = True

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

        # Find z_ct
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

        # Initialize particles
        particles = np.zeros((self.N, len(x_flat)))
        for i in range(self.N):
            if i == 0 and z_ct is not None:
                particles[i] = z_ct
            else:
                particles[i] = np.random.uniform(self.x_min, self.x_max)
                particles[i][~self.actionable_mask] = x_flat[~self.actionable_mask]
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

        w_pso = 0.729
        c1 = 1.4945
        c2 = 1.4945

        def evaluate_particle(u):
            u_win = self._reconstruct_window(u, test_window)
            try:
                surv = get_survival_curve(self.model, u_win, self.time_points)
            except Exception:
                return np.inf, None
            dist = np.linalg.norm(u - x_flat)
            penalty = self.C * max(0, target_surv - surv[j])
            if self.preserve_shape:
                if not _shape_preserved(orig_d1, orig_d2, surv, self.dt, self.alpha):
                    penalty += self.C
            return dist + penalty, surv

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

            r1 = np.random.uniform(0, 1, size=particles.shape)
            r2 = np.random.uniform(0, 1, size=particles.shape)
            velocities = (w_pso * velocities + r1 * c1 * (personal_bests - particles) + r2 * c2 * (global_best - particles))
            particles = particles + velocities
            particles = np.clip(particles, self.x_min, self.x_max)
            for i in range(self.N):
                particles[i][~self.actionable_mask] = x_flat[~self.actionable_mask]

        target_reached = False
        if global_best_surv is not None and global_best_surv[j] >= target_surv:
            if not self.preserve_shape or _shape_preserved(orig_d1, orig_d2, global_best_surv, self.dt, self.alpha):
                target_reached = True
        target_reached = bool(target_reached)

        if global_best is None:
            best_window = test_window
            best_surv = orig_surv
        else:
            best_window = self._reconstruct_window(global_best, test_window)
            best_surv = global_best_surv if global_best_surv is not None else orig_surv

        dev = shape_deviation(orig_d1, orig_d2, best_surv, self.dt)
        shape_ok = bool(_shape_preserved(orig_d1, orig_d2, best_surv, self.dt, self.alpha))

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
            'shape_deviation': float(dev)
        }
