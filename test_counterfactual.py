import sys
sys.path.insert(0, '/home/ioannis/CESA-main/training_code')
# ==========================================

import numpy as np
import pandas as pd
import pickle
import sys
import os
import time
import json
import math
import argparse
import random
import pdmlabs
random.seed(42)
np.random.seed(42)

# Inject pdmlabs as OnlineADEngine to support legacy imports
try:
    import pdmlabs
    sys.modules['OnlineADEngine'] = pdmlabs
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import dataset_info, generic
from utils.survival_utils import get_survival_curve, flatten_window, calculate_tau, calculate_delta_1, robust_corr, mlflow_rapper
from counterfactual_explainer import CounterfactualExplainer
from pso_explainer import PSOExplainer
from nn_explainer import NNExplainer
from CESA.candidate_selection import CandidateIndex
from CESA.feature_grouping import group_features
from surv_cf_explainer import SurvCFExplainer


# Make mlflow_rapper available in __main__ for unpickling
import __main__
__main__.mlflow_rapper = mlflow_rapper


# ============================================================================
# Configuration
# ============================================================================

# Global toggles for the CESA explainer
USE_SHAPE_PRESERVATION = True      # Enforce shape preservation during search
USE_CORRELATION_PRESERVATION = True  # Consider correlation distance in rankings
USE_PROXIMITY = True               # Consider Euclidean distance in rankings
FIXED_POOL = False                 # Use fixed k-nearest candidates (True) or adaptive (False)
THETA_MODE = "fixed"               # Grouping threshold mode: 'fixed' (0.7), 'percentile', or 'group_size'

# Shape preservation test configuration
# Options:
#   'derivative'                 : two-tailed t-test on derivative differences
#   'mad_sad'                    : absolute survival deviation thresholds (MAD, SAD)
#   'derivative_tost_pct_max'    : TOST equivalence test with epsilon = SHAPE_DELTA * (1/dt)
SHAPE_TEST = "derivative_tost_pct_max"

# SHAPE_DELTA: tolerance margin for shape tests.
# For 'derivative_tost_pct_max', this is a percentage of the maximum possible slope (1/dt).
SHAPE_DELTA = 0.05


# ============================================================================
# Helper functions
# ============================================================================

def extract_all_windows(dataset_dict, SEQ_LENGTH):
    """Extract sliding windows, RUL, event flags, vehicle IDs, and local indices."""
    all_windows = []
    all_rul = []
    all_event = []
    all_vehicle = []
    all_local_idx = []
    for src_idx, (df, labels) in enumerate(zip(dataset_dict['target_data'],
                                               dataset_dict['target_labels'])):
        df_num = df.select_dtypes(include=[np.number]).copy()
        for i in range(0, len(df_num) - SEQ_LENGTH + 1, SEQ_LENGTH):
            window = df_num.iloc[i:i+SEQ_LENGTH]
            rul = labels[i+SEQ_LENGTH-1][0]
            event = labels[i+SEQ_LENGTH-1][1]
            all_windows.append(window)
            all_rul.append(rul)
            all_event.append(event)
            all_vehicle.append(f"vehicle_{src_idx}")
            all_local_idx.append(i)
    return all_windows, all_rul, all_event, all_vehicle, all_local_idx


def correlation_preservation(w1, w2):
    """Frobenius norm between correlation matrices of two windows."""
    c1 = robust_corr(w1)
    c2 = robust_corr(w2)
    return np.linalg.norm(c1 - c2, 'fro')


def proximity_metric(w1, w2):
    """Euclidean distance between flattened windows."""
    return np.linalg.norm(flatten_window(w1) - flatten_window(w2))


def compute_adaptive_theta(train_windows, actionable_set, mode="fixed"):
    """
    Compute the correlation threshold (theta) for feature grouping.
    Modes:
      - 'fixed': returns 0.7
      - 'percentile': 75th percentile of absolute correlations (min 0.5)
      - 'group_size': binary search to achieve average group size ~2.0
    """
    if mode == "fixed":
        return 0.7
    feat_list = sorted(actionable_set)
    if len(feat_list) < 2:
        return 0.7
    sample_size = min(len(train_windows), 500)
    indices = np.random.choice(len(train_windows), sample_size, replace=False)
    all_corrs = []
    for idx in indices:
        win = train_windows[idx]
        data = win[feat_list]
        corr_mat = robust_corr(data)
        upper_tri = corr_mat[np.triu_indices_from(corr_mat, k=1)]
        all_corrs.extend(np.abs(upper_tri))
    all_corrs = np.array(all_corrs)
    if mode == "percentile":
        theta = np.percentile(all_corrs, 75)
        theta = max(theta, 0.5)
        return float(theta)
    elif mode == "group_size":
        target_size = 2.0
        def avg_group_size_for_threshold(th):
            sizes = []
            eval_indices = np.random.choice(len(train_windows), min(len(train_windows), 100), replace=False)
            for wi in eval_indices:
                groups = group_features(train_windows[wi], actionable_set, threshold=th)
                if groups:
                    sizes.append(np.mean([len(g) for g in groups]))
                else:
                    sizes.append(1.0)
            return np.mean(sizes) if sizes else 1.0
        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = (lo + hi) / 2.0
            current_avg = avg_group_size_for_threshold(mid)
            if current_avg < target_size:
                hi = mid
            else:
                lo = mid
        theta = (lo + hi) / 2.0
        theta = max(0.3, min(0.95, theta))
        return float(theta)
    else:
        raise ValueError(f"Unknown theta mode: {mode}")


# ============================================================================
# Dataset preparation
# ============================================================================

def prepare_dataset(SEQ_LENGTH, dataset_name="HNEI", MAX_TEST_WINDOWS=None, MIN_RUL=0, theta_mode="fixed"):
    """Load dataset, extract windows, determine actionable features, and prepare vehicle history."""
    print("Loading data...")
    info = dataset_info[dataset_name]
    train_dataset, test_dataset = generic(info)

    non_actionable_cols = test_dataset.get('non_actionable_columns', set())
    if not isinstance(non_actionable_cols, set):
        non_actionable_cols = set(non_actionable_cols)

    print("Extracting windows from training set...")
    train_windows, train_rul, train_event, train_vehicle, train_local = extract_all_windows(train_dataset, SEQ_LENGTH)
    print(f"  -> {len(train_windows)} windows from {len(set(train_vehicle))} vehicles")

    print("Extracting windows from test set...")
    test_windows, test_rul, test_event, test_vehicle, test_local = extract_all_windows(test_dataset, SEQ_LENGTH)
    print(f"  -> {len(test_windows)} windows from {len(set(test_vehicle))} vehicles")

    all_columns = set(test_windows[0].columns)
    actual_non_actionable = non_actionable_cols.intersection(all_columns)
    actionable = all_columns - actual_non_actionable

    print(f"Non-actionable columns: {actual_non_actionable}")
    print(f"Actionable columns: {actionable}")

    theta = compute_adaptive_theta(train_windows, actionable, mode=theta_mode)
    print(f"Theta mode: {theta_mode} -> theta = {theta:.4f}")

    eligible = [(i, test_rul[i]) for i, ev in enumerate(test_event) if ev == 1 and test_rul[i] >= MIN_RUL]
    eligible.sort(key=lambda x: x[1], reverse=True)
    eligible_indices = [i for i, _ in eligible]

    if MAX_TEST_WINDOWS is not None:
        eligible_indices = eligible_indices[:MAX_TEST_WINDOWS]

    print(f"Eligible test windows: {len(eligible_indices)} (out of {len(test_windows)})")

    all_vehicles = train_vehicle + test_vehicle
    all_locals = train_local + test_local
    all_windows = train_windows + test_windows

    vehicle_to_past = {}
    for veh, idx, win in zip(all_vehicles, all_locals, all_windows):
        if veh not in vehicle_to_past:
            vehicle_to_past[veh] = []
        vehicle_to_past[veh].append((win, idx))

    for veh in vehicle_to_past:
        vehicle_to_past[veh].sort(key=lambda x: x[1])

    return (actionable, eligible, eligible_indices, train_windows, train_rul,
            train_event, train_vehicle, train_local, test_windows, test_rul,
            test_event, test_vehicle, test_local, vehicle_to_past, theta)


# ============================================================================
# Model loading
# ============================================================================

def load_model(model_path):
    """Load a pretrained survival model and extract time points and dt."""
    model_path = model_path.replace("SCANIAn", "SCANIA")

    import run_model

    with open(f"pretrainedmodels/{model_path}", "rb") as f:
        model = pickle.load(f)["wrapper_model"]

    # Silence verbose outputs from submodels
    if hasattr(model, 'model_per_source') and isinstance(model.model_per_source, dict):
        for src, submodel in model.model_per_source.items():
            if hasattr(submodel, 'verbose'):
                submodel.verbose = 0
    if hasattr(model, 'model') and hasattr(model.model, 'verbose'):
        model.model.verbose = 0
    if hasattr(model, 'verbose'):
        model.verbose = 0

    if hasattr(model, 'avail_times_per_source'):
        src = next(iter(model.avail_times_per_source))
        time_points = model.avail_times_per_source[src]
    elif hasattr(model, 'time_points'):
        time_points = model.time_points
    else:
        raise ValueError("Model does not have time_points or avail_times_per_source attribute.")
    
    dt = time_points[1] - time_points[0] if len(time_points) > 1 else 1.0
    return model, time_points, dt


# ============================================================================
# Main experiment loop
# ============================================================================

def main_experiment_loop(explainer, RHO_LIST, DELTA_1_LIST, eligible_indices,
                         test_windows, test_rul, test_vehicle, test_local,
                         model, time_points):
    """Run the counterfactual generation loop over all parameter combinations and test instances."""
    print("START MAIN")
    all_results = []
    total = len(RHO_LIST) * len(DELTA_1_LIST) * len(eligible_indices)
    counter = 0

    shape_candidate_count = 0
    gain_only_candidate_count = 0
    no_candidate_count = 0

    shape_preserved_counterfactual_count = 0
    gain_only_counterfactual_count = 0
    no_counterfactual_count = 0

    for rho in RHO_LIST:
        for d1 in DELTA_1_LIST:
            for idx in eligible_indices:
                counter += 1
                win = test_windows[idx]
                rul = test_rul[idx]
                veh = test_vehicle[idx]
                local_idx = test_local[idx]

                orig_surv = get_survival_curve(model, win, time_points)
                tau = calculate_tau(rho, time_points, orig_surv)
                delta_1 = calculate_delta_1(tau, orig_surv, d1, time_points)

                t0 = time.perf_counter()
                res = explainer.explain(
                    win, rul, tau, orig_surv, delta_1,
                    vehicle_id=veh,
                    window_start_idx=local_idx
                )
                gen_time = time.perf_counter() - t0

                res.setdefault('group_min', 0)
                res.setdefault('group_max', 0)
                res.setdefault('group_mean', 0.0)
                res.setdefault('chosen_group_count', 0)

                shape_relaxed = res.get('shape_relaxed', False)
                initial_pool_size = res.get('initial_pool_size', 0)
                final_pool_size = res.get('final_pool_size', 0)

                j = res['position tau']
                surv_old = res['original_survival'][j]
                surv_new = res['new_survival'][j]

                had_shape_candidates = res.get('had_shape_candidates', False)
                no_candidates_found = res.get('no_candidates_found', True)
                if had_shape_candidates:
                    shape_candidate_count += 1
                elif not no_candidates_found:
                    gain_only_candidate_count += 1
                else:
                    no_candidate_count += 1

                if res['target_reached']:
                    if res.get('shape_preserved', False):
                        shape_preserved_counterfactual_count += 1
                    else:
                        gain_only_counterfactual_count += 1
                else:
                    no_counterfactual_count += 1

                inf_time = 0.0

                if res['target_reached']:
                    t1 = time.perf_counter()
                    _ = get_survival_curve(model, res['new_window'], time_points)
                    inf_time = time.perf_counter() - t1
                    sparsity = len(res['explanation'])
                    prox = proximity_metric(win, res['new_window'])
                    corr_pres = correlation_preservation(win, res['new_window'])

                    changes = []
                    for f in res['explanation']:
                        changes.append({
                            'feature': f,
                            'old_values': win[f].tolist(),
                            'new_values': res['new_window'][f].tolist()
                        })

                    all_results.append({
                        'rho': rho,
                        'perc_d_1': d1,
                        'delta_1': float(delta_1),
                        'tau': float(tau),
                        'vehicle_id': veh,
                        'window_in_vehicle': int(local_idx),
                        'test_rul': float(rul),
                        'success': bool(res['target_reached']),
                        'old_surv_at_tau': float(surv_old),
                        'new_surv_at_tau': float(surv_new),
                        'gain_at_tau': float(res['gain_at_tau']),
                        'sparsity': sparsity,
                        'proximity': float(prox),
                        'correlation_preservation': float(corr_pres),
                        'gen_time': float(gen_time),
                        'inf_time': None if math.isnan(inf_time) else float(inf_time),
                        'changed_features': changes,
                        'phase': res.get('phase', 'unknown'),
                        'time_candidate_selection': float(res.get('time_candidate_selection', 0.0)),
                        'time_priority': float(res.get('time_priority', 0.0)),
                        'time_full': float(res.get('time_full', 0.0)),
                        'time_fallback': float(res.get('time_fallback', 0.0)),
                        'num_model_calls': res.get('num_model_calls', 0),
                        'total_model_time': float(res.get('total_model_time', 0.0)),
                        'num_shape_tests': res.get('num_shape_tests', 0),
                        'time_shape_tests': float(res.get('time_shape_tests', 0.0)),
                        'time_ann_queries': float(res.get('time_ann_queries', 0.0)),
                        'time_exact_distances': float(res.get('time_exact_distances', 0.0)),
                        'shape_preserved': bool(res.get('shape_preserved', False)),
                        'shape_deviation': float(res.get('shape_deviation', 0.0)),
                        'had_shape_candidates': had_shape_candidates,
                        'no_candidates_found': no_candidates_found,
                        'group_min': res.get('group_min', 0),
                        'group_max': res.get('group_max', 0),
                        'group_mean': res.get('group_mean', 0.0),
                        'chosen_group_count': res.get('chosen_group_count', 0),
                        'mae': float(res.get('mae', 0.0)),
                        'mad': float(res.get('mad', 0.0)),
                        'sad': float(res.get('sad', 0.0)),
                        'shape_relaxed': shape_relaxed,
                        'initial_pool_size': initial_pool_size,
                        'final_pool_size': final_pool_size,
                    })
                else:
                    all_results.append({
                        'rho': rho,
                        'perc_d_1': d1,
                        'delta_1': float(delta_1),
                        'tau': float(tau),
                        'vehicle_id': veh,
                        'window_in_vehicle': int(local_idx),
                        'test_rul': float(rul),
                        'success': bool(res['target_reached']),
                        'old_surv_at_tau': float(surv_old),
                        'had_shape_candidates': had_shape_candidates,
                        'no_candidates_found': no_candidates_found,
                        'group_min': res.get('group_min', 0),
                        'group_max': res.get('group_max', 0),
                        'group_mean': res.get('group_mean', 0.0),
                        'chosen_group_count': res.get('chosen_group_count', 0),
                        'mae': float(res.get('mae', 0.0)),
                        'mad': float(res.get('mad', 0.0)),
                        'sad': float(res.get('sad', 0.0)),
                        'shape_relaxed': shape_relaxed,
                        'initial_pool_size': initial_pool_size,
                        'final_pool_size': final_pool_size,
                    })
                print(f"[{counter}/{total}] ρ={rho} δ₁={d1} veh={veh} win={local_idx} {'✓' if res['target_reached'] else '✗'}", flush=True)

    print("  [Final counterfactual outcomes]")
    return all_results


# ============================================================================
# Evaluation wrappers for different explainers
# ============================================================================

def evaluate_CESA(RHO_LIST, DELTA_1_LIST, dataset_name, model_path, MAX_TEST_WINDOWS=None,
                  use_shape=True, use_corr=True, use_prox=True, fixed_pool=False, theta_mode="fixed",
                  shape_test="derivative", shape_delta=0.05):
    """Run experiments using the CESA (Counterfactual Explainer with Search and Aggregation) method."""
    model, time_points, dt = load_model(model_path=model_path)
    (actionable, eligible, eligible_indices, train_windows, train_rul, train_event,
     train_vehicle, train_local, test_windows, test_rul, test_event, test_vehicle,
     test_local, vehicle_past_windows, theta) = prepare_dataset(
        SEQ_LENGTH=model.seq_length, dataset_name=dataset_name, MAX_TEST_WINDOWS=MAX_TEST_WINDOWS,
        theta_mode=theta_mode)

    ALPHA = 0.05
    K_CANDIDATES = 20
    BETA = 0.5
    M_NEIGHBOURS = 20

    candidate_index = CandidateIndex(train_windows, M=M_NEIGHBOURS)
    explainer = CounterfactualExplainer(
        model, train_windows, time_points, dt,
        alpha=ALPHA, beta=BETA, k=K_CANDIDATES, M=M_NEIGHBOURS,
        actionable_set=actionable, candidate_index=candidate_index,
        use_shape=use_shape, use_corr=use_corr, use_prox=use_prox,
        fixed_pool=fixed_pool,
        vehicle_past_windows=vehicle_past_windows,
        grouping_threshold=theta,
        shape_test=shape_test,
        shape_delta=shape_delta
    )

    all_results = main_experiment_loop(
        explainer, RHO_LIST, DELTA_1_LIST, eligible_indices,
        test_windows, test_rul, test_vehicle, test_local,
        model, time_points
    )

    pool_suffix = "_fixedpool" if fixed_pool else ""
    theta_suffix = f"_theta_{theta_mode}"
    shape_suffix = f"_shape_{shape_test}"
    delta_suffix = f"_delta_{str(shape_delta).replace('.', '_')}"
    OUTPUT_JSON = f"exp_results_CESA_{dataset_name}_{model_path.split('_model')[0]}{pool_suffix}{theta_suffix}{shape_suffix}{delta_suffix}.json"
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")


def evaluate_PSO(RHO_LIST, DELTA_1_LIST, dataset_name, model_path, MAX_TEST_WINDOWS=None,
                 shape_test="derivative", shape_delta=0.05):
    """Run experiments using the PSO-based explainer."""
    model, time_points, dt = load_model(model_path=model_path)
    (actionable, eligible, eligible_indices, train_windows, train_rul, train_event,
     train_vehicle, train_local, test_windows, test_rul, test_event, test_vehicle,
     test_local, _, _) = prepare_dataset(
        SEQ_LENGTH=model.seq_length, dataset_name=dataset_name, MAX_TEST_WINDOWS=MAX_TEST_WINDOWS,
        theta_mode="fixed")

    ALPHA = 0.05
    N = 5
    N_iter = 100
    PRESERVE_SHAPE = False

    explainer = PSOExplainer(
        model, train_windows, time_points, dt,
        actionable_set=actionable, N=N, N_iter=N_iter,
        preserve_shape=PRESERVE_SHAPE, alpha=ALPHA, C=1e6,
        shape_test=shape_test,
        shape_delta=shape_delta
    )

    all_results = main_experiment_loop(
        explainer, RHO_LIST, DELTA_1_LIST, eligible_indices,
        test_windows, test_rul, test_vehicle, test_local,
        model, time_points
    )

    shape_suffix = f"_shape_{shape_test}"
    OUTPUT_JSON = f"exp_results_PSO_{dataset_name}_{model_path.split('_model')[0]}{shape_suffix}.json"
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")


def evaluate_NN(RHO_LIST, DELTA_1_LIST, dataset_name, model_path, MAX_TEST_WINDOWS=None,
                shape_test="derivative", shape_delta=0.05):
    """Run experiments using the nearest-neighbour explainer."""
    model, time_points, dt = load_model(model_path=model_path)
    (actionable, eligible, eligible_indices, train_windows, train_rul, train_event,
     train_vehicle, train_local, test_windows, test_rul, test_event, test_vehicle,
     test_local, _, _) = prepare_dataset(
        SEQ_LENGTH=model.seq_length, dataset_name=dataset_name, MAX_TEST_WINDOWS=MAX_TEST_WINDOWS,
        theta_mode="fixed")

    explainer = NNExplainer(
        model, train_windows, time_points, dt,
        actionable_set=actionable,
        alpha=0.05,
        shape_test=shape_test,
        shape_delta=shape_delta
    )

    all_results = main_experiment_loop(
        explainer, RHO_LIST, DELTA_1_LIST, eligible_indices,
        test_windows, test_rul, test_vehicle, test_local,
        model, time_points
    )

    shape_suffix = f"_shape_{shape_test}"
    OUTPUT_JSON = f"exp_results_NN_{dataset_name}_{model_path.split('_model')[0]}{shape_suffix}.json"
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")


def evaluate_surv_cf(RHO_LIST, DELTA_1_LIST, dataset_name="HNEI", model_path="pretrainedmodels/HNEI_CoxPH_model.pkl",
                     optimizer_name="pso", weight_target=1e5, weight_distance=1.0, weight_anomaly=1.0,
                     weight_mutual_exclusion=0.0, norm_distance=1, norm_target=2, hinge_target=True,
                     n_particles=5, n_iterations=100, patience=100, sa_T_start=10.0, sa_T_stop=0.001,
                     sa_step=0.01, sa_step_decrease_rate=0.0, da_maxiter=100,
                     MAX_TEST_WINDOWS=None, min_rul=0, output_path=None,
                     shape_test="derivative", shape_delta=0.05):
    """Run experiments using the generic optimization-based counterfactual explainer."""
    model, time_points, dt = load_model(model_path=model_path)
    (actionable, eligible, eligible_indices, train_windows, train_rul, train_event,
     train_vehicle, train_local, test_windows, test_rul, test_event, test_vehicle,
     test_local, _, _) = prepare_dataset(
        SEQ_LENGTH=model.seq_length, dataset_name=dataset_name, MAX_TEST_WINDOWS=MAX_TEST_WINDOWS, MIN_RUL=min_rul,
        theta_mode="fixed")

    explainer = SurvCFExplainer(
        model=model,
        training_windows=train_windows,
        time_points=time_points,
        dt=dt,
        actionable_set=actionable,
        optimizer_name=optimizer_name,
        norm_distance=norm_distance,
        norm_target=norm_target,
        weight_distance=weight_distance,
        weight_target=weight_target,
        weight_anomaly=weight_anomaly,
        weight_mutual_exclusion=weight_mutual_exclusion,
        hinge_target=hinge_target,
        n_particles=n_particles,
        n_iterations=n_iterations,
        patience=patience,
        sa_T_start=sa_T_start,
        sa_T_stop=sa_T_stop,
        sa_step=sa_step,
        sa_step_decrease_rate=sa_step_decrease_rate,
        da_maxiter=da_maxiter,
        shape_test=shape_test,
        shape_delta=shape_delta
    )

    all_results = main_experiment_loop(
        explainer, RHO_LIST, DELTA_1_LIST,
        eligible_indices, test_windows, test_rul, test_vehicle, test_local,
        model, time_points)

    if output_path is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0].replace("_model", "")
        shape_suffix = f"_shape_{shape_test}"
        output_path = f"exp_results_SurvCF_{optimizer_name.upper()}_{model_name}{shape_suffix}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Experiment parameters
    RHO_LIST = ["change_point"]
    DELTA_1_LIST = [0.25]

    # Shape delta values to sweep
    SHAPE_DELTA_LIST = [0.05]

    # CESA-specific parameters
    K_CANDIDATES = 20          # Number of candidate windows to retrieve
    SHAPE_TEST = "derivative_tost_pct_max"

    # Datasets and survival models to evaluate
    dataset_names = ["HNEI"]
    method_names = ["CoxPH"]

    for dataset in dataset_names:
        for method in method_names:
            for sd in SHAPE_DELTA_LIST:
                model_path = f"{dataset}_{method}_model.pkl"
                print(f"\n=== Running ablation: SHAPE_DELTA = {sd} ===")
                evaluate_CESA(
                    dataset_name=dataset,
                    model_path=model_path,
                    RHO_LIST=RHO_LIST,
                    DELTA_1_LIST=DELTA_1_LIST,
                    use_shape=True,
                    use_corr=USE_CORRELATION_PRESERVATION,
                    use_prox=USE_PROXIMITY,
                    fixed_pool=FIXED_POOL,
                    theta_mode=THETA_MODE,
                    shape_test=SHAPE_TEST,
                    shape_delta=sd
                )

            # Additional explainers (commented out)
            # evaluate_NN(dataset_name=dataset, model_path=model_path, RHO_LIST=RHO_LIST, DELTA_1_LIST=DELTA_1_LIST)
            # evaluate_PSO(dataset_name=dataset, model_path=model_path, RHO_LIST=RHO_LIST, DELTA_1_LIST=DELTA_1_LIST)
            # evaluate_surv_cf(dataset_name=dataset, model_path=model_path, RHO_LIST=RHO_LIST, DELTA_1_LIST=DELTA_1_LIST)
