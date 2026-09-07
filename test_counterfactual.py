import random

import numpy as np
import pandas as pd
import pickle
import sys
import os
import time
import json
import math
import argparse

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
from surv_cf_explainer import SurvCFExplainer

# Make mlflow_rapper available in __main__ for unpickling
import __main__
__main__.mlflow_rapper = mlflow_rapper

# ========== CONFIGURATION ==========
# Choose which explainer to run by uncommenting the desired call in __main__
# ===================================

def extract_all_windows(dataset_dict, SEQ_LENGTH):
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
    c1 = robust_corr(w1)
    c2 = robust_corr(w2)
    return np.linalg.norm(c1 - c2, 'fro')

def proximity_metric(w1, w2):
    return np.linalg.norm(flatten_window(w1) - flatten_window(w2))

def prepare_dataset(SEQ_LENGTH, dataset_name="HNEI", MAX_TEST_WINDOWS=None, MIN_RUL=0):
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

    eligible = [(i, test_rul[i]) for i, ev in enumerate(test_event) if ev == 1 and test_rul[i] >= MIN_RUL]
    eligible.sort(key=lambda x: x[1], reverse=True)
    eligible_indices = [i for i, _ in eligible]

    if MAX_TEST_WINDOWS is not None:
        if len(eligible_indices) > MAX_TEST_WINDOWS:
            eligible_indices = random.sample(eligible_indices, MAX_TEST_WINDOWS)

    print(f"Eligible test windows: {len(eligible_indices)} (out of {len(test_windows)})")
    return actionable, eligible, eligible_indices, train_windows, train_rul, train_event, train_vehicle, train_local, test_windows, test_rul, test_event, test_vehicle, test_local

def load_model(model_path):
    model_path=model_path.replace("SCANIAn", "SCANIA")
    basename = os.path.basename(model_path)
    with open(f"pretrainedmodels/{basename}", "rb") as f:
        model = pickle.load(f)["wrapper_model"]

    # Mute verbosity for ensemble estimators (like RSF) to avoid terminal spam
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

def main_experiment_loop(explainer, RHO_LIST, DELTA_1_LIST, eligible_indices,
                         test_windows, test_rul, test_vehicle, test_local,
                         model, time_points):
    print("START MAIN")
    all_results = []
    total = len(RHO_LIST) * len(DELTA_1_LIST) * len(eligible_indices)
    counter = 0

    # Counters for candidate selection
    shape_candidate_count = 0
    gain_only_candidate_count = 0
    no_candidate_count = 0

    # Counters for final counterfactual outcomes
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
                res = explainer.explain(win, rul, tau, orig_surv, delta_1)
                gen_time = time.perf_counter() - t0

                j = res['position tau']
                surv_old = res['original_survival'][j]
                surv_new = res['new_survival'][j]

                # Count candidate categories
                had_shape_candidates = res.get('had_shape_candidates', False)
                no_candidates_found = res.get('no_candidates_found', True)
                if had_shape_candidates:
                    shape_candidate_count += 1
                elif not no_candidates_found:
                    gain_only_candidate_count += 1
                else:
                    no_candidate_count += 1

                # Count final counterfactual outcomes
                if res['target_reached']:
                    if res.get('shape_preserved', False):
                        shape_preserved_counterfactual_count += 1
                    else:
                        gain_only_counterfactual_count += 1
                else:
                    no_counterfactual_count += 1

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
                        'no_candidates_found': no_candidates_found
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
                        'no_candidates_found': no_candidates_found
                    })
                print(f"[{counter}/{total}] ρ={rho} δ₁={d1} veh={veh} win={local_idx} {'✓' if res['target_reached'] else '✗'}", flush=True)

    # #print summary with both candidate selection and final counterfactual outcomes
    #print("\n" + "="*60)
    #print(f"SUMMARY: Total test windows processed: {len(eligible_indices)} (across all (ρ,δ₁) combinations: {total} entries)")
    #print("  [Candidate selection]")
    #print(f"    - Combinations where shape‑preserving candidates were found and used: {shape_candidate_count}")
    #print(f"    - Combinations where only gain‑achieving candidates were found (shape not preserved): {gain_only_candidate_count}")
    #print(f"    - Combinations where no candidates were found: {no_candidate_count}")
    #print("-"*60)
    print("  [Final counterfactual outcomes]")
    #print(f"    - Combinations where shape‑preserving counterfactuals were found and used: {shape_preserved_counterfactual_count}")
    #print(f"    - Combinations where only gain‑achieving counterfactuals were found (shape not preserved): {gain_only_counterfactual_count}")
    #print(f"    - Combinations where no counterfactuals were found: {no_counterfactual_count}")
    #print("="*60)

    return all_results

def evaluate_CESA(RHO_LIST, DELTA_1_LIST, dataset_name, model_path, MAX_TEST_WINDOWS=None):
    model, time_points, dt = load_model(model_path=model_path)
    (actionable, eligible, eligible_indices, train_windows, train_rul, train_event, train_vehicle,
     train_local, test_windows, test_rul, test_event, test_vehicle, test_local) = prepare_dataset(
        SEQ_LENGTH=model.seq_length, dataset_name=dataset_name, MAX_TEST_WINDOWS=MAX_TEST_WINDOWS)

    ALPHA = 0.05
    K_CANDIDATES = 10
    BETA = 0.5
    M_NEIGHBOURS = 50

    candidate_index = CandidateIndex(train_windows, M=M_NEIGHBOURS)
    explainer = CounterfactualExplainer(
        model, train_windows, time_points, dt,
        alpha=ALPHA, beta=BETA, k=K_CANDIDATES, M=M_NEIGHBOURS,
        actionable_set=actionable, candidate_index=candidate_index
    )

    all_results = main_experiment_loop(
        explainer, RHO_LIST, DELTA_1_LIST, eligible_indices,
        test_windows, test_rul, test_vehicle, test_local,
        model, time_points
    )

    OUTPUT_JSON = f"exp_results_CESA_{dataset_name}_{model_path.split('_model')[0]}.json"
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")

def evaluate_PSO(RHO_LIST, DELTA_1_LIST, dataset_name, model_path, MAX_TEST_WINDOWS=None):
    model, time_points, dt = load_model(model_path=model_path)
    (actionable, eligible, eligible_indices, train_windows, train_rul, train_event, train_vehicle,
     train_local, test_windows, test_rul, test_event, test_vehicle, test_local) = prepare_dataset(
        SEQ_LENGTH=model.seq_length, dataset_name=dataset_name, MAX_TEST_WINDOWS=MAX_TEST_WINDOWS)

    ALPHA = 0.05
    N = 5
    N_iter = 100
    PRESERVE_SHAPE = False

    explainer = PSOExplainer(
        model, train_windows, time_points, dt,
        actionable_set=actionable, N=N, N_iter=N_iter,
        preserve_shape=PRESERVE_SHAPE, alpha=ALPHA, C=1e6
    )

    all_results = main_experiment_loop(
        explainer, RHO_LIST, DELTA_1_LIST, eligible_indices,
        test_windows, test_rul, test_vehicle, test_local,
        model, time_points
    )

    OUTPUT_JSON = f"exp_results_PSO_{dataset_name}_{model_path.split('_model')[0]}.json"
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")

def evaluate_NN(RHO_LIST, DELTA_1_LIST, dataset_name, model_path, MAX_TEST_WINDOWS=None):
    model, time_points, dt = load_model(model_path=model_path)
    (actionable, eligible, eligible_indices, train_windows, train_rul, train_event, train_vehicle,
     train_local, test_windows, test_rul, test_event, test_vehicle, test_local) = prepare_dataset(
        SEQ_LENGTH=model.seq_length, dataset_name=dataset_name, MAX_TEST_WINDOWS=MAX_TEST_WINDOWS)

    explainer = NNExplainer(
        model, train_windows, time_points, dt,
        actionable_set=actionable
    )

    all_results = main_experiment_loop(
        explainer, RHO_LIST, DELTA_1_LIST, eligible_indices,
        test_windows, test_rul, test_vehicle, test_local,
        model, time_points
    )

    OUTPUT_JSON = f"exp_results_NN_{dataset_name}_{model_path.split('_model')[0]}.json"
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")

def evaluate_surv_cf(RHO_LIST, DELTA_1_LIST, dataset_name="HNEI", model_path="pretrainedmodels/HNEI_CoxPH_model.pkl",
                     optimizer_name="pso", weight_target=1e5, weight_distance=1.0, weight_anomaly=1.0,
                     weight_mutual_exclusion=0.0, norm_distance=1, norm_target=2, hinge_target=True,
                     n_particles=5, n_iterations=100, patience=100, sa_T_start=10.0, sa_T_stop=0.001,
                     sa_step=0.01, sa_step_decrease_rate=0.0, da_maxiter=100,
                     MAX_TEST_WINDOWS=None, min_rul=0, output_path=None):
    model, time_points, dt = load_model(model_path=model_path)
    (actionable, eligible, eligible_indices, train_windows, train_rul, train_event, train_vehicle,
     train_local, test_windows, test_rul, test_event, test_vehicle, test_local) = prepare_dataset(
        SEQ_LENGTH=model.seq_length, dataset_name=dataset_name, MAX_TEST_WINDOWS=MAX_TEST_WINDOWS, MIN_RUL=min_rul)

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
        da_maxiter=da_maxiter
    )

    all_results = main_experiment_loop(
        explainer, RHO_LIST, DELTA_1_LIST,
        eligible_indices, test_windows, test_rul, test_vehicle, test_local,
        model, time_points)

    if output_path is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0].replace("_model", "")
        output_path = f"exp_results_SurvCF_{optimizer_name.upper()}_{model_name}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run counterfactual explanation experiments.")
    parser.add_argument("--explainer", type=str, default="cesa", choices=["cesa", "pso", "nn", "surv_cf"],
                        help="Explainer algorithm to run.")
    parser.add_argument("--dataset", type=str, default="HNEI",
                        help="Name of the dataset to evaluate (e.g., HNEI, PBC).")
    parser.add_argument("--model-path", type=str, default="HNEI_CoxPH_model.pkl",
                        help="Filename of the pretrained model in pretrainedmodels/ (e.g., HNEI_CoxPH_model.pkl).")
    parser.add_argument("--rho-list", type=str, default="0.3333333333333333,0.6666666666666666",
                        help="Comma-separated target times rho (or 'change_point').")
    parser.add_argument("--delta1-list", type=str, default="0.05,0.1,0.2,0.3,0.5",
                        help="Comma-separated delta percentages.")
    parser.add_argument("--max-test-windows", type=int, default=None,
                        help="Limit the number of test windows evaluated.")
    parser.add_argument("--min-rul", type=float, default=0.0,
                        help="Minimum RUL for test windows (used by surv_cf).")
    parser.add_argument("--optimizer", type=str, default="pso", 
                        help="Optimizer for surv_cf (e.g., pso, da, sa).")
    
    args = parser.parse_args()
    
    rho_list_str = args.rho_list.split(",")
    RHO_LIST = []
    for r in rho_list_str:
        if r.lower() == "change_point":
            RHO_LIST.append("change_point")
        else:
            RHO_LIST.append(float(r))
            
    DELTA_1_LIST = [float(x) for x in args.delta1_list.split(",")]
    
    if args.explainer == "cesa":
                        'shape_preserved': bool(res.get('shape_preserved', False)),
                        'shape_deviation': float(res.get('shape_deviation', 0.0)),
                        'had_shape_candidates': had_shape_candidates,
                        'no_candidates_found': no_candidates_found
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
                        'no_candidates_found': no_candidates_found
                    })
                print(f"[{counter}/{total}] ρ={rho} δ₁={d1} veh={veh} win={local_idx} {'✓' if res['target_reached'] else '✗'}", flush=True)

    # #print summary with both candidate selection and final counterfactual outcomes
    #print("\n" + "="*60)
    #print(f"SUMMARY: Total test windows processed: {len(eligible_indices)} (across all (ρ,δ₁) combinations: {total} entries)")
    #print("  [Candidate selection]")
    #print(f"    - Combinations where shape‑preserving candidates were found and used: {shape_candidate_count}")
    #print(f"    - Combinations where only gain‑achieving candidates were found (shape not preserved): {gain_only_candidate_count}")
    #print(f"    - Combinations where no candidates were found: {no_candidate_count}")
    #print("-"*60)
    print("  [Final counterfactual outcomes]")
    #print(f"    - Combinations where shape‑preserving counterfactuals were found and used: {shape_preserved_counterfactual_count}")
    #print(f"    - Combinations where only gain‑achieving counterfactuals were found (shape not preserved): {gain_only_counterfactual_count}")
    #print(f"    - Combinations where no counterfactuals were found: {no_counterfactual_count}")
    #print("="*60)

    return all_results

def evaluate_CESA(RHO_LIST, DELTA_1_LIST, dataset_name, model_path, MAX_TEST_WINDOWS=None):
    model, time_points, dt = load_model(model_path=model_path)
    (actionable, eligible, eligible_indices, train_windows, train_rul, train_event, train_vehicle,
     train_local, test_windows, test_rul, test_event, test_vehicle, test_local) = prepare_dataset(
        SEQ_LENGTH=model.seq_length, dataset_name=dataset_name, MAX_TEST_WINDOWS=MAX_TEST_WINDOWS)

    ALPHA = 0.05
    K_CANDIDATES = 10
    BETA = 0.5
    M_NEIGHBOURS = 50

    candidate_index = CandidateIndex(train_windows, M=M_NEIGHBOURS)
    explainer = CounterfactualExplainer(
        model, train_windows, time_points, dt,
        alpha=ALPHA, beta=BETA, k=K_CANDIDATES, M=M_NEIGHBOURS,
        actionable_set=actionable, candidate_index=candidate_index
    )

    all_results = main_experiment_loop(
        explainer, RHO_LIST, DELTA_1_LIST, eligible_indices,
        test_windows, test_rul, test_vehicle, test_local,
        model, time_points
    )

    OUTPUT_JSON = f"exp_results_CESA_{dataset_name}_{model_path.split('_model')[0]}.json"
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")

def evaluate_PSO(RHO_LIST, DELTA_1_LIST, dataset_name, model_path, MAX_TEST_WINDOWS=None):
    model, time_points, dt = load_model(model_path=model_path)
    (actionable, eligible, eligible_indices, train_windows, train_rul, train_event, train_vehicle,
     train_local, test_windows, test_rul, test_event, test_vehicle, test_local) = prepare_dataset(
        SEQ_LENGTH=model.seq_length, dataset_name=dataset_name, MAX_TEST_WINDOWS=MAX_TEST_WINDOWS)

    ALPHA = 0.05
    N = 5
    N_iter = 100
    PRESERVE_SHAPE = False

    explainer = PSOExplainer(
        model, train_windows, time_points, dt,
        actionable_set=actionable, N=N, N_iter=N_iter,
        preserve_shape=PRESERVE_SHAPE, alpha=ALPHA, C=1e6
    )

    all_results = main_experiment_loop(
        explainer, RHO_LIST, DELTA_1_LIST, eligible_indices,
        test_windows, test_rul, test_vehicle, test_local,
        model, time_points
    )

    OUTPUT_JSON = f"exp_results_PSO_{dataset_name}_{model_path.split('_model')[0]}.json"
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")

def evaluate_NN(RHO_LIST, DELTA_1_LIST, dataset_name, model_path, MAX_TEST_WINDOWS=None):
    model, time_points, dt = load_model(model_path=model_path)
    (actionable, eligible, eligible_indices, train_windows, train_rul, train_event, train_vehicle,
     train_local, test_windows, test_rul, test_event, test_vehicle, test_local) = prepare_dataset(
        SEQ_LENGTH=model.seq_length, dataset_name=dataset_name, MAX_TEST_WINDOWS=MAX_TEST_WINDOWS)

    explainer = NNExplainer(
        model, train_windows, time_points, dt,
        actionable_set=actionable
    )

    all_results = main_experiment_loop(
        explainer, RHO_LIST, DELTA_1_LIST, eligible_indices,
        test_windows, test_rul, test_vehicle, test_local,
        model, time_points
    )

    OUTPUT_JSON = f"exp_results_NN_{dataset_name}_{model_path.split('_model')[0]}.json"
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")

def evaluate_surv_cf(RHO_LIST, DELTA_1_LIST, dataset_name="HNEI", model_path="pretrainedmodels/HNEI_CoxPH_model.pkl",
                     optimizer_name="pso", weight_target=1e5, weight_distance=1.0, weight_anomaly=1.0,
                     weight_mutual_exclusion=0.0, norm_distance=1, norm_target=2, hinge_target=True,
                     n_particles=5, n_iterations=100, patience=100, sa_T_start=10.0, sa_T_stop=0.001,
                     sa_step=0.01, sa_step_decrease_rate=0.0, da_maxiter=100,
                     MAX_TEST_WINDOWS=None, min_rul=0, output_path=None):
    model, time_points, dt = load_model(model_path=model_path)
    (actionable, eligible, eligible_indices, train_windows, train_rul, train_event, train_vehicle,
     train_local, test_windows, test_rul, test_event, test_vehicle, test_local) = prepare_dataset(
        SEQ_LENGTH=model.seq_length, dataset_name=dataset_name, MAX_TEST_WINDOWS=MAX_TEST_WINDOWS, MIN_RUL=min_rul)

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
        da_maxiter=da_maxiter
    )

    all_results = main_experiment_loop(
        explainer, RHO_LIST, DELTA_1_LIST,
        eligible_indices, test_windows, test_rul, test_vehicle, test_local,
        model, time_points)

    if output_path is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0].replace("_model", "")
        output_path = f"exp_results_MPSO_{dataset_name}_{model_name}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run counterfactual explanation experiments.")
    parser.add_argument("--explainer", type=str, default="cesa", choices=["cesa", "pso", "nn", "mpso"],
                        help="Explainer algorithm to run.")
    parser.add_argument("--dataset", type=str, default="HNEI",
                        help="Name of the dataset to evaluate (e.g., HNEI, PBC).")
    parser.add_argument("--model-path", type=str, default="HNEI_CoxPH_model.pkl",
                        help="Filename of the pretrained model in pretrainedmodels/ (e.g., HNEI_CoxPH_model.pkl).")
    parser.add_argument("--rho-list", type=str, default="0.3333333333333333,0.6666666666666666",
                        help="Comma-separated target times rho (or 'change_point').")
    parser.add_argument("--delta1-list", type=str, default="0.05,0.1,0.2,0.3,0.5",
                        help="Comma-separated delta percentages.")
    parser.add_argument("--max-test-windows", type=int, default=None,
                        help="Limit the number of test windows evaluated.")
    parser.add_argument("--min-rul", type=float, default=0.0,
                        help="Minimum RUL for test windows (used by mpso).")
    parser.add_argument("--optimizer", type=str, default="pso", 
                        help="Optimizer for mpso (e.g., pso, da, sa).")
    
    args = parser.parse_args()
    
    rho_list_str = args.rho_list.split(",")
    RHO_LIST = []
    for r in rho_list_str:
        if r.lower() == "change_point":
            RHO_LIST.append("change_point")
        else:
            RHO_LIST.append(float(r))
            
    DELTA_1_LIST = [float(x) for x in args.delta1_list.split(",")]
    
    if args.explainer == "cesa":
        evaluate_CESA(RHO_LIST, DELTA_1_LIST, args.dataset, args.model_path, args.max_test_windows)
    elif args.explainer == "pso":
        evaluate_PSO(RHO_LIST, DELTA_1_LIST, args.dataset, args.model_path, args.max_test_windows)
    elif args.explainer == "nn":
        evaluate_NN(RHO_LIST, DELTA_1_LIST, args.dataset, args.model_path, args.max_test_windows)
    elif args.explainer == "mpso":
        evaluate_surv_cf(RHO_LIST, DELTA_1_LIST, dataset_name=args.dataset, model_path=args.model_path,
                         optimizer_name=args.optimizer, MAX_TEST_WINDOWS=args.max_test_windows, min_rul=args.min_rul)
