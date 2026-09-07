#!/usr/bin/env python3
"""
analyze_results.py

Reads one or more JSON result files and computes aggregated statistics per:
- Dataset (extracted from filename)
- Explainer (CESA, PSO, NN)
- Model (e.g., HNEI_RSF, PBC_CoxPH)
- (rho, perc_d_1) combination

Usage:
    python analyze_results.py exp_results_*.json
    python analyze_results.py --dir results_folder
"""

import json
import os
import sys
import glob
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np

def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"File {filepath} does not contain a JSON list.")
    return data

def parse_filename(filepath):
    """
    Extract dataset, explainer, and model from filename.
    Expected pattern: exp_results_{explainer}_{dataset}_{model}.json
    Example: exp_results_CESA_HNEI_RSF.json -> ('HNEI', 'CESA', 'HNEI_RSF')
    Example: exp_results_PSO_PBC_GradientBoostingSurvival.json -> ('PBC', 'PSO', 'PBC_GradientBoostingSurvival')
    """
    base = os.path.basename(filepath).replace('.json', '')
    parts = base.split('_')
    if len(parts) < 3:
        # fallback: use whole name as model, unknown dataset/explainer
        return ('unknown', 'unknown', base)
    # First part after 'exp_results' is explainer
    # Usually pattern: exp_results_EXPLAINER_... so parts[2] is explainer? Actually:
    # parts[0] = 'exp', parts[1] = 'results', parts[2] = explainer
    # Example: ['exp', 'results', 'CESA', 'HNEI', 'RSF'] -> explainer='CESA', dataset='HNEI', model='HNEI_RSF'
    if len(parts) >= 4:
        explainer = parts[2]
        dataset = parts[3]
        model = '_'.join(parts[3:])  # everything from dataset onward
    else:
        explainer = parts[2] if len(parts) > 2 else 'unknown'
        dataset = 'unknown'
        model = base
    return (dataset, explainer, model)

def compute_averages(entries):
    n = len(entries)
    if n == 0:
        return {}
    
    success_entries = [e for e in entries if e.get('success', False)]
    n_success = len(success_entries)
    
    avg = {
        'count': n,
        'success_count': n_success,
        'success_rate': n_success / n if n > 0 else 0.0,
    }
    
    if n_success > 0:
        avg['sparsity'] = np.mean([e.get('sparsity', 0) for e in success_entries])
        avg['proximity'] = np.mean([e.get('proximity', 0.0) for e in success_entries])
        avg['correlation_preservation'] = np.mean([e.get('correlation_preservation', 0.0) for e in success_entries])
        avg['gen_time'] = np.mean([e.get('gen_time', 0.0) for e in success_entries])
        inf_times = [e.get('inf_time', 0.0) or 0.0 for e in success_entries]
        avg['inf_time'] = np.mean(inf_times)
        avg['shape_deviation'] = np.mean([e.get('shape_deviation', 0.0) for e in success_entries])
        avg['time_candidate_selection'] = np.mean([e.get('time_candidate_selection', 0.0) for e in success_entries])
        avg['time_priority'] = np.mean([e.get('time_priority', 0.0) for e in success_entries])
        avg['time_full'] = np.mean([e.get('time_full', 0.0) for e in success_entries])
        avg['time_fallback'] = np.mean([e.get('time_fallback', 0.0) for e in success_entries])
        avg['num_model_calls'] = np.mean([e.get('num_model_calls', 0) for e in success_entries])
        avg['num_shape_tests'] = np.mean([e.get('num_shape_tests', 0) for e in success_entries])
        
        # New: shape preservation count and rate
        shape_preserved_count = sum(1 for e in success_entries if e.get('shape_preserved', False))
        avg['shape_preserved_count'] = shape_preserved_count
        avg['shape_preserved_rate'] = shape_preserved_count / n_success if n_success > 0 else 0.0
    else:
        for metric in ['sparsity', 'proximity', 'correlation_preservation', 'gen_time', 'inf_time',
                       'shape_deviation', 'time_candidate_selection', 'time_priority', 'time_full',
                       'time_fallback', 'num_model_calls', 'num_shape_tests',
                       'shape_preserved_count', 'shape_preserved_rate']:
            avg[metric] = np.nan
    
    # Phase distribution (optional, keep as before)
    phases = [e.get('phase', 'none') for e in success_entries]
    for phase in ['priority', 'full', 'fallback', 'none']:
        avg[f'phase_{phase}_count'] = phases.count(phase)
        avg[f'phase_{phase}_rate'] = phases.count(phase) / n_success if n_success > 0 else 0.0
    
    return avg

def main():
    parser = argparse.ArgumentParser(description='Analyze counterfactual results JSON files.')
    parser.add_argument('files', nargs='*', help='JSON files to analyze')
    parser.add_argument('--dir', help='Directory containing JSON files')
    parser.add_argument('--output', default='analysis_summary.csv', help='Output CSV filename')
    args = parser.parse_args()
    
    filepaths = []
    if args.dir:
        filepaths = glob.glob(os.path.join(args.dir, '*.json'))
    elif args.files:
        # Expand wildcards if any (works on Windows with glob)
        for pattern in args.files:
            filepaths.extend(glob.glob(pattern))
    else:
        print("Please provide JSON files or a directory using --dir")
        sys.exit(1)
    
    if not filepaths:
        print("No JSON files found.")
        sys.exit(1)
    
    # Data structures: (dataset, explainer, model, rho, d1) -> list of entries
    aggregation = defaultdict(list)
    
    for fp in filepaths:
        print(f"Loading {fp}...")
        data = load_json(fp)
        dataset, explainer, model = parse_filename(fp)
        for entry in data:
            if entry.get('delta_1') == 0.0:
                continue
            rho = entry.get('rho', 'unknown')
            d1 = entry.get('perc_d_1', 'unknown')
            key = (dataset, explainer, model, rho, d1)
            aggregation[key].append(entry)
    
    # Build rows
    rows = []
    for (dataset, explainer, model, rho, d1), entries in aggregation.items():
        avg = compute_averages(entries)
        row = {
            'dataset': dataset,
            'explainer': explainer,
            'model': model,
            'rho': rho,
            'perc_d_1': d1,
            **avg
        }
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    # Reorder columns
    base_cols = ['dataset', 'explainer', 'model', 'rho', 'perc_d_1', 'count', 'success_count', 'success_rate']
    metric_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + metric_cols]
    df = df.sort_values(['dataset', 'explainer', 'model', 'rho', 'perc_d_1'])
    
    df.to_csv(args.output, index=False)
    print(f"Saved analysis to {args.output}")
    
    # Print summary per dataset + explainer + model (overall)
    print("\n=== Summary per (dataset, explainer, model) ===")
    for (dataset, explainer, model), group_df in df.groupby(['dataset', 'explainer', 'model']):
        # filter overall row (rho='all', perc_d_1='all') if present, otherwise aggregate
        overall = group_df[(group_df['rho'] == 'all') & (group_df['perc_d_1'] == 'all')]
        if overall.empty:
            # fallback: aggregate all rows for this group
            entries = []
            for _, row in group_df.iterrows():
                # We don't have original entries, so skip.
                continue
            print(f"Skipping {dataset}/{explainer}/{model} - no overall row")
        else:
            row = overall.iloc[0]
            print(f"\n{dataset} | {explainer} | {model}")
            print(f"  Total windows: {row['count']}")
            print(f"  Success rate: {row['success_rate']:.2%}")
            print(f"  Avg sparsity: {row['sparsity']:.2f}")
            print(f"  Avg proximity: {row['proximity']:.4f}")
            print(f"  Shape preserved rate: {row['shape_preserved']:.2%}")

if __name__ == '__main__':
    main()