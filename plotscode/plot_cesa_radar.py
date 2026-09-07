import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Styling
# ──────────────────────────────────────────────────────────────────────────────

CF_METHODS = ['CESA', 'NN', 'PSO', 'MPSO']
METHOD_COLORS = {
    'CESA': '#4C72B0',  # Muted Blue
    'NN':   '#DD8452',  # Muted Orange
    'PSO':  '#55A868',  # Muted Green
    'MPSO': '#C44E52'   # Muted Red
}

METRIC_CONFIGS = [
    # (column_name, higher_is_better, label)
    ('success_rate',             True,  'Success\nRate'),
    ('sparsity',                 True, 'Conciseness'),
    ('proximity',                False, 'Proximity'),
    ('correlation_preservation', False, 'Correlation\nPreservation'),
    ('gen_time',                 False, 'Generation\nTime'),
    ('shape_deviation',          False, 'Shape\nDeviation')
]

def load_data(csv_path: str, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[(df['rho'] != 'ALL') & (df['perc_d_1'] != 'ALL')]
    df = df[df['dataset'] == dataset]
    df['Counterfactual_model'] = df['explainer']
    return df

def generate_radar_plot(csv_path: str = 'analysis_summary.csv', output_path: str = 'cesa_radar_chart.png', include_scania: bool = False):
    if include_scania:
        output_path = output_path.replace(".png", "_with_scania.png").replace(".pdf", "_with_scania.pdf")
    datasets = ['PBC', 'HNEI', 'SCANIAn'] if include_scania else ['PBC', 'HNEI']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 9.0 if include_scania else 6.0), subplot_kw=dict(polar=True))
    
    # Calculate angles for radar chart
    num_vars = len(METRIC_CONFIGS)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1] # Close the loop
    
    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        df = load_data(csv_path, dataset)
        
        # Precompute global min/max for normalization
        min_max = {}
        for metric, hib, _ in METRIC_CONFIGS:
            valid_data = pd.to_numeric(df[metric], errors='coerce').dropna()
            if len(valid_data) == 0:
                min_max[metric] = (0, 1)
            else:
                min_max[metric] = (valid_data.min(), valid_data.max())
        
        # Plot each method
        for method in CF_METHODS:
            method_df = df[df['Counterfactual_model'] == method]
            
            values = []
            for metric, hib, _ in METRIC_CONFIGS:
                metric_vals = pd.to_numeric(method_df[metric], errors='coerce').dropna()
                avg_val = metric_vals.mean() if len(metric_vals) > 0 else 0
                
                c_min, c_max = min_max[metric]
                # Avoid division by zero
                denom = (c_max - c_min) if (c_max - c_min) > 0 else 1.0
                
                # Normalize between 0.1 and 1.0 so center is 0, outer is 1.
                # If higher is better: score = (val - min) / denom
                # If lower is better: score = (max - val) / denom
                if hib:
                    norm_val = (avg_val - c_min) / denom
                else:
                    norm_val = (c_max - avg_val) / denom
                
                # Scale slightly so worst is 0.1, best is 1.0
                norm_val = 0.1 + (0.9 * norm_val)
                values.append(norm_val)
            
            values += values[:1] # Close the loop
            
            color = METHOD_COLORS[method]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=method, color=color)
            ax.fill(angles, values, color=color, alpha=0.15)
        
        # Axis customization
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw labels
        labels = [l for _, _, l in METRIC_CONFIGS]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
        
        # Hide y-tick labels and concentric circles
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        
        # Add a black hexagon for the outer border
        ax.plot(angles, [1.0]*len(angles), color='black', linewidth=1.5, zorder=0)
        
        # Add padding to push the letters outside the shape
        ax.tick_params(pad=20)
        
        ax.set_title(dataset, size=15, fontweight='bold', position=(0.5, 1.15), pad=20)
        
    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05), fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved: {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='analysis_summary.csv')
    parser.add_argument('--include-scania', action='store_true')
    args = parser.parse_args()
    csv_path = 'analysis_summary_with_scania.csv' if args.include_scania and args.csv == 'analysis_summary.csv' else args.csv
    generate_radar_plot(csv_path=csv_path, include_scania=args.include_scania)

