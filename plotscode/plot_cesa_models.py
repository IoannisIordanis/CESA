import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Styling
# ──────────────────────────────────────────────────────────────────────────────

SA_MODELS = ['CoxPH', 'DeepHIT', 'GradientBoostingSurvival', 'RSF']
SA_LABELS = ['CoxPH', 'DeepHIT', 'GBS', 'RSF']

CF_METHODS = ['CESA', 'NN', 'PSO', 'MPSO']
METHOD_COLORS = {
    'CESA': '#4C72B0',  # Muted Blue
    'NN':   '#DD8452',  # Muted Orange
    'PSO':  '#55A868',  # Muted Green
    'MPSO': '#C44E52'   # Muted Red
}
METHOD_MARKERS = {
    'CESA': 'o',
    'NN':   's',
    'PSO':  '^',
    'MPSO': 'D'
}

METRIC_CONFIGS = [
    # (column_name, higher_is_better, title_suffix, y_label, use_log)
    ('success_rate',             True,  'Success Rate',          'Success Rate (%)',        False),
    ('sparsity',                 True, 'Avg. Conciseness',      'Conciseness',             False),
    ('proximity',                False, 'Avg. Proximity',        'Proximity (L2)',          False),
    ('correlation_preservation', False, 'Avg. Corr. Preserv.',   'Corr. Preservation',      False),
    ('gen_time',                 False, 'Gen. Time',             'Generation Time (sec)',   True),
    ('shape_deviation',          False, 'Avg. Shape Deviation',  'Shape Deviation',         False)
]

def load_and_penalize_metrics(csv_path: str, metric: str, higher_is_better: bool, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[(df['rho'] != 'ALL') & (df['perc_d_1'] != 'ALL')]
    df = df[df['dataset'] == dataset]

    df['Counterfactual_model'] = df['explainer']
    df['SA_model'] = df['model'].apply(lambda x: x.split('_')[-1])

    if metric not in ['correlation_preservation', 'gen_time']:
        if higher_is_better:
            df[metric] = df[metric].fillna(0.0)
        else:
            worst = df[metric].max()
            if pd.notna(worst):
                df[metric] = df[metric].fillna(worst)
    else:
        if metric == 'gen_time':
            worst = df[metric].max()
            if pd.notna(worst):
                df[metric] = df[metric].fillna(worst)

    return df.dropna(subset=[metric])

def generate_model_performance_plot(csv_path: str = 'analysis_summary.csv', output_path: str = 'cesa_model_performance.png', include_scania: bool = False):
    if include_scania:
        output_path = output_path.replace(".png", "_with_scania.png").replace(".pdf", "_with_scania.pdf")
    sns.set_theme(style='ticks', context='paper')
    
    fig, axes = plt.subplots(nrows=3 if include_scania else 2, ncols=6, figsize=(14, 7.5 if include_scania else 5.0))
    fig.suptitle('Performance Across Survival Models', fontsize=14, fontweight='bold')
    
    datasets = ['PBC', 'HNEI', 'SCANIAn'] if include_scania else ['PBC', 'HNEI']
    box_width = 0.2
    n_methods = len(CF_METHODS)
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * box_width

    for row_idx, dataset in enumerate(datasets):
        for col_idx, (col_name, hib, title_suffix, y_label, use_log) in enumerate(METRIC_CONFIGS):
            ax = axes[row_idx, col_idx]
            
            df = load_and_penalize_metrics(csv_path, col_name, hib, dataset)

            for method_idx, method in enumerate(CF_METHODS):
                color = METHOD_COLORS[method]
                marker = METHOD_MARKERS[method]
                offset = offsets[method_idx]

                method_df = df[df['Counterfactual_model'] == method]

                group_data = []
                x_centers = []

                for i, sa_model in enumerate(SA_MODELS):
                    vals = method_df[method_df['SA_model'] == sa_model][col_name].values
                    if col_name == 'success_rate':
                        if len(vals) > 0 and vals.max() <= 1.0:
                            vals = vals * 100.0
                    group_data.append(vals)
                    x_centers.append(i + offset)

                # Draw boxplots
                ax.boxplot(
                    group_data,
                    positions=x_centers,
                    widths=box_width * 0.85,
                    patch_artist=True,
                    manage_ticks=False,
                    medianprops=dict(color='white', linewidth=1.5),
                    whiskerprops=dict(color=color, linewidth=1.0),
                    capprops=dict(color=color, linewidth=1.0),
                    flierprops=dict(marker='.', markerfacecolor=color, alpha=0.3, markersize=3, markeredgecolor='none'),
                    boxprops=dict(facecolor=color, alpha=0.6, linewidth=1.0, edgecolor=color),
                )

                # Connect medians with a line
                medians = [np.median(d) if len(d) > 0 else np.nan for d in group_data]
                ax.plot(
                    x_centers, medians,
                    color=color, marker=marker, linewidth=1.0,
                    markersize=3.5, zorder=5, label=method
                )

            # Axis customization
            ax.set_xticks(list(range(len(SA_MODELS))))
            ax.set_xticklabels(SA_LABELS, fontsize=7, rotation=30, ha='right')
            ax.set_title(f"{dataset} - {title_suffix}", fontsize=8, fontweight='bold', pad=3)
            ax.set_ylabel(y_label, fontsize=7.5)
            
            if use_log:
                ax.set_yscale('log')
                
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.spines[['top', 'right']].set_visible(False)
            
            ax.set_xlim(-0.5, len(SA_MODELS) - 0.5)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m], 
               lw=1.0, markersize=4.0, label=m)
        for m in CF_METHODS
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.05)
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15, hspace=0.6, wspace=0.38)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved: {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='analysis_summary.csv')
    parser.add_argument('--include-scania', action='store_true')
    args = parser.parse_args()
    csv_path = 'analysis_summary_with_scania.csv' if args.include_scania and args.csv == 'analysis_summary.csv' else args.csv
    generate_model_performance_plot(csv_path=csv_path, include_scania=args.include_scania)

