import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Styling
# ──────────────────────────────────────────────────────────────────────────────

PERC_D_1_ORDER = ['0.05', '0.1', '0.25', '0.5']
PERC_D_1_LABELS = ['0.05', '0.1', '0.25', '0.5']

CF_METHODS = ['CESA', 'NN', 'PSO', 'MPSO']
CF_LABELS = {
    'CESA': 'CESA',
    'NN':   'NN',
    'PSO':  'PSO',
    'MPSO': 'MPSO'
}

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
    ('sparsity',                 True, 'Avg. Conciseness',      'Conciseness', False),
    ('proximity',                False, 'Avg. Proximity',        'Proximity (L2)',          False),
    ('correlation_preservation', False, 'Avg. Corr. Preserv.',   'Corr. Preservation',      False),
    ('shape_deviation',          False, 'Avg. Shape Deviation',  'Shape Deviation',         False)
]

# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_and_penalize_metrics(
    csv_path: str,
    metric: str,
    higher_is_better: bool,
    dataset: str
) -> pd.DataFrame:
    """Load results, filter by dataset, normalize model names, and penalize NaNs."""
    df = pd.read_csv(csv_path)

    # Strip summary rows
    df = df[(df['rho'] != 'ALL') & (df['perc_d_1'] != 'ALL')]
    df = df[df['dataset'] == dataset]

    # Map Counterfactual_model and SA_model
    df['Counterfactual_model'] = df['explainer']
    df['SA_model'] = df['model'].apply(lambda x: x.split('_')[-1])

    # Penalize failed runs (skip correlation_preservation and gen_time)
    if metric not in ['correlation_preservation', 'gen_time']:
        if higher_is_better:
            df[metric] = df[metric].fillna(0.0)
        else:
            worst = df[metric].max()
            if pd.notna(worst):
                df[metric] = df[metric].fillna(worst)
    else:
        # For gen_time, we just drop NaNs or fill with worst run if present
        if metric == 'gen_time':
            worst = df[metric].max()
            if pd.notna(worst):
                df[metric] = df[metric].fillna(worst)

    return df.dropna(subset=[metric])

# ──────────────────────────────────────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_grid_plot(csv_path: str = 'analysis_summary.csv', output_path: str = 'delta_effect_grid.png', include_scania: bool = False):
    if include_scania:
        output_path = output_path.replace(".png", "_with_scania.png").replace(".pdf", "_with_scania.pdf")
    sns.set_theme(style='ticks', context='paper')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'text.usetex': False,
        'axes.unicode_minus': True
    })

    fig, axes = plt.subplots(nrows=3 if include_scania else 2, ncols=5, figsize=(12.0, 6.0 if include_scania else 4.0), sharex=True)
    
    datasets = ['PBC', 'HNEI', 'SCANIAn'] if include_scania else ['PBC', 'HNEI']
    box_width = 0.16
    n_methods = len(CF_METHODS)
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * box_width

    for row_idx, dataset in enumerate(datasets):
        for col_idx, (col_name, hib, title_suffix, y_label, use_log) in enumerate(METRIC_CONFIGS):
            ax = axes[row_idx, col_idx]
            
            df = load_and_penalize_metrics(csv_path, col_name, hib, dataset)
            df['perc_d_1'] = df['perc_d_1'].astype(str)

            for method_idx, method in enumerate(CF_METHODS):
                color = METHOD_COLORS[method]
                marker = METHOD_MARKERS[method]
                offset = offsets[method_idx]

                method_df = df[df['Counterfactual_model'] == method]

                group_data = []
                x_centers = []

                for i, val in enumerate(PERC_D_1_ORDER):
                    vals = method_df[method_df['perc_d_1'] == val][col_name].values
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
                    markersize=3.5, zorder=5, label=CF_LABELS[method]
                )

            # Axis customization
            ax.set_xticks(list(range(len(PERC_D_1_ORDER))))
            ax.set_xticklabels(PERC_D_1_LABELS, fontsize=7)
            ax.set_title(f"{dataset} - {title_suffix}", fontsize=8, fontweight='bold', pad=3)
            ax.set_ylabel(y_label, fontsize=7.5)
            
            if use_log:
                ax.set_yscale('log')
                
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.spines[['top', 'right']].set_visible(False)
            
            # Ensure nice margins
            ax.set_xlim(-0.5, len(PERC_D_1_ORDER) - 0.5)

    # Set x-label on the bottom row subplots
    for col_idx in range(5):
        axes[1, col_idx].set_xlabel(r'Desired Survival Gain ($\delta_1$)', fontsize=7.5, labelpad=3)

    # Create a nice legend
    legend_elements = [
        Line2D([0], [0], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m], 
               lw=1.0, markersize=4.0, label=CF_LABELS[m])
        for m in CF_METHODS
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=7.5,
        frameon=True,
        facecolor='#f9f9f9',
        edgecolor='#e0e0e0',
        bbox_to_anchor=(0.5, -0.015)
    )

    plt.tight_layout()
    # Leave space at the bottom for the legend
    plt.subplots_adjust(bottom=0.18, hspace=0.3, wspace=0.38)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved: {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='analysis_summary.csv')
    parser.add_argument('--include-scania', action='store_true')
    args = parser.parse_args()
    csv_path = 'analysis_summary_with_scania.csv' if args.include_scania and args.csv == 'analysis_summary.csv' else args.csv
    generate_grid_plot(csv_path=csv_path, include_scania=args.include_scania)

