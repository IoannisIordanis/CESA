import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Styling
# ──────────────────────────────────────────────────────────────────────────────

RHO_ORDER = ['0.3333333333333333', '0.6666666666666666', 'change_point']
RHO_LABELS = [r'$1/3$', r'$2/3$', 'CP']

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
    'MPSO': 'v'
}

METRIC_CONFIGS = [
    # (column_name, higher_is_better, title_suffix, y_label)
    ('success_rate',             True,  'Success Rate',          'Success (%)'),
    ('sparsity',                 True, 'Avg. Conciseness',      'Conciseness'),
    ('proximity',                False, 'Avg. Proximity',        'Proximity (L2)'),
    ('correlation_preservation', False, 'Avg. Corr. Preserv.',   'Corr. Preservation'),
    ('shape_deviation',          False, 'Avg. Shape Deviation',  'Shape Deviation')
]

# ──────────────────────────────────────────────────────────────────────────────
# Data Loading & Penalization
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

    return df.dropna(subset=[metric])

# ──────────────────────────────────────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_grid_plot(csv_path: str = 'analysis_summary.csv', output_path: str = 'rho_effect_grid.png', include_scania: bool = False):
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
    box_width = 0.15
    n_methods = len(CF_METHODS)
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * box_width

    for row_idx, dataset in enumerate(datasets):
        for col_idx, (col_name, hib, title_suffix, y_label) in enumerate(METRIC_CONFIGS):
            ax = axes[row_idx, col_idx]
            
            df = load_and_penalize_metrics(csv_path, col_name, hib, dataset)
            df['rho'] = df['rho'].astype(str)

            for method_idx, method in enumerate(CF_METHODS):
                color = METHOD_COLORS[method]
                marker = METHOD_MARKERS[method]
                offset = offsets[method_idx]

                method_df = df[df['Counterfactual_model'] == method]

                group_data = []
                x_centers = []

                for i, val in enumerate(RHO_ORDER):
                    vals = method_df[method_df['rho'] == val][col_name].values
                    if col_name == 'success_rate':
                        # Check if data is already in 0-100 or 0-1
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
                    medianprops=dict(color='white', linewidth=1.0),
                    whiskerprops=dict(color=color, linewidth=0.8),
                    capprops=dict(color=color, linewidth=0.8),
                    flierprops=dict(marker='.', markerfacecolor=color, alpha=0.3, markersize=2, markeredgecolor='none'),
                    boxprops=dict(facecolor=color, alpha=0.6, linewidth=0.8, edgecolor=color),
                )

                # Connect medians with a line
                medians = [np.median(d) if len(d) > 0 else np.nan for d in group_data]
                ax.plot(
                    x_centers, medians,
                    color=color, marker=marker, linewidth=1.0,
                    markersize=3.0, zorder=5, label=CF_LABELS[method]
                )

            # Axis customization
            ax.set_xticks(list(range(len(RHO_ORDER))))
            ax.set_xticklabels(RHO_LABELS, fontsize=6.5)
            ax.set_title(f"{dataset} - {title_suffix}", fontsize=7.5, fontweight='bold', pad=3)
            ax.set_ylabel(y_label, fontsize=7.0, labelpad=2)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.spines[['top', 'right']].set_visible(False)
            
            # Ensure nice margins
            ax.set_xlim(-0.5, len(RHO_ORDER) - 0.5)

    # Set x-label on the bottom row subplots
    for col_idx in range(5):
        axes[1, col_idx].set_xlabel(r'Eval. Curve Pos. ($\rho$)', fontsize=7.0, labelpad=3)

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
        fontsize=6.5,
        frameon=True,
        facecolor='#f9f9f9',
        edgecolor='#e0e0e0',
        bbox_to_anchor=(0.1, -0.015)
    )

    plt.tight_layout()
    # Leave space at the bottom for the legend
    plt.subplots_adjust(bottom=0.18, hspace=0.35, wspace=0.38)
    
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

