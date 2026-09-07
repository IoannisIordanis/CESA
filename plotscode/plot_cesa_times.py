import os
import json
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

METHOD_COLORS = {
    'CESA': '#4C72B0',  # Muted Blue
    'NN':   '#DD8452',  # Muted Orange
    'PSO':  '#55A868',  # Muted Green
    'MPSO': '#C44E52'   # Muted Red
}

def extract_times(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    times = []
    for d in data:
        if 'gen_time' in d and isinstance(d['gen_time'], (int, float)):
            gen_t = d['gen_time']
            inf_t = d.get('inf_time', 0.0)
            if not isinstance(inf_t, (int, float)):
                inf_t = 0.0
            times.append((gen_t, gen_t + inf_t))
    return times

def main():
    folder = 'CESATIMES'
    
    # regexes
    dim_pattern = re.compile(r'exp_results_(CESA|NN|PSO)_SCANIA(\d+)_SCANIA\d+_GradientBoostingSurvival\.json')
    win_pattern = re.compile(r'exp_results_(CESA|NN|PSO)_SCANIA_SCANIA_GradientBoostingSurvivald(\d+)\.json')
    survcf_dim_pattern = re.compile(r'exp_results_SurvCF_SCANIA(\d+)_GradientBoostingSurvival\.json')
    survcf_win_pattern = re.compile(r'exp_results_SurvCF_SCANIA_GradientBoostingSurvivald(\d+)\.json')
    
    dim_data = []
    win_data = []
    
    for fname in os.listdir(folder):
        if not fname.endswith('.json'): continue
        
        path = os.path.join(folder, fname)
        
        # Check standard dimension file
        m_dim = dim_pattern.match(fname)
        if m_dim:
            tech = m_dim.group(1)
            dim_val = int(m_dim.group(2))
            times = extract_times(path)
            for gen_t, total_t in times:
                dim_data.append({'Technique': tech, 'Dimension': dim_val, 'Generation Time (sec)': gen_t})
            continue
            
        # Check SurvCF dimension file
        m_survcf_dim = survcf_dim_pattern.match(fname)
        if m_survcf_dim:
            tech = 'MPSO'  # SurvCF maps to MPSO
            dim_val = int(m_survcf_dim.group(1))
            times = extract_times(path)
            for gen_t, total_t in times:
                dim_data.append({'Technique': tech, 'Dimension': dim_val, 'Generation Time (sec)': gen_t})
            continue
            
        # Check standard window size file
        m_win = win_pattern.match(fname)
        if m_win:
            tech = m_win.group(1)
            win_val = int(m_win.group(2))
            times = extract_times(path)
            for gen_t, total_t in times:
                win_data.append({'Technique': tech, 'Window Size': win_val, 'Generation Time (sec)': gen_t})
            continue

        # Check SurvCF window size file (just in case they exist or will be added)
        m_survcf_win = survcf_win_pattern.match(fname)
        if m_survcf_win:
            tech = 'MPSO'
            win_val = int(m_survcf_win.group(1))
            times = extract_times(path)
            for gen_t, total_t in times:
                win_data.append({'Technique': tech, 'Window Size': win_val, 'Generation Time (sec)': gen_t})
            continue
                
    df_dim = pd.DataFrame(dim_data)
    df_win = pd.DataFrame(win_data)
    
    # Sort them
    if not df_dim.empty:
        df_dim = df_dim.sort_values(by='Dimension')
    if not df_win.empty:
        df_win = df_win.sort_values(by='Window Size')

    sns.set_theme(style='ticks', context='paper', font_scale=1.8)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # ---------------------------------------------------------
    # Generation Time
    # ---------------------------------------------------------
    ax1 = axes[0]
    if not df_dim.empty:
        sns.boxplot(data=df_dim, x='Dimension', y='Generation Time (sec)', hue='Technique', palette=METHOD_COLORS, ax=ax1, 
                    boxprops=dict(alpha=0.7), showfliers=False)
        sns.pointplot(data=df_dim, x='Dimension', y='Generation Time (sec)', hue='Technique', palette=METHOD_COLORS, ax=ax1, 
                      markers='o', errorbar=None, estimator=np.median, dodge=0.4, scale=0.8)
    
    ax1.set_title('Effect of Dimensionality', fontweight='bold')
    ax1.set_ylabel('Generation Time (sec)')
    ax1.set_xlabel('Number of Features (Dimension)')
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    ax2 = axes[1]
    if not df_win.empty:
        sns.boxplot(data=df_win, x='Window Size', y='Generation Time (sec)', hue='Technique', palette=METHOD_COLORS, ax=ax2, 
                    boxprops=dict(alpha=0.7), showfliers=False)
        sns.pointplot(data=df_win, x='Window Size', y='Generation Time (sec)', hue='Technique', palette=METHOD_COLORS, ax=ax2, 
                      markers='o', errorbar=None, estimator=np.median, dodge=0.4, scale=0.8)
    
    ax2.set_title('Effect of Window Size', fontweight='bold')
    ax2.set_ylabel('Generation Time (sec)')
    ax2.set_xlabel('Window Size (Timesteps)')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Clean up legends: only keep one legend for the whole figure
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
            
    # Collect all unique techniques across both plots to create a global legend
    all_techs = []
    if not df_dim.empty:
        all_techs.extend(df_dim['Technique'].unique())
    if not df_win.empty:
        all_techs.extend(df_win['Technique'].unique())
    unique_techs = list(dict.fromkeys(all_techs)) # maintain order, remove duplicates
            
    handles = [plt.Line2D([0], [0], color=METHOD_COLORS[tech], lw=4, label=tech) for tech in unique_techs]
    fig.legend(handles=handles, loc='lower center', ncol=len(unique_techs), bbox_to_anchor=(0.5, -0.05), frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig('cesa_times_analysis.png', dpi=300, bbox_inches='tight')
    
    # Export median times to CSV
    if not df_dim.empty:
        df_dim_agg = df_dim.groupby(['Technique', 'Dimension'])['Generation Time (sec)'].median().reset_index()
        df_dim_agg.to_csv('cesa_times_dimension.csv', index=False)
        print("Saved cesa_times_dimension.csv")
        
    if not df_win.empty:
        df_win_agg = df_win.groupby(['Technique', 'Window Size'])['Generation Time (sec)'].median().reset_index()
        df_win_agg.to_csv('cesa_times_window.csv', index=False)
        print("Saved cesa_times_window.csv")
        
    print("Successfully generated and saved: cesa_times_analysis.png")

if __name__ == '__main__':
    main()
