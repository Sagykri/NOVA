import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.figures.plotting_utils import save_plot, FONT_PATH
from src.figures.plot_config import PlotConfig

import numpy as np
import pandas as pd
import natsort

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm


import matplotlib
from matplotlib import font_manager as fm

fm.fontManager.addfont(FONT_PATH)
matplotlib.rcParams['font.family'] = 'Arial'

def plot_combined_effect_sizes_barplots(combined_effects_df, single_effects_df,saveroot:str,config_plot:PlotConfig, combine_on='batch'):
    
    for (baseline, pert), cur_df_combined in combined_effects_df.groupby(['baseline','pert']):
        if saveroot:
            savepath = os.path.join(saveroot, f'{pert}_vs_{baseline}_barplot')
        cur_df_single = single_effects_df[(single_effects_df.baseline==baseline)&(single_effects_df.pert==pert)]
        __plot_barplot(cur_df_combined, baseline, pert, savepath, config_plot, cur_df_single,
                       combine_on=combine_on)
        savepath = savepath.replace('barplot','forestplot')
        __plot_forest_plot(cur_df_combined, baseline, pert, savepath, config_plot, cur_df_single, config_plot.FIGSIZE,
                           combine_on=combine_on)

def __plot_barplot(combined_effects_df, baseline, pert, savepath, config_plot, cur_df_single,
                   combine_on = 'batch'):
    for pval_col in ['pvalue','p_heterogeneity']:
        combined_effects_df[f'stars_{pval_col}'] = combined_effects_df[f'adj_{pval_col}'].apply(__convert_pvalue_to_asterisks)
    
    combined_effects_df['low_error'] = combined_effects_df['combined_effect'] - combined_effects_df['ci_low']
    combined_effects_df['high_error'] = combined_effects_df['ci_upp'] - combined_effects_df['combined_effect']
        
    combined_effects_df = combined_effects_df.sort_values('combined_effect', ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(combined_effects_df['marker'], combined_effects_df['combined_effect'], 
            yerr = np.array([combined_effects_df['low_error'], combined_effects_df['high_error']]),
            capsize=5, edgecolor='k', color='gray',
            error_kw={'elinewidth': 0.7,'capthick': 0.7})

    # Add significance stars
    for index, row in combined_effects_df.iterrows():
        height = max(row['combined_effect'], row['ci_upp'],0)
        for pval_col,color, bonus in zip(['pvalue','p_heterogeneity'],['red','blue'],[0,0.1]):
            star = row[f'stars_{pval_col}']
            if star:
                ax.text(index, height+bonus, star, ha='center', 
                        fontsize=12, color=color)
    legend_elements = [
        Line2D([0], [0], color='red', lw=0, marker='*', label='adj p-value'),
        Line2D([0], [0], color='blue', lw=0, marker='*', label='adj p-heterogeneity'),
    ]

    # Add the separate effect sizes
    marker_order = combined_effects_df['marker']
    cur_df_single['marker'] = pd.Categorical(cur_df_single['marker'], categories=marker_order, ordered=True)
    cur_df_single = cur_df_single.sort_values('marker')

    combine_on_values = natsort.natsorted(cur_df_single[combine_on].unique())
    colors = cm.get_cmap('tab10', len(combine_on_values))

    for i, combine_on_value in enumerate(combine_on_values):
        df_single = cur_df_single[cur_df_single[combine_on] == combine_on_value]
        ax.plot(df_single['marker'].to_numpy(), df_single['effect_size'].to_numpy(),
                linestyle='None', marker='.', color=colors(i), markersize=3, label=combine_on_value)
        legend_elements.append(Line2D([0], [0], marker='.', color=colors(i), lw=0, label=f'{combine_on_value}'))

    # Aesthetics
    name_key=config_plot.MAPPINGS_ALIAS_KEY
    marker_name_color_dict = config_plot.COLOR_MAPPINGS_MARKERS
    x_ticklabels = [marker_name_color_dict[marker][name_key] if marker_name_color_dict else marker for marker in combined_effects_df['marker']]
    ax.set_xticklabels(x_ticklabels, rotation=90)#, ha='right')
    ax.set_ylabel("Combined Effect Size")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    ax.legend(handles=legend_elements, loc='best', frameon=False)
    plt.title(f'{config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION[pert][name_key]} vs {config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION[baseline][name_key]}')
    plt.tight_layout()
    
    if savepath:
        save_plot(fig, savepath, dpi=150, save_eps=True)
    else:
        plt.show()
    return

def __plot_forest_plot(combined_effects_df, baseline, pert, savepath, config_plot, cur_df_single, figsize=(5, 7),
                       combine_on='batch'):    
    combined_effects_df = combined_effects_df.sort_values('combined_effect', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    for _, row in combined_effects_df.iterrows():
        ax.errorbar(
            x=row["combined_effect"], y=row['marker'], fmt='o', #y=marker_index
            xerr=[[row["combined_effect"] - row["ci_low"]], 
                  [row["ci_upp"] - row["combined_effect"]]],
            color='black', capsize=3, lw=1)

    legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='Combined effect',
           markersize=6, linestyle='None'),]

    marker_order = combined_effects_df['marker']
    cur_df_single['marker'] = pd.Categorical(cur_df_single['marker'], categories=marker_order, ordered=True)
    cur_df_single = cur_df_single.sort_values('marker',ascending=True)
    combine_on_values = natsort.natsorted(cur_df_single[combine_on].unique())
    colors = cm.get_cmap('tab10', len(combine_on_values))

    for i, combine_on_value in enumerate(combine_on_values):
        df_single = cur_df_single[cur_df_single[combine_on] == combine_on_value]
        ax.plot(df_single['effect_size'].to_numpy(),df_single['marker'].to_numpy(),
                linestyle='None', marker='.', color=colors(i), markersize=3, label=combine_on_value)
        legend_elements.append(Line2D([0], [0], marker='.', color=colors(i), lw=0, label=f'{combine_on_value}'))

    # Aesthetics
    name_key=config_plot.MAPPINGS_ALIAS_KEY
    marker_name_color_dict = config_plot.COLOR_MAPPINGS_MARKERS
    y_ticklabels = [marker_name_color_dict[marker][name_key] if marker_name_color_dict else marker for marker in combined_effects_df['marker']]
    ax.set_yticks(range(len(combined_effects_df)))
    ax.set_yticklabels(y_ticklabels)#, ha='right')
    ax.axvline(0, color="gray", linestyle="--", zorder=0, lw=0.5)
    ax.set_xlabel("Effect Size (Log2FC)")

    ax.set_title(f'{config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION[pert][name_key]} vs {config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION[baseline][name_key]}')

    ax.legend(handles=legend_elements, bbox_to_anchor = (1.02,0.9), loc='lower left', frameon=False)
    
    if savepath:
        save_plot(fig, savepath, dpi=300, save_eps=True)
    else:
        plt.show()
    return

def plot_multiplex_forestplot(combined_effects_df,cur_df_single, savepath, config_plot, figsize=(5, 7),
                       combine_on='batch'):
    
    combined_effects_df = combined_effects_df.sort_values('combined_effect', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    for _, row in combined_effects_df.iterrows():
        ax.errorbar(
            x=row["combined_effect"], y=row['pert'], fmt='o',
            xerr=[[row["combined_effect"] - row["ci_low"]], 
                  [row["ci_upp"] - row["combined_effect"]]],
            color='black', capsize=3, lw=1)

    legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='Combined effect',
           markersize=6, linestyle='None'),]

    pert_order = combined_effects_df['pert']
    cur_df_single['pert'] = pd.Categorical(cur_df_single['pert'], categories=pert_order, ordered=True)
    cur_df_single = cur_df_single.sort_values('pert',ascending=True)
    combine_on_values = natsort.natsorted(cur_df_single[combine_on].unique())
    colors = cm.get_cmap('tab10', len(combine_on_values))

    for i, combine_on_value in enumerate(combine_on_values):
        df_single = cur_df_single[cur_df_single[combine_on] == combine_on_value]
        ax.plot(df_single['effect_size'].to_numpy(),df_single['pert'].to_numpy(),
                linestyle='None', marker='.', color=colors(i), markersize=3, label=combine_on_value)
        legend_elements.append(Line2D([0], [0], marker='.', color=colors(i), lw=0, label=f'{combine_on_value}'))

    # Aesthetics
    name_key=config_plot.MAPPINGS_ALIAS_KEY
    pert_name_color_dict = config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION
    y_ticklabels = [pert_name_color_dict[pert][name_key] if pert_name_color_dict else pert for pert in combined_effects_df['pert']]
    ax.set_yticks(range(len(combined_effects_df)))
    ax.set_yticklabels(y_ticklabels)#, ha='right')
    ax.axvline(0, color="gray", linestyle="--", zorder=0, lw=0.5)
    ax.set_xlabel("Effect Size (Log2FC)")

    ax.legend(handles=legend_elements, bbox_to_anchor = (1.02,0.9), loc='lower left', frameon=False)
    
    if savepath:
        savepath = os.path.join(savepath, f'SM_batches_forest')
        save_plot(fig, savepath, dpi=300, save_eps=True)
    else:
        plt.show()
    return

def plot_barplot_alyssa_old(effects_df, savepath, config_plot):
    effects_df[f'stars_pvalue'] = effects_df[f'adj_pvalue'].apply(__convert_pvalue_to_asterisks)

    name_key=config_plot.MAPPINGS_ALIAS_KEY
    color_key=config_plot.MAPPINGS_COLOR_KEY
    condition_name_color_dict = config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION
    condition_to_color = {key: value[color_key] for key, value in condition_name_color_dict.items()}

    # effects_df = effects_df.sort_values('marker', ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax = sns.barplot(data=effects_df, x='marker', y='effect_size', 
                         hue='pert', palette=condition_to_color)

    # Add significance stars
    for bar, (_, row) in zip(ax.patches, effects_df.iterrows()):
        height = bar.get_height()
        if row['stars_pvalue']:
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # center of the bar
                height + 0.01 * effects_df['effect_size'].max(),  # just above the bar
                row['stars_pvalue'],
                ha='center',
                va='bottom',
                fontsize=10,
            )

    # Aesthetics
    name_key=config_plot.MAPPINGS_ALIAS_KEY
    marker_name_color_dict = config_plot.COLOR_MAPPINGS_MARKERS
    x_ticklabels = [marker_name_color_dict[marker][name_key] if marker_name_color_dict else marker for marker in effects_df['marker']]
    ax.set_xticklabels(x_ticklabels, rotation=45)#, ha='right')
    ax.set_ylabel("Effect Size")
    ax.set_xlabel('Marker')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend(bbox_to_anchor=(1.5, 1), loc='upper right', borderaxespad=0, frameon=False)
    # plt.tight_layout()
    
    if savepath:
        save_plot(fig, savepath, dpi=150, save_eps=True)
    else:
        plt.show()
    return
            
def __convert_pvalue_to_asterisks(pval:float)->str:
    if pval <= 0.0001:
        return '****'
    elif pval <= 0.001:
        return '***'
    elif pval <= 0.01:
        return '**'
    elif pval <= 0.05:
        return '*'
    else:
        return ''