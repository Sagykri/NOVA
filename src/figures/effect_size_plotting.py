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
import scipy.stats as stats
from typing import Literal
import matplotlib
from matplotlib import font_manager as fm

import matplotlib.colors as mcolors

fm.fontManager.addfont(FONT_PATH)
matplotlib.rcParams['font.family'] = 'Arial'

def plot_combined_effect_sizes_forestplot(combined_effects_df, single_effects_df,saveroot:str,config_plot:PlotConfig, combine_on='batch'):
    for (baseline, pert), cur_df_combined in combined_effects_df.groupby(['baseline','pert']):
        if saveroot:
            savepath = os.path.join(saveroot, f'{pert}_vs_{baseline}_forestplot')
            
        cur_df_single = single_effects_df[(single_effects_df.baseline==baseline)&(single_effects_df.pert==pert)]
        __plot_forest_plot(cur_df_combined, cur_df_single, config_plot, baseline=baseline, pert=pert, savepath=savepath, figsize=config_plot.FIGSIZE,
                           combine_on=combine_on, unit='marker', show_only_significant=False)
        savepath += '_only_significant'
        __plot_forest_plot(cur_df_combined, cur_df_single, config_plot, baseline=baseline, pert=pert, savepath=savepath, figsize=config_plot.FIGSIZE,
                           combine_on=combine_on, unit='marker', show_only_significant=True)

def __plot_forest_plot(combined_effects_df, cur_df_single, config_plot, baseline=None, pert=None, savepath=None, figsize=None,
                       combine_on='batch', unit:Literal['marker', 'pert'] = 'marker', show_only_significant: bool = True, add_reproducability_table=True):    

    combined_effects_df = combined_effects_df.sort_values('combined_effect', ascending=True).reset_index(drop=True)

    if combined_effects_df.empty:
        raise ValueError("combined_effects_df is empty")

    if show_only_significant:
        keep = combined_effects_df['adj_pvalue'] <= 0.05
        combined_effects_df = combined_effects_df.loc[keep].copy()
        unit_markers = combined_effects_df[unit].tolist()
        cur_df_single = cur_df_single[cur_df_single[unit].isin(unit_markers)].copy()

        if combined_effects_df.empty:
            raise ValueError(f"No {unit}s remain after filtering by significance")

    figsize = (5, max(len(cur_df_single), 1) * 0.1) if figsize is None else figsize
    fig, ax = plt.subplots(figsize=figsize)

    # reserve some room on the right for the temp table showing tau and I2
    plt.subplots_adjust(right=0.78) 

    # Prepare y positions for markers
    unit_order = combined_effects_df[unit].tolist()
    row_gap = 3.35 # enlarge spacing between marker rows
    y_pos = np.arange(len(unit_order)) * row_gap
    y_map = {m: y_pos[i] for i, m in enumerate(unit_order)}

    # Create legend for combined effect
    legend_elements = [Line2D([0], [0], marker='D', color='black', label='Combined effect', markersize=5, linestyle='None'),]

    cur_df_single = cur_df_single.copy()
    cur_df_single[unit] = pd.Categorical(cur_df_single[unit], categories=unit_order, ordered=True)
    cur_df_single = cur_df_single.sort_values(unit, ascending=True)

    combine_on_values = natsort.natsorted(cur_df_single[combine_on].unique())
    
    palette = list(plt.get_cmap('tab10').colors)

    def _darker(color, factor=0.65):
        """Darken an RGB color by multiplying by `factor` (0–1)."""
        r, g, b = mcolors.to_rgb(color)
        return (r*factor, g*factor, b*factor)

    for i, combine_on_value in enumerate(combine_on_values):
        df_single = cur_df_single[cur_df_single[combine_on] == combine_on_value]

        x_vals = df_single['effect_size'].to_numpy()
        # Get y positions per unit
        base_y = df_single[unit].map(y_map).to_numpy()
        y_vals = base_y - (i+1)*0.15

        # Show CI per batch
        se = np.sqrt(df_single['variance'].to_numpy())  # Standard error (no need to divide by sqrt(n), variance already accounts for n since it's from bootstrapping)
        crit = stats.norm.isf(0.05 / 2)
        xerr = crit * se

        color = palette[i % len(palette)]

        marker_color   = _darker(color, 0.60)      # stronger dot (darker)
        whisker_ecolor = mcolors.to_rgba(color, 0.35)  # translucent whiskers

        # Show error bar for each batch
        ax.errorbar(
            x=x_vals, y=y_vals,
            xerr=xerr,
            markerfacecolor='none',
            fmt='.', color=marker_color, ecolor=whisker_ecolor, markersize=7, capsize=0, elinewidth=3, lw=0, label=None
        )
        ###

        # Add batch to legend
        legend_elements.append(Line2D([0], [0], marker='.', color=color, lw=0, 
            markerfacecolor='none', label=f'{combine_on_value}'))

    # Show combined effect for each unit
    for _, row in combined_effects_df.iterrows():
        y_top = y_map[row[unit]] + row_gap/2.0 - 0.7

        # Show error bar for combined effect (at the top of each row)
        ax.errorbar(
            x=row["combined_effect"], y=y_top, fmt='D', 
            xerr=[[row["combined_effect"] - row["ci_low"]], 
                    [row["ci_upp"] - row["combined_effect"]]],
            
            color='black', capsize=0,markersize=5, lw=1)

    # Aesthetics
    name_key=config_plot.MAPPINGS_ALIAS_KEY
    name_color_dict = config_plot.COLOR_MAPPINGS_MARKERS if unit == 'marker' else config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION

    # add marker names to list for y-axis ticks
    y_ticklabels = []
    for _, row in combined_effects_df.iterrows():
        m = row[unit]
        name = name_color_dict[m][name_key] 
        y_ticklabels.append(name)

    
    ymin = y_pos[0] - row_gap/2.0
    ymax = y_pos[-1] + row_gap/2.0
    ax.set_ylim(ymin, ymax)

    # Add alternating row colors for better readability
    row_colors = ("white", "#f2f2f2")  # white / light gray
    for i, y in enumerate(y_pos):
        y0, y1 = y - row_gap / 2.0, y + row_gap / 2.0
        ax.axhspan(y0, y1, facecolor=row_colors[i % 2], edgecolor="none", zorder=-10)

    # Show marker names on y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_ticklabels)

    # Show the 0 line
    ax.axvline(0, color="gray", linestyle="--", zorder=0, lw=0.5)

    ax.set_xlabel("Effect Size (Log2FC)")

    if baseline and pert:
        ax.set_title(f'{config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION[pert][name_key]} vs {config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION[baseline][name_key]}')

    ax.legend(handles=legend_elements, bbox_to_anchor = (1,0.75) if not add_reproducability_table else (2, 0.75), loc='lower left', frameon=False)


    # TODO: FOR debbugin only, Delete later
    if add_reproducability_table:

        # ============================
        # I² and τ “table”
        # ============================

        # use a dedicated slim axes to the right (not twinx), so nothing overlaps.
        # It shares the same vertical scale to align rows.
        main_pos = ax.get_position()
        table_width = 0.17  # fraction of figure width; fits into the 0.22 we left above
        table_left = 0.80   # just to the right of the main axes
        table_ax = fig.add_axes([table_left, main_pos.y0, table_width, main_pos.height])  


        # hide frame and ticks
        for spine in table_ax.spines.values():
            spine.set_visible(False)

        table_ax.tick_params(axis='both', which='both', length=0)
        table_ax.set_ylim(ymin, ymax)
        table_ax.set_xticks([])
        table_ax.set_yticks([])

        # Column headers (kept above, not inside any cell); no borders for header to avoid shifting alignment
        table_ax.text(0.35, ymax + 0.08*row_gap, 'I²', ha='center', va='bottom')
        table_ax.text(1.8, ymax + 0.08*row_gap, '(τ²_CI_low, τ², τ²_CI_upp)',  ha='center', va='bottom')
        table_ax.text(3.3, ymax + 0.08*row_gap, 'Q',  ha='center', va='bottom')

        # Draw bordered cells per row and write text inside
        col_lefts = [0.00, 0.70]     # left edge of I² and τ columns
        col_w     = 0.70             # each column spans half the table axis
        box_h     = row_gap * 0.90   # a bit of vertical padding within each row

        for i, row in combined_effects_df.iterrows():
            y = y_map[row[unit]]
        
            table_ax.text(0.35, y, f"{row['I2']:.2f}", ha='center', va='center')
            table_ax.text(1.8, y, f"({row['tau2_ci_low']:.4f}, {row['tau2']:.4f}, {row['tau2_ci_upp']:.4f})", ha='center', va='center')
            table_ax.text(3.3, y, f"{row['Q']:.4f}", ha='center', va='center')

    if savepath:
        save_plot(fig, savepath, dpi=300, save_eps=True)
    else:
        plt.show()

def plot_combined_effect_sizes_forestplot_multiplex(combined_effects_df, single_effects_df,saveroot:str,config_plot:PlotConfig, combine_on='batch'):
    if saveroot:
        savepath = os.path.join(saveroot, f'SM_batches_forest')

    __plot_forest_plot(combined_effects_df, single_effects_df, config_plot, savepath=savepath, figsize=config_plot.FIGSIZE,
                        combine_on=combine_on, unit='pert', show_only_significant=False)
    savepath += '_only_significant'
    __plot_forest_plot(combined_effects_df, single_effects_df, config_plot, savepath=savepath, figsize=config_plot.FIGSIZE,
                        combine_on=combine_on, unit='pert', show_only_significant=True)

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