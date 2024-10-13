import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.figures.plotting_utils import save_plot, FONT_PATH
from src.datasets.dataset_config import DatasetConfig
from src.figures.plot_config import PlotConfig
from src.common.utils import get_if_exists, save_config

import numpy as np
import pandas as pd
import logging
import math

from typing import List, Tuple
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import ttest_ind
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import nan_euclidean_distances

import matplotlib
from matplotlib import font_manager as fm

fm.fontManager.addfont(FONT_PATH)
matplotlib.rcParams['font.family'] = 'Arial'

def plot_distances_plots(distances:pd.DataFrame, config_data:DatasetConfig, config_plot:PlotConfig, saveroot:str,
                         metric:str='ARI_KMeansConstrained')->None:
    """Wrapper function to create the folder of distances plots and plot them

    Args:
        distances (pd.DataFrame): dataframe with calculated distances per marker
        config_data (DatasetConfig): dataset config
        config_plot (PlotConfig): plot config
        output_folder_path (str): root path to save the plots and configuration in
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.
    """
    if saveroot:
        os.makedirs(saveroot, exist_ok=True)
        save_config(config_data, saveroot)
        save_config(config_plot, saveroot)
    
    plot_marker_ranking(distances, saveroot, config_data, config_plot, metric=metric, show_effect_size=True)
    plot_clustermap(distances, saveroot, config_data, config_plot, metric=metric)
    plot_bubble_plot(distances, saveroot, config_data, config_plot, metric=metric)

def plot_marker_ranking(distances:pd.DataFrame, saveroot:str, config_data:DatasetConfig,
                        config_plot:PlotConfig, metric:str='ARI_KMeansConstrained', show_effect_size:bool=False)->None:
    """Generate and save a boxplot of marker distances with p-values, separately for each condition.

    Args:
        distances (pd.DataFrame): Distances between conditions and baseline for each marker
        The DataFrame will have the following columns:
            - 'marker': The marker for which the distance was calculated on.
            - 'batch': The batch for which the distance was calculated on.
            - 'repA': The rep of the condition data
            - 'repB': The rep of the baseline data
            - 'distance_metric': The calculated distance metric (eg 'ARI_KMeansConstrained').
        saveroot (str): Path to the folder where the plot should be saved
        config_data (DatasetConfig): dataset config
        config_plot (PlotConfig): plot config
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.
        show_effect_size (bool, optional): If True, effect sizes are displayed on the plot. Defaults to False.
    """
    logging.info(f"[plot_marker_ranking]")
    baseline:str = config_data.BASELINE_CELL_LINE_CONDITION
    conditions:List[str] = config_data.CELL_LINES_CONDITIONS

    pvalues_df = __calculate_marker_pvalue_per_condition(distances, baseline, conditions, metric, __cliffs_delta)
    for cond in conditions:
        savepath = None
        show_baseline = get_if_exists(config_plot, 'SHOW_BASELINE', True)
        if saveroot:
            savepath = os.path.join(saveroot, f'{cond}_vs_{baseline}_boxplot') 
            if not show_baseline:
                savepath = f'{savepath}_without_baseline'
        upper_graph_ylim = get_if_exists(config_plot, 'UPPER_GRAPH_YLIM', None)
        lower_graph_ylim = get_if_exists(config_plot, 'LOWER_GRAPH_YLIM', None)
        __plot_boxplot(distances, baseline, cond, metric, 
                     pvalues_df, config_plot, show_effect_size=show_effect_size, savepath = savepath,
                     upper_graph_ylim=upper_graph_ylim, 
                     lower_graph_ylim=lower_graph_ylim, show_baseline=show_baseline)
        
def plot_clustermap(distances:pd.DataFrame, saveroot:str, config_data:DatasetConfig, config_plot:PlotConfig, metric:str='ARI_KMeansConstrained',
                    cmap:str='Blues_r', figsize:Tuple[int,int]=(10, 10))->None:
    """Generate and save a clustermap of marker p-values per condition.

    Args:
        distances (pd.DataFrame): Distances between conditions and baseline for each marker
        The DataFrame will have the following columns:
            - 'marker': The marker for which the distance was calculated on.
            - 'batch': The batch for which the distance was calculated on.
            - 'repA': The rep of the condition data
            - 'repB': The rep of the baseline data
            - 'distance_metric': The calculated distance metric (eg 'ARI_KMeansConstrained').
        saveroot (str): Path to the folder where the plot should be saved
        config_data (DatasetConfig): dataset config
        config_plot (PlotConfig): plot config
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.
        cmap (str): name of colormap for the heatmap
        figsize (Tuple[int,int]): figure size
    """
    logging.info(f"[_plot_clustermap]")
    baseline:str = config_data.BASELINE_CELL_LINE_CONDITION
    conditions:List[str] = config_data.CELL_LINES_CONDITIONS

    # Calculate marker p-values and process the dataframe
    pvalues_df = __calculate_marker_pvalue_per_condition(distances, baseline, conditions, metric, __cliffs_delta)  
    # we want our data organzied as marker in the rows and condition in the columns (values are the pvalues)
    marker_pvalue_per_condition = pvalues_df.pivot(index='marker', columns='condition', values='pvalue')
    
    # Perform hierarchical clustering
    marker_linkage = __calculate_hierarchical_clustering(marker_pvalue_per_condition)
    condition_linkage = __calculate_hierarchical_clustering(marker_pvalue_per_condition.T)
    
    # Determine clustering parameters
    to_cluster_markers = True if marker_linkage is not None else False
    to_cluster_conditions = True if condition_linkage is not None else False
    na_mask = marker_pvalue_per_condition.isnull()
    
    # Generate the clustermap
    clustermap = sns.clustermap(marker_pvalue_per_condition,mask=na_mask, cmap=cmap, figsize=figsize,
                                row_cluster=to_cluster_markers, col_cluster=to_cluster_conditions, row_linkage=marker_linkage, 
                                col_linkage=condition_linkage, yticklabels=True, xticklabels=True, 
                                annot=True, vmax=0.05)

    ## Optional: Highlight significant values
    # for i in range(clustered_df.shape[0]):
    #     for j in range(clustered_df.shape[1]):
    #         if clustered_df.iloc[i, j] <= 0.05:
    #             # Add a rectangle around the cell
    #             g.ax_heatmap.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='tomato', lw=2))
    baseline = __convert_labels(clustermap, baseline, config_plot)

    plt.title(f'vs {baseline}')
    
    # save the plot
    savepath = None
    if saveroot:
        savepath = os.path.join(saveroot, f'vs_{baseline}_clustermap') 
    if savepath:
        save_plot(clustermap, savepath, dpi=100, save_eps=True)
    else:
        plt.show()
    return
      
def plot_bubble_plot(distances:pd.DataFrame, saveroot:str, config_data:DatasetConfig, config_plot:PlotConfig, 
                     metric:str='ARI_KMeansConstrained', effect_cmap:str = 'Blues', vmin_d:int =-1, vmax_d:int =10)->None:
    """Generate and save a bubble plot of marker p-values and effect size per condition.
    Args:
        distances (pd.DataFrame): Distances between conditions and baseline for each marker
        The DataFrame will have the following columns:
            - 'marker': The marker for which the distance was calculated on.
            - 'batch': The batch for which the distance was calculated on.
            - 'repA': The rep of the condition data
            - 'repB': The rep of the baseline data
            - 'distance_metric': The calculated distance metric (eg 'ARI_KMeansConstrained').
        saveroot (str): Path to the folder where the plot should be saved
        config_data (DatasetConfig): dataset config
        config_plot (PlotConfig): plot config
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.
        effect_cmap (str, optional): Colormap for the bubble_plot. Defaults to 'Blues'.
        vmin_d (int, optional): Minimum value of the effect (for normalization). Defaults to -1.
        vmax_d (int, optional): Maximum value of the effect (for normalization). Defaults to 10.
    """

    logging.info(f"[_plot_bubble_plot]")
    baseline:str = config_data.BASELINE_CELL_LINE_CONDITION
    conditions:List[str] = config_data.CELL_LINES_CONDITIONS

    # Calculate marker p-values and process the dataframe
    pvalues_df = __calculate_marker_pvalue_per_condition(distances, baseline, conditions, metric, __cliffs_delta)  
    __sort_markers_by_linkage(pvalues_df)
    conditions_order = get_if_exists(config_plot, 'ORDERED_CELL_LINES_NAMES',None)
    if conditions_order is not None:
        pvalues_df['condition'] = pd.Categorical(pvalues_df['condition'], categories=conditions_order, ordered=True)
        pvalues_df.sort_values(by=['condition'])

    # Normalize effect sizes and adjust p-values for visualization
    norm_d = mcolors.Normalize(vmin=vmin_d, vmax=vmax_d) if vmin_d is not None else None
    pvalues_df['log_pvalue'] = -np.log10(__bin_pvalues(pvalues_df['pvalue']))

    # Plot
    fig_width = int(len(conditions)*4/3)
    fig_height = max(int(np.unique(pvalues_df.marker).shape[0]/4),3)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    scatter = sns.scatterplot(
        data=pvalues_df,
        x="condition",
        y="marker",
        size='log_pvalue',
        hue='d',
        hue_norm = norm_d,
        legend='full',
        palette=effect_cmap,
        sizes=(1, 200),
        edgecolor='black',
        linewidth=0.5,
        )
    
    baseline = __convert_labels(scatter, baseline, config_plot)
    plt.xticks(rotation=90)
    plt.title(f'vs {baseline}')

    __customize_bubbleplot_legend(scatter, effect_cmap, norm_d)

    savepath = None
    if saveroot:
        savepath = os.path.join(saveroot, f'vs_{baseline}_bubbleplot')
    if savepath:
        save_plot(fig, savepath, dpi=100, save_eps=True)
    else:
        plt.show()
    return 

def __calculate_marker_pvalue_per_condition(distances:pd.DataFrame, baseline:str, conditions:List[str], 
                                            metric:str, effect_size_function:callable)->pd.DataFrame:
    """
    Calculate the statistical significance and effect size for the difference in distances 
    between a baseline condition and all other conditions, across all markers.

    Args:
        distances (pd.DataFrame): A DataFrame containing distance measurements for various markers across conditions.
        The DataFrame will have the following columns:
            - 'marker': The marker for which the distance was calculated on.
            - 'batch': The batch for which the distance was calculated on.
            - 'repA': The rep of the condition data
            - 'repB': The rep of the baseline data
            - 'distance_metric': The calculated distance metric (eg 'ARI_KMeansConstrained').
        baseline (str): The name of the baseline condition against which comparisons are made.
        conditions (List[str]): List of the conditions to comapre to the baseline
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.

    Returns:
        pd.DataFrame: A DataFrame with calculated p-values and effect sizes as columns,
                      indicating the significance of differences between the baseline and each condition.
    """
    marker_pvalue_per_condition = None
    for cond in conditions:
        marker_pval = __calc_pvalue_and_effect(distances, baseline=baseline, condition=cond, metric=metric,
                                               effect_size_function=effect_size_function)        
        cur_df = pd.DataFrame(marker_pval).T.reset_index(names='marker')
        cur_df['condition'] = cond
        if marker_pvalue_per_condition is None:
            marker_pvalue_per_condition = cur_df
        else:
            marker_pvalue_per_condition = pd.concat([marker_pvalue_per_condition, cur_df], ignore_index=True)
    return marker_pvalue_per_condition

def __plot_boxplot(distances:pd.DataFrame, baseline:str, condition:str, 
                  metric:str, pvalues_df:pd.DataFrame, config_plot:PlotConfig, show_effect_size:bool=False,
                  savepath:str=None, upper_graph_ylim:Tuple[float,float]=None, figsize:Tuple[int,int]=(12,3),
                  lower_graph_ylim:Tuple[float,float]=None, show_baseline:bool=True)->None:
    """
    Plot a boxplot to visualize the distribution and significance of distances for a given condition compared to a baseline.
    The markers are displayed in descending order of their distance from baseline. Optionally, effect sizes can be shown.

    Args:
        distances (pd.DataFrame): DataFrame containing the distance metrics
        The DataFrame will have the following columns:
            - 'marker': The marker for which the distance was calculated on.
            - 'batch': The batch for which the distance was calculated on.
            - 'repA': The rep of the condition data
            - 'repB': The rep of the baseline data
            - 'distance_metric': The calculated distance metric (eg 'ARI_KMeansConstrained').
        baseline (str): Name of the baseline condition for comparison.
        condition (str): Name of the condition to compare against the baseline.
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.
        pvalues_df (pd.DataFrame): A DataFrame with calculated p-values and effect sizes as columns,
                                    indicating the significance of differences between the baseline and each condition.
        config_plot (PlotConfig): plot config
        show_effect_size (bool, optional): If True, effect sizes are displayed on the plot. Defaults to False.
        savepath (str, optional): File path to save the plot. If None, the plot is shown but not saved. Defaults to None.
    """
    # Filter and sort the data
    cur_distances=distances[distances.condition==condition] # for sorting we want only the condition distances
    median_variance = cur_distances.groupby("marker")[metric].agg(['median', 'var']) # ordering by median, then variance.
    pvalues_df['significant'] = np.where(pvalues_df.pvalue<=0.05,1,0)
    median_variance = median_variance.merge(pvalues_df[pvalues_df.condition==condition][['marker','significant']], left_index=True, right_on='marker')
    median_variance = median_variance.sort_values(by=['significant','median', 'var'], ascending=[False, False,True]).reset_index(drop=True)
    ####cliffs delta version:
    # median_variance = median_variance.merge(pvalues_df[pvalues_df.condition==condition][['marker','d']], left_index=True, right_on='marker')
    # median_variance = median_variance.sort_values(by=['d'], ascending=[False]).reset_index(drop=True)

    dists_order = median_variance.marker.values # do the ordering
    if show_baseline:
        cur_distances=distances[distances.condition.isin([baseline,condition])] # after sorting, we can include also the baseline distances
    # Plotting
    marker_name_color_dict = config_plot.COLOR_MAPPINGS_MARKERS
    name_key=config_plot.UMAP_MAPPINGS_ALIAS_KEY
    color_key=config_plot.UMAP_MAPPINGS_COLOR_KEY
    condition_name_color_dict = config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION
    condition_to_color = {key: value[color_key] for key, value in condition_name_color_dict.items()}
    if not upper_graph_ylim: # case where we don't split the y axis
        fig = plt.figure(figsize=figsize)
        boxplot=sns.boxplot(data=cur_distances, order=dists_order, hue='condition',
                x='marker', y=metric, fliersize=0, palette=condition_to_color)
    
        labels = []
        # Add pavlues
        for i, marker in enumerate(dists_order):
            cur_marker = pvalues_df[(pvalues_df.condition==condition)&(pvalues_df.marker==marker)]
            marker_pvalue = cur_marker.pvalue.values[0]
            __add_pvalue(marker, i, dists_order, marker_pvalue, show_baseline, ax=boxplot)
            effect_size_formatted = round(cur_marker.d.values[0],2)
            
            label = marker_name_color_dict[marker][name_key] if marker_name_color_dict else marker
            if show_effect_size:
                label = f'{label} (d={effect_size_formatted})'
            labels.append(label)

        boxplot.set_xticklabels(labels,rotation=90)
        
        # add dashed line between significants
        if  median_variance[median_variance['significant'] == 0].shape[0]>0:
            first_non_sig = median_variance[median_variance['significant'] == 0].iloc[0].name
            boxplot.axvline(x=(first_non_sig+ first_non_sig-1)/2, color='k', linestyle='--', linewidth=1)
            boxplot.axvline(x=(first_non_sig+ first_non_sig-1)/2, color='k', linestyle='--',linewidth=1)

        plt.ylabel('ARI')
        plt.xlabel('Markers')
        plt.title(f'{config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION[condition][name_key]} vs {config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION[baseline][name_key]}')
        _, current_labels = boxplot.get_legend_handles_labels()
        updated_labels = [config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION[label][name_key] for label in current_labels]
        legend = boxplot.legend_
        legend.set_title("Conditions")
        for text, new_label in zip(legend.get_texts(), updated_labels):
            text.set_text(new_label)

        # Remove the bottom and right spines
        boxplot.spines['right'].set_visible(False)
        boxplot.spines['top'].set_visible(False)

        # Remove the box around the legend
        legend.set_frame_on(False)
    
    else: #break the y axis
        fig, axs = plt.subplots(figsize=figsize, nrows=2)
        fig.subplots_adjust(hspace=0.0)

        ax_lower = axs[1]
        ax_upper = axs[0]

        # upper graph
        sns.boxplot(ax=ax_upper, data=cur_distances, order=dists_order, hue='condition',
            x='marker', y=metric, fliersize=0, palette=condition_to_color)
        ax_upper.set_ylim(upper_graph_ylim[0], upper_graph_ylim[1])
        ax_upper.set_xlim(-1,dists_order.shape[0])
        ax_upper.set_ylabel(None)
        ax_upper.spines['bottom'].set_visible(False)
        ax_upper.spines['top'].set_visible(False)
        ax_upper.spines['right'].set_visible(False)
        ax_upper.set_xlabel(None)
        ax_upper.set_xticks('')

        # lower graph
        sns.boxplot(ax=ax_lower, data=cur_distances, order=dists_order, hue='condition',
            x='marker', y=metric, fliersize=0, palette=condition_to_color)
        ax_lower.set_ylim(lower_graph_ylim[0], lower_graph_ylim[1])
        ax_lower.set_xlim(-1, dists_order.shape[0])
        max_tick = math.floor(lower_graph_ylim[1]*10) + 1
        ax_lower.set_yticks([round(i * 0.1, 1) for i in range(max_tick)])
        ax_lower.set_ylabel(None)

        ax_lower.legend().remove()
        ax_lower.spines['top'].set_visible(False)
        ax_lower.spines['right'].set_visible(False)
        ax_lower.set_xlabel("Markers")

        # Diagonal lines to indicate the "cut"
        d = .01  # Size of diagonal lines
        kwargs = dict(transform=ax_upper.transAxes, color='k', clip_on=False)
        ax_upper.plot((-d, +d), (-d, +d), **kwargs)  # Top-left diagonal line

        kwargs.update(transform=ax_lower.transAxes)  # Switch to the lower plot
        ax_lower.plot((-d, +d), (0.9 - d, 0.9 + d), **kwargs)  # Bottom-left diagonal line

        # add dashed line between significants
        first_non_sig = median_variance[median_variance['significant'] == 0].iloc[0].name
        ax_lower.axvline(x=(first_non_sig+ first_non_sig-1)/2, color='k', linestyle='--', linewidth=1)
        ax_upper.axvline(x=(first_non_sig+ first_non_sig-1)/2, color='k', linestyle='--',linewidth=1)
        
        labels = []
        # Add pavlues
        for i, marker in enumerate(dists_order):
            cur_marker = pvalues_df[(pvalues_df.condition==condition)&(pvalues_df.marker==marker)]
            marker_pvalue = cur_marker.pvalue.values[0]
            if marker_pvalue <= 0.05:
                __add_pvalue(marker, i, dists_order, marker_pvalue, show_baseline, upper_graph_ylim=upper_graph_ylim, ax_upper=ax_upper, ax_lower=ax_lower)
            
            label = marker_name_color_dict[marker][name_key] if marker_name_color_dict else marker
            effect_size_formatted = round(cur_marker.d.values[0],2)
            if show_effect_size:
                label = f'{label} (d={effect_size_formatted})'
            labels.append(label)
        
        ax_lower.set_xticklabels(labels, rotation=90) # update x axis labels
        plt.text(-2.5, 0.25, 'ARI',rotation =90, fontsize=10)
        plt.suptitle(f'{config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION[condition][name_key]} vs {config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION[baseline][name_key]}')

        # update legend labels
        _, current_labels = ax_upper.get_legend_handles_labels()
        updated_labels = [config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION[label][name_key] for label in current_labels]
        legend = ax_upper.legend_
        legend.set_title("Conditions")
        for text, new_label in zip(legend.get_texts(), updated_labels):
            text.set_text(new_label)
        legend.set_frame_on(False)

    if savepath:
        save_plot(fig, savepath, dpi=100, save_eps=True)
    else:
        plt.show()
    return

def __sort_markers_by_linkage(pvalues_df:pd.DataFrame)->None:
    # we want our data organzied as marker in the rows and condition in the columns (values are the pvalues)
    marker_pvalue_per_condition = pvalues_df.pivot(index='marker', columns='condition', values='pvalue')
    
    # Perform hierarchical clustering
    marker_linkage = __calculate_hierarchical_clustering(marker_pvalue_per_condition)
    marker_order = __get_order_from_linkage(marker_linkage, np.unique(pvalues_df.marker))

    pvalues_df['marker'] = pd.Categorical(pvalues_df['marker'], categories=marker_order, ordered=True)

    # Sort the DataFrame based on the hierarchical clustering order
    pvalues_df.sort_values(by=['marker'])
    return None

def __convert_labels(plot, baseline:str, config_plot:PlotConfig)->str:
    """
    Given a plot object, edit the marker and condition labels using the config_plot.

    Returns:
        str: The edited baseline string.
    """
    marker_name_color_dict = config_plot.COLOR_MAPPINGS_MARKERS
    condition_name_color_dict = config_plot.COLOR_MAPPINGS_CELL_LINE_CONDITION
    name_key=config_plot.UMAP_MAPPINGS_ALIAS_KEY
    if not marker_name_color_dict:
        return baseline

    if isinstance(plot, sns.matrix.ClusterGrid): # when plot is a clustermap it's a clustergrid so we need to extract the heatmap ax.
        plot = plot.ax_heatmap

    marker_labels = plot.get_yticklabels()
    marker_labels = [marker_name_color_dict[label.get_text()][name_key] for label in marker_labels]
    ytick_positions = plot.get_yticks()
    plot.set_yticks(ytick_positions)
    plot.set_yticklabels(marker_labels)
    
    condition_labels = plot.get_xticklabels()
    condition_labels = [condition_name_color_dict[label.get_text()][name_key] for label in condition_labels]
    xtick_positions = plot.get_xticks()
    plot.set_xticks(xtick_positions)
    plot.set_xticklabels(condition_labels)
    
    baseline = condition_name_color_dict[baseline][name_key]
    return baseline

def __calc_pvalue_and_effect(distances:pd.DataFrame, baseline:str, condition:str, 
                             metric:str, effect_size_function:callable)->dict:
    """Calculate the significance and the effect size of the difference between the baseline distances and the condition distances, for each marker.

    Args:
        distances (pd.DataFrame): Distances between conditions and baseline for each marker
        The DataFrame will have the following columns:
            - 'marker': The marker for which the distance was calculated on.
            - 'batch': The batch for which the distance was calculated on.
            - 'repA': The rep of the condition data
            - 'repB': The rep of the baseline data
            - 'distance_metric': The calculated distance metric (eg 'ARI_KMeansConstrained').
        baseline (str): The name of the 'cell_line_condition' which is the baseline in the calculations
        condition (str): The name of the 'cell_line_condition' to compare with the baseline
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.

    Returns:
        dict: Dictionary containing for each marker its pvalue and effect size
    """
    marker_pval = {}
    for marker, marker_distances in distances.groupby('marker'):
        if marker_distances.shape[0]==0:
            continue
        baseline_distances = marker_distances[marker_distances.condition == baseline][metric]
        condition_distances = marker_distances[marker_distances.condition == condition][metric]
        pval = ttest_ind(condition_distances, baseline_distances,alternative = 'greater')[1]

        marker_pval[marker]={}
        marker_pval[marker]['pvalue'] = pval
        
        # calc effect size
        d = effect_size_function(condition_distances, baseline_distances)

        marker_pval[marker]['d']=d
        
    return marker_pval

def __calc_cohens_d(x, y)->float:
    """Calculate Cohen's d for two independent samples.

    Args:
        x (array-like): First sample data.
        y (array-like): Second sample data.

    Returns:
        float: Cohen's d value.
    """
    # Calculate the means of the two groups
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculate the standard deviations of the two groups
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)

    # Calculate the pooled standard deviation
    n_x = len(x)
    n_y = len(y)
    pooled_std = np.sqrt(((n_x - 1) * std_x**2 + (n_y - 1) * std_y**2) / (n_x + n_y - 2))

    # Calculate Cohen's d
    d = (mean_x - mean_y) / pooled_std

    return d

def __cliffs_delta(x, y):
    n_x = len(x)
    n_y = len(y)
    comparisons = np.sum([np.sign(a - b) for a in x for b in y])
    delta = comparisons / (n_x * n_y)
    return delta

def __add_pvalue(marker:str, marker_index:int, dists_order:List[str], pvalue:float, show_baseline:bool=True, upper_graph_ylim:Tuple[float,float]=None,
                 ax_upper=None, ax_lower=None, ax=None)->None:
    # find the highest bar between the baseline and condition for the given marker
    if not upper_graph_ylim:
        patches = ax.patches[:-1-int(show_baseline)]
    else:
        patches = ax_upper.patches[:-2]
    
    height_1 = max(patches[marker_index].get_path().vertices[:, 1])
    if show_baseline:
        height_2 = max(patches[marker_index+len(dists_order)].get_path().vertices[:, 1])
        pval_loc = max(height_1, height_2)
    else:
        pval_loc = height_1

    if pvalue <= 0.05: # plot significant asterix
        asterisks = __convert_pvalue_to_asterisks(pvalue)
        # Add the asterisks above the box
        if not upper_graph_ylim:
            plt.text(list(dists_order).index(marker), pval_loc, asterisks, 
                    ha='center', va='bottom', fontsize=10)
        else:
            if pval_loc > upper_graph_ylim[0]:
                ax = ax_upper
            else:
                ax = ax_lower
            ax.text(list(dists_order).index(marker), pval_loc, asterisks, 
                ha='center', va='bottom', fontsize=8)
            
def __convert_pvalue_to_asterisks(pval:float)->str:
    if pval <= 0.0001:
        asterisks = '****'
    elif pval <= 0.001:
        asterisks = '***'
    elif pval <= 0.01:
        asterisks = '**'
    else:
        asterisks = '*'
    
    return asterisks

def __bin_pvalues(pvalues):
    """Adjust p-values and bin them for better visualization."""
    adjusted_pvalues = np.where(pvalues > 0.06, 0.06, pvalues)
    adjusted_pvalues = np.where((0.05 <= adjusted_pvalues) & (adjusted_pvalues < 0.06), 0.05, adjusted_pvalues)
    adjusted_pvalues = np.where((0.01 <= adjusted_pvalues) & (adjusted_pvalues < 0.05), 0.01, adjusted_pvalues)
    adjusted_pvalues = np.where((0.0001 <= adjusted_pvalues) & (adjusted_pvalues < 0.01), 0.0001, adjusted_pvalues)
    return np.where(adjusted_pvalues < 0.0001, 10**math.floor(np.log10(adjusted_pvalues.min())), adjusted_pvalues)

def __get_order_from_linkage(linkage, items):
    """Get the order of items based on hierarchical clustering."""
    if linkage is not None:
        dendro = dendrogram(linkage, no_plot=True)
        return items[dendro['leaves']]
    return items

def __customize_bubbleplot_legend(scatter, effect_cmap:str, norm_d:Normalize):
    """Customize the legend in the bubble plot.
    """
    # Get the current legend handles and labels from the scatter plot
    handles, labels = scatter.get_legend_handles_labels()

    # Remove the existing legend from the scatter plot (it's showing both log pvalues and effect size as binary values)
    scatter.legend_.remove()

    # First part of legend: p-values 
    # keep only handles and labels of the 'log_pvalue' label
    handles = handles[labels.index("log_pvalue")+1:]
    labels = labels[labels.index("log_pvalue")+1:]
    
    # Add the customized pvalue legend with formatted labels showing -log p-value and actual p-value
    plt.legend(
        handles=handles,
        labels=[f'{round(float(l),1)}' for l in labels], # Format legend labels
        title="-log pvalue",
        loc='upper left',
        bbox_to_anchor=(1, 1)  # Place the legend outside the plot
    )

    # Second part of legend: effect size color bar
    cax = plt.gcf().add_axes([0.95, 0.15, 0.03, 0.4])
    # Create a scalar mappable for the colorbar using the effect size colormap
    sm = plt.cm.ScalarMappable(cmap=effect_cmap, norm=norm_d)
    # Add the colorbar and set its label for the effect size
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label("Effect Size")

def __calculate_hierarchical_clustering(data)->np.ndarray[float]:
    """calculate hierarchical clustering, while supporting for missing values and using optimal_ordering of the clusters.

    Args:
        data: array-like of shape (n_samples_X, n_features). An array where each row is a sample and each column is a feature.

    Returns:
        nd.array: The hierarchical clustering encoded as a linkage matrix of shape (n_samples-1, 4). See scipy.cluster.hierarchy.linkage() for detailed explanations.
    """
    if data.shape[0]==1: # cannoy perform hierarchical clustering if only one sample is in data
        return None
    else:
        # Compute pairwise distances ignoring NaNs between samples, result is of shape (n_samples, n_samples)
        pairwise_dists:np.ndarray = nan_euclidean_distances(data)
        # Convert to a condensed distance matrix for samples, result is of shape ((n_samples*n_sample-1)/2,)
        condensed_dists:np.ndarray = squareform(pairwise_dists, checks=False)
        # Compute hierarchical clustering of samples
        linkage_matrix:np.ndarray = linkage(condensed_dists, method='average', optimal_ordering=True)
        return linkage_matrix