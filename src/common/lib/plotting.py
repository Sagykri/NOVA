import os
import json
import numpy as np
import pandas as pd
import logging
import math
import datetime

from typing import Dict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import ttest_ind
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import nan_euclidean_distances

from src.common.lib.metrics import get_metrics_figure
from src.common.lib.utils import get_if_exists
from src.common.configs.dataset_config import DatasetConfig

def plot_umap0(features:np.ndarray, config_data:DatasetConfig, output_folder_path:str)->None:
    """Plot 2d UMAP of given embeddings, for each marker separately

    Args:
        features (np.ndarray): array containing the umap embeddings and labels
        config_data (DatasetConfig): dataset config
        output_folder_path (str): root path to save the plots and configuration in
    """
    logging.info(f"[plot_umap0]")
    umap_embeddings, labels = features[:,:2], features[:,2]
    markers = np.unique([m.split('_')[0] if '_' in m else m for m in np.unique(labels.reshape(-1,))]) 
    logging.info(f"[plot umap0] Detected markers: {markers}")
    folder = _generate_folder_name(config_data, include_time=True)

    saveroot = os.path.join(output_folder_path, folder)
    if not os.path.exists(saveroot):
        os.makedirs(saveroot, exist_ok=True)
    
    _save_config(config_data, saveroot)
    
    for marker in markers:
        logging.info(f"[plot_umap0]: Marker: {marker}")
        indices = np.where(np.char.startswith(labels.astype(str), f"{marker}_"))[0]
        logging.info(f"[plot_umap0]: {len(indices)} indexes have been selected")

        if len(indices) == 0:
            logging.info(f"[plot_umap0] No data for marker {marker}, skipping.")
            continue

        marker_umap_embeddings, marker_labels = np.copy(umap_embeddings[indices]), np.copy(labels[indices].reshape(-1,))
        
        savepath = os.path.join(saveroot, f'{marker}')

        label_data = _map_labels(marker_labels, config_data)
        show_ari = np.unique(label_data).shape[0] <= 10
                
        _plot_umap_embeddings(marker_umap_embeddings, label_data, config_data, 
                             savepath=savepath, show_ari=show_ari, title=marker)      

def plot_umap1(features:np.ndarray, config_data:DatasetConfig, output_folder_path:str)->None:
    """Plot 2d UMAP of given embeddings, all markers together

    Args:
        features (np.ndarray): array containing the umap embeddings and labels
        config_data (DatasetConfig): dataset config
        output_folder_path (str): root path to save the plot and configuration in
    """
    logging.info(f"[plot_umap1]")
    umap_embeddings, labels = features[:,:2], features[:,2]
    folder = _generate_folder_name(config_data, include_time=True)
   
    saveroot = os.path.join(output_folder_path, folder)
    if not os.path.exists(saveroot):
        os.makedirs(saveroot, exist_ok=True)
        
    _save_config(config_data, saveroot)
    
    label_data = _map_labels(labels, config_data)
    
    ordered_marker_names = get_if_exists(config_data, 'ORDERED_MARKER_NAMES', None)
    if ordered_marker_names is not None:
        ordered_names = [config_data.UMAP_MAPPINGS[marker]['alias'] for marker in ordered_marker_names]
    
    savepath = os.path.join(saveroot,'umap1')
    
    _plot_umap_embeddings(umap_embeddings, label_data, config_data, savepath, 
                        ordered_names = ordered_names, show_ari=False)

def plot_umap2(features:np.ndarray, config_data:DatasetConfig, output_folder_path:str)->None:
    """Plot 2d UMAP of given concatenated embeddings

    Args:
        features (np.ndarray): array containing the umap embeddings and labels
        config_data (DatasetConfig): dataset config
        output_folder_path (str): root path to save the plot and configuration in
    """
    logging.info(f"[plot_umap2]")
    umap_embeddings, labels = features[:,:2], features[:,2]
    folder = _generate_folder_name(config_data, include_time=True)
    
    saveroot = os.path.join(output_folder_path, folder)
    if not os.path.exists(saveroot):
        os.makedirs(saveroot, exist_ok=True)
        
    _save_config(config_data, saveroot)  

    label_data = _map_labels(labels, config_data)
    
    savepath = os.path.join(saveroot, folder) 
    _plot_umap_embeddings(umap_embeddings, label_data, config_data, savepath, show_ari=False)

def _plot_umap_embeddings(umap_embeddings: np.ndarray, 
                         label_data: np.ndarray, 
                         config_data: DatasetConfig, 
                         savepath: str = None,
                         title: str = 'UMAP projection of Embeddings', 
                         outliers_fraction: float = 0.1,
                         dpi: int = 300, 
                         figsize: tuple = (6,5), 
                         ordered_names: list = None, 
                         show_ari: bool = True,
                         unique_groups: np.ndarray = None) -> None:
    """Plots UMAP embeddings with given labels and configurations."""
    
    name_color_dict =  config_data.UMAP_MAPPINGS
    name_key=config_data.UMAP_MAPPINGS_ALIAS_KEY
    color_key=config_data.UMAP_MAPPINGS_COLOR_KEY
    marker_size = config_data.SIZE
    alpha = config_data.ALPHA
    cell_line_cond_high = get_if_exists(config_data, 'CELL_LINE_COND_HIGH', None)

    if unique_groups is None:
        unique_groups = np.unique(label_data)
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2,1,height_ratios=[20,1])

    ax = fig.add_subplot(gs[0])
    for i, group in enumerate(unique_groups):
        logging.info(f'[_plot_umap_embeddings]: adding {group}')
        indices = label_data == group
        indices = indices.reshape(-1,)
        
        if name_color_dict is not None:
            if cell_line_cond_high is not None:
                color = any(cl in group for cl in cell_line_cond_high)
                color_array = np.array([name_color_dict[group][color_key]] * sum(indices) if color else ['gray'] * sum(indices))
            else:
                color_array = np.array([name_color_dict[group][color_key]] * sum(indices))
        else:
            color_array = np.array([plt.get_cmap('tab20')(i)] * sum(indices))
        
        label=group if name_color_dict is None else name_color_dict[group][name_key]
        ax.scatter(
            umap_embeddings[indices, 0],
            umap_embeddings[indices, 1],
            s=marker_size,
            alpha=alpha,
            c=color_array,
            marker = 'o',
            label=label,

        )
        logging.info(f'[_plot_umap_embeddings]: adding label {label}')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('UMAP1') # Nancy for figure 2A - remove axis label - comment this out
    ax.set_ylabel('UMAP2') # Nancy for figure 2A - remove axis label - comment this out
    ax.set_title(title)
    
    ax.set_xticklabels([]) 
    ax.set_yticklabels([]) 
    ax.set_xticks([]) 
    ax.set_yticks([]) 

    _format_UMAP_legend(ax, ordered_names, marker_size)
        
    if show_ari:
        gs_bottom = fig.add_subplot(gs[1])
        ax, _ = get_metrics_figure(umap_embeddings, label_data, ax=gs_bottom, outliers_fraction=outliers_fraction)
    
    fig.tight_layout()
    
    _save_or_show_plot(fig, savepath, dpi) 
    return

def plot_distances_plots(distances:pd.DataFrame, config_data:DatasetConfig, output_folder_path:str)->None:
    """Wrapper function to create the folder of distances plots and plot them

    Args:
        distances (pd.DataFrame): dataframe with calculated distances per marker
        config_data (DatasetConfig): dataset config
        output_folder_path (str): root path to save the plots and configuration in
    """
    folder = _generate_folder_name(config_data, include_time=True)
    saveroot = os.path.join(output_folder_path, folder)
    if not os.path.exists(saveroot):
        os.makedirs(saveroot, exist_ok=True)
    
    _save_config(config_data, saveroot)
    
    baseline = config_data.BASELINE_CELL_LINE_CONDITION

    _plot_marker_ranking(distances, baseline, saveroot, metric='ARI_KMeansConstrained')
    _plot_clustermap(distances, baseline, saveroot, metric='ARI_KMeansConstrained')
    _plot_bubble_plot(distances, baseline, saveroot, metric='ARI_KMeansConstrained')

def _plot_marker_ranking(distances:pd.DataFrame, baseline:str, saveroot:str, metric:str)->None:
    """Generate and save a boxplot of marker distances with p-values, separately for each condition.

    Args:
        distances (pd.DataFrame): Distances between conditions and baseline for each marker
        baseline (str): The name of the 'cell_line_condition' which is the baseline in the calculations
        saveroot (str): Path to the folder where the plot should be saved
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.
    """
    logging.info(f"[plot_marker_ranking]")
    conditions = distances.condition.drop_duplicates().to_list()
    conditions.remove(baseline)
    for cond in conditions:
        marker_pvalue = _calc_pvalue_and_effect(distances, baseline, cond, metric)
        savepath = os.path.join(saveroot, f'{cond}_vs_{baseline}_boxplot') 
        _plot_boxplot(distances, baseline, cond, metric, 
                     marker_pvalue, show_effect_size=True, savepath = savepath)

def _plot_clustermap(distances:pd.DataFrame, baseline:str, saveroot:str, metric:str)->None:
    """Generate and save a clustermap of marker p-values per condition.

    Args:
        distances (pd.DataFrame): Distances between conditions and baseline for each marker
        baseline (str): The name of the 'cell_line_condition' which is the baseline in the calculations
        saveroot (str): Path to the folder where the plot should be saved
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.
    """
    logging.info(f"[_plot_clustermap]")
    # Calculate marker p-values and process the dataframe
    pvalues_df = _calculate_marker_pvalue_per_condition(distances, baseline, metric)  
    pvalues_df = pvalues_df[["marker", "condition", "pvalue","d"]].drop_duplicates().reset_index(drop=True) 
    marker_pvalue_per_condition = pvalues_df.pivot(index='marker', columns='condition', values='pvalue')
    
    # Perform hierarchical clustering
    linkage_row = _calculate_hierarchical_clustering(marker_pvalue_per_condition)
    linkage_col = _calculate_hierarchical_clustering(marker_pvalue_per_condition.T)
    
    # Determine clustering parameters
    row_cluster=marker_pvalue_per_condition.shape[0]>1
    col_cluster=marker_pvalue_per_condition.shape[1]>1
    na_mask = marker_pvalue_per_condition.isnull()
    
    # Generate the clustermap
    g = sns.clustermap(marker_pvalue_per_condition,mask=na_mask, cmap='Blues_r', figsize=(10, 10), row_cluster=row_cluster,
                       col_cluster=col_cluster,
                       row_linkage=linkage_row, col_linkage=linkage_col,
                   yticklabels=True, xticklabels=True, annot=True, vmax=0.05)

    ## Optional: Highlight significant values
    # for i in range(clustered_df.shape[0]):
    #     for j in range(clustered_df.shape[1]):
    #         if clustered_df.iloc[i, j] <= 0.05:
    #             # Add a rectangle around the cell
    #             g.ax_heatmap.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='tomato', lw=2))

    plt.title(f'vs {baseline}')
    
    # save the plot
    savepath = os.path.join(saveroot, f'vs_{baseline}_clustermap') 
    _save_or_show_plot(g, savepath, dpi=100)
    return
      
def _plot_bubble_plot(distances:pd.DataFrame, baseline:str, saveroot:str, metric:str, effect_cmap:str = 'Blues', vmin_d:int =-1, vmax_d:int =10)->None:
    """Generate and save a bubble plot of marker p-values and effect size per condition.

    Args:
        distances (pd.DataFrame): Distances between conditions and baseline for each marker
        baseline (str): The name of the 'cell_line_condition' which is the baseline in the calculations
        saveroot (str): Path to the folder where the plot should be saved
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.
        effect_cmap (str, optional): Colormap for the bubble_plot. Defaults to 'Blues'.
        vmin_d (int, optional): Minimum value of the effect (for normalization). Defaults to -1.
        vmax_d (int, optional): Maximum value of the effect (for normalization). Defaults to 10.
    """

    logging.info(f"[_plot_bubble_plot]")
    # Calculate marker p-values and process the dataframe
    pvalues_df = _calculate_marker_pvalue_per_condition(distances, baseline, metric)  
    pvalues_df = pvalues_df[["marker", "condition", "pvalue","d"]].drop_duplicates().reset_index(drop=True) 
    marker_pvalue_per_condition = pvalues_df.pivot(index='marker', columns='condition', values='pvalue')
    
    # Perform hierarchical clustering
    linkage_row = _calculate_hierarchical_clustering(marker_pvalue_per_condition)
    linkage_col = _calculate_hierarchical_clustering(marker_pvalue_per_condition.T)
    
    marker_order = _get_order_from_linkage(linkage_row, marker_pvalue_per_condition.index)
    condition_order = _get_order_from_linkage(linkage_col, marker_pvalue_per_condition.columns)

    pvalues_df['marker'] = pd.Categorical(pvalues_df['marker'], categories=marker_order, ordered=True)
    pvalues_df['condition'] = pd.Categorical(pvalues_df['condition'], categories=condition_order, ordered=True)

    # Sort the DataFrame based on the hierarchical clustering order
    pvalues_df.sort_values(by=['condition','marker'])
    
    # Normalize effect sizes and adjust p-values for visualization
    norm_d = mcolors.Normalize(vmin=vmin_d, vmax=vmax_d) if vmin_d is not None else None
    pvalues_df['log_pvalue'] = -np.log10(_bin_pvalues(pvalues_df['pvalue']))
    size_norm = mcolors.LogNorm(vmin=pvalues_df['log_pvalue'].min(), vmax=pvalues_df['log_pvalue'].max())

    # Plot
    fig = plt.figure(figsize=(8, 6), dpi=300)
    s = sns.scatterplot(
        data=pvalues_df,
        x="condition",
        y="marker",
        size='log_pvalue',
        size_norm = size_norm,
        hue='d',
        hue_norm = norm_d,
        legend='full',
        palette=effect_cmap,
        sizes=(1, 200),
        edgecolor='black',
        linewidth=0.5,
        )
    plt.xticks(rotation=90)
    plt.title(f'vs {baseline}')

    # Customize legend
    _customize_legend(s, effect_cmap, norm_d)


    savepath = os.path.join(saveroot, f'vs_{baseline}_bubbleplot')
    _save_or_show_plot(fig, savepath, dpi=100)
    return 

def _format_UMAP_legend(ax, ordered_names: list, marker_size: int) -> None:
    """Formats the legend in the plot."""
    handles, labels = ax.get_legend_handles_labels()

    if ordered_names:
        logging.info('Ordering legend labels!')
        handles = [h for l, h in sorted(zip(labels, handles), key=lambda x: ordered_names.index(x[0]))]
        labels = ordered_names

    leg = ax.legend(handles, labels, prop={'size': 6},
                    bbox_to_anchor=(1, 1), loc='upper left',
                    ncol=1 + len(labels) // 25, frameon=False)

    for handle in leg.legendHandles:
        handle.set_alpha(1)
        handle.set_sizes([max(6, marker_size)])

def _save_or_show_plot(fig, savepath: str, dpi: int, save_png:bool=True, save_eps:bool=False) -> None:
    """Saves the plot if a savepath is provided, otherwise shows the plot."""
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        logging.info(f"Saving plot to {savepath}")
        if save_png:
            fig.savefig(f"{savepath}.png", dpi=dpi, bbox_inches='tight')
        elif save_eps:
            fig.savefig(f"{savepath}.eps", dpi=dpi, format='eps')
        else:
            logging.info(f"save_eps and save_png are both False, not saving!")
    else:
        plt.show()

def _save_config(config_data: DatasetConfig, output_folder_path: str) -> None:
    """Saves the configuration data to a JSON file."""
    os.makedirs(output_folder_path, exist_ok=True)
    with open(os.path.join(output_folder_path, 'config.json'), 'w') as json_file:
        json.dump(config_data.__dict__, json_file, indent=4)

def _map_labels(labels: np.ndarray, config_data: DatasetConfig) -> np.ndarray:
    """Maps labels based on the provided function in the configuration."""
    map_function = get_if_exists(config_data, 'MAP_LABELS_FUNCTION', None)
    if map_function:
        map_function = eval(map_function)(config_data)
        return map_function(labels)
    return labels

def _generate_folder_name(config_data: DatasetConfig, include_time=True) -> str:
    """Generate a unique output directory based on the current time and configuration."""
    now = datetime.datetime.now()
    input_folders = '_'.join([os.path.basename(f) for f in config_data.INPUT_FOLDERS])

    reps = '_'.join(config_data.REPS) if config_data.REPS else "both_reps"
    cell_lines = '_'.join(config_data.CELL_LINES) if config_data.CELL_LINES else "all_cell_lines"
    
    if include_time:
        return f"{input_folders}_{reps}_{cell_lines}_{now.strftime('%d%m%y_%H%M%S_%f')}"
    else:
        return f"{input_folders}_{reps}_{cell_lines}"

def _calc_pvalue_and_effect(distances:pd.DataFrame, baseline:str, condition:str, metric:str)->dict:
    """Calculate the significance and the effect size of the difference between the baseline distances and the condition distances, for each marker.

    Args:
        distances (pd.DataFrame): Distances between conditions and baseline for each marker
        baseline (str): The name of the 'cell_line_condition' which is the baseline in the calculations
        condition (str): The name of the 'cell_line_condition' to compare with the baseline
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.

    Returns:
        dict: Dictionary containing for each marker its pvalue and effect size (Cohen's d)
    """
    markers = distances.marker.drop_duplicates()
    marker_pval = {}
    for m in markers:
        cur_distacnes = distances[distances.marker==m]
        if cur_distacnes.shape[0]==0:
            continue
        baseline_distances = cur_distacnes[cur_distacnes.condition == baseline][metric]
        condition_distances = cur_distacnes[cur_distacnes.condition == condition][metric]
        pval = ttest_ind(condition_distances, baseline_distances,alternative = 'greater')[1]

        marker_pval[m]={}
        marker_pval[m]['pvalue'] = pval
        
        # calc effect size
        d = _cohen_d(condition_distances, baseline_distances)
        marker_pval[m]['d']=d
        
    return marker_pval

def _cohen_d(x, y)->float:
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

def _plot_boxplot(distances:pd.DataFrame, baseline:str, condition:str, 
                  metric:str, marker_pval:Dict, show_effect_size:bool=False,
                  savepath:str=None)->None:
    """
    Plot a boxplot to visualize the distribution and significance of distances for a given condition compared to a baseline.
    The markers are displayed in descending order of their distance from baseline. Optionally, effect sizes can be shown.

    Args:
        distances (pd.DataFrame): DataFrame containing the distance metrics
        baseline (str): Name of the baseline condition for comparison.
        condition (str): Name of the condition to compare against the baseline.
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.
        marker_pval (Dict): Nested dictionary where keys are marker names. Each marker is a dictionary, 
                            with keys for p-values and effect size indicating the statistical significance 
                            of the difference between the condition and baseline.
        show_effect_size (bool, optional): If True, effect sizes are displayed on the plot. Defaults to False.
        savepath (str, optional): File path to save the plot. If None, the plot is shown but not saved. Defaults to None.
    """
    cur_distances=distances[distances.condition==condition]
    median_variance = cur_distances.groupby("marker")[metric].agg(['median', 'var'])
    dists_order = median_variance.sort_values(by=['median', 'var'], ascending=[False, False]).index

    cur_distances=distances[distances.condition.isin([baseline,condition])]

    fig = plt.figure(figsize=(10,4))
    b=sns.boxplot(data=cur_distances, order=dists_order, hue='condition',
                x='marker', y=metric, fliersize=0)
    plt.xticks(rotation=90)
    patches = b.patches[:-2]
    labels = []
    for i, marker in enumerate(dists_order):
        height_1 = max(patches[i].get_path().vertices[:, 1])
        height_2 = max(patches[i+len(dists_order)].get_path().vertices[:, 1])
        pval_loc = max(height_1, height_2)

        pval = marker_pval[marker]['pvalue']
        if pval <= 0.05:
            asterisks = _convert_pvalue_to_asterisks(pval)
            # Add the asterisks above the box
            plt.text(dists_order.to_list().index(marker), pval_loc, asterisks, 
                     ha='center', va='bottom', fontsize=10)
        else:     
            plt.text(dists_order.to_list().index(marker), pval_loc + 0.1*pval_loc, round(pval,4), 
                     ha='center', va='bottom', fontsize=7, rotation=90)
    
        labels.append(f'{marker} (d={round(marker_pval[marker]["d"],2)})')
    if show_effect_size:
        b.set_xticklabels(labels)
    
    plt.title(f'{condition} vs {baseline}')
    _save_or_show_plot(fig, savepath, dpi=100)
    return

def _convert_pvalue_to_asterisks(pval:float)->str:
    if pval <= 0.0001:
        asterisks = '****'
    elif pval <= 0.001:
        asterisks = '***'
    elif pval <= 0.01:
        asterisks = '**'
    else:
        asterisks = '*'
    
    return asterisks

def _calculate_marker_pvalue_per_condition(distances:pd.DataFrame, baseline:str, metric:str)->pd.DataFrame:
    """
    Calculate the statistical significance and effect size for the difference in distances 
    between a baseline condition and all other conditions, across all markers.

    Args:
        distances (pd.DataFrame): A DataFrame containing distance measurements for various markers across conditions.
        baseline (str): The name of the baseline condition against which comparisons are made.
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.

    Returns:
        pd.DataFrame: A DataFrame with calculated p-values and effect sizes as columns,
                      indicating the significance of differences between the baseline and each condition.
    """
    conditions = distances.condition.drop_duplicates().to_list()
    conditions.remove(baseline)
    marker_pvalue_per_condition = None
    for cond in conditions:
        marker_pval = _calc_pvalue_and_effect(distances, baseline=baseline, condition=cond, metric=metric)        
        cur_df = pd.DataFrame(marker_pval).T.reset_index(names='marker')
        cur_df['condition'] = cond
        if marker_pvalue_per_condition is None:
            marker_pvalue_per_condition = cur_df
        else:
            marker_pvalue_per_condition = pd.concat([marker_pvalue_per_condition, cur_df], ignore_index=True)
    return marker_pvalue_per_condition

def _bin_pvalues(pvalues):
    """Adjust p-values for better visualization."""
    adjusted_pvalues = np.where(pvalues > 0.06, 0.06, pvalues)
    adjusted_pvalues = np.where((0.05 <= adjusted_pvalues) & (adjusted_pvalues < 0.06), 0.05, adjusted_pvalues)
    adjusted_pvalues = np.where((0.01 <= adjusted_pvalues) & (adjusted_pvalues < 0.05), 0.01, adjusted_pvalues)
    adjusted_pvalues = np.where((0.0001 <= adjusted_pvalues) & (adjusted_pvalues < 0.01), 0.0001, adjusted_pvalues)
    return np.where(adjusted_pvalues < 0.0001, 10**math.floor(np.log10(adjusted_pvalues.min())), adjusted_pvalues)

def _get_order_from_linkage(linkage, items):
    """Get the order of items based on hierarchical clustering."""
    if linkage is not None:
        dendro = dendrogram(linkage, no_plot=True)
        return items[dendro['leaves']]
    return items

def _customize_legend(s, effect_cmap, norm_d):
    """Customize the legend in the bubble plot."""
    handles, labels = s.get_legend_handles_labels()
    s.legend_.remove()
    handles = handles[labels.index("log_pvalue")+1:]
    labels = labels[labels.index("log_pvalue")+1:]
    
    labels_dict = {}
    for l in labels:
        p_value = 10**(-float(l))
        labels_dict[l] = f'{p_value:.0e}' if p_value < 0.001 else f'{p_value:.2f}'
    plt.legend(
        handles=handles,
        labels=[f'{round(float(l),1)} ({labels_dict[l]})' for l in labels],
        title="-log pvalue",
        loc='upper left',
        bbox_to_anchor=(1, 1)
    )
    sm = plt.cm.ScalarMappable(cmap=effect_cmap, norm=norm_d)
    sm.set_array([])
    cax = plt.gcf().add_axes([0.95, 0.15, 0.03, 0.4])
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.ax.tick_params(length=0)
    cbar.set_label("Effect Size (Cohen's d)")

def _calculate_hierarchical_clustering(data):
    if data.shape[0]>1:
        # Compute pairwise distances ignoring NaNs for rows
        pairwise_dists = nan_euclidean_distances(data)
        # Convert to a condensed distance matrix for rows
        condensed_dists = squareform(pairwise_dists, checks=False)
        # Compute linkage for rows
        linkage = linkage(condensed_dists, method='average', optimal_ordering=True)
    else:
        linkage = None
    
    return linkage