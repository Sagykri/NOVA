import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.dataset_config import DatasetConfig
from src.common.lib.utils import save_plot

import numpy as np
import pandas as pd
import logging
import math

from typing import List
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import ttest_ind
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import nan_euclidean_distances

def plot_distances_plots(distances:pd.DataFrame, config_data:DatasetConfig, saveroot:str,
                         metric:str='ARI_KMeansConstrained')->None:
    """Wrapper function to create the folder of distances plots and plot them

    Args:
        distances (pd.DataFrame): dataframe with calculated distances per marker
        config_data (DatasetConfig): dataset config
        output_folder_path (str): root path to save the plots and configuration in
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.
    """
    if saveroot:
        os.makedirs(saveroot, exist_ok=True)
        config_data._save_config(saveroot)
    else:
        saveroot = None

    plot_marker_ranking(distances, saveroot, config_data, metric=metric, show_effect_size=True)
    plot_clustermap(distances, saveroot, config_data, metric=metric)
    plot_bubble_plot(distances, saveroot, config_data, metric=metric)

def plot_marker_ranking(distances:pd.DataFrame, saveroot:str, config_data:DatasetConfig,
                        metric:str, show_effect_size:bool=False)->None:
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
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.
        show_effect_size (bool, optional): If True, effect sizes are displayed on the plot. Defaults to False.
    """
    logging.info(f"[plot_marker_ranking]")
    baseline:str = config_data.BASELINE_CELL_LINE_CONDITION

    conditions = distances.condition.drop_duplicates().to_list()
    conditions.remove(baseline)
    pvalues_df = __calculate_marker_pvalue_per_condition(distances, baseline, metric)
    for cond in conditions:
        savepath = None
        if saveroot:
            savepath = os.path.join(saveroot, f'{cond}_vs_{baseline}_boxplot') 
        __plot_boxplot(distances, baseline, cond, metric, 
                     pvalues_df, config_data, show_effect_size=show_effect_size, savepath = savepath)
        

def plot_clustermap(distances:pd.DataFrame, saveroot:str, config_data:DatasetConfig, metric:str)->None:
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
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.
    """
    logging.info(f"[_plot_clustermap]")
    baseline:str = config_data.BASELINE_CELL_LINE_CONDITION
    # Calculate marker p-values and process the dataframe
    pvalues_df = __calculate_marker_pvalue_per_condition(distances, baseline, metric)  
    # we want our data organzied as marker in the rows and condition in the columns (values are the pvalues)
    marker_pvalue_per_condition = pvalues_df.pivot(index='marker', columns='condition', values='pvalue')
    
    # Perform hierarchical clustering
    marker_linkage = __calculate_hierarchical_clustering(marker_pvalue_per_condition)
    condition_linkage = __calculate_hierarchical_clustering(marker_pvalue_per_condition.T)
    
    # Determine clustering parameters
    row_cluster=marker_pvalue_per_condition.shape[0]>1
    col_cluster=marker_pvalue_per_condition.shape[1]>1
    na_mask = marker_pvalue_per_condition.isnull()
    
    # Generate the clustermap
    clustermap = sns.clustermap(marker_pvalue_per_condition,mask=na_mask, cmap='Blues_r', figsize=(10, 10), row_cluster=row_cluster,
                       col_cluster=col_cluster,
                       row_linkage=marker_linkage, col_linkage=condition_linkage,
                   yticklabels=True, xticklabels=True, annot=True, vmax=0.05)

    ## Optional: Highlight significant values
    # for i in range(clustered_df.shape[0]):
    #     for j in range(clustered_df.shape[1]):
    #         if clustered_df.iloc[i, j] <= 0.05:
    #             # Add a rectangle around the cell
    #             g.ax_heatmap.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='tomato', lw=2))
    baseline = __convert_labels(clustermap, config_data)

    plt.title(f'vs {baseline}')
    
    # save the plot
    savepath = None
    if saveroot:
        savepath = os.path.join(saveroot, f'vs_{baseline}_clustermap') 
    if savepath:
        save_plot(clustermap, savepath, dpi=100)
    else:
        plt.show()
    return
      
def plot_bubble_plot(distances:pd.DataFrame, saveroot:str, config_data:DatasetConfig, 
                     metric:str, effect_cmap:str = 'Blues', vmin_d:int =-1, vmax_d:int =10)->None:
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
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.
        effect_cmap (str, optional): Colormap for the bubble_plot. Defaults to 'Blues'.
        vmin_d (int, optional): Minimum value of the effect (for normalization). Defaults to -1.
        vmax_d (int, optional): Maximum value of the effect (for normalization). Defaults to 10.
    """

    logging.info(f"[_plot_bubble_plot]")
    baseline:str = config_data.BASELINE_CELL_LINE_CONDITION
    
    # Calculate marker p-values and process the dataframe
    pvalues_df = __calculate_marker_pvalue_per_condition(distances, baseline, metric)  
    __sort_markers_and_conditions_by_linkage(pvalues_df)

    # Normalize effect sizes and adjust p-values for visualization
    norm_d = mcolors.Normalize(vmin=vmin_d, vmax=vmax_d) if vmin_d is not None else None
    pvalues_df['log_pvalue'] = -np.log10(__bin_pvalues(pvalues_df['pvalue']))

    # Plot
    fig = plt.figure(figsize=(8, 6), dpi=300)
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
    
    baseline = __convert_labels(scatter, config_data)
    plt.xticks(rotation=90)
    plt.title(f'vs {baseline}')

    __customize_bubbleplot_legend(scatter, effect_cmap, norm_d)

    savepath = None
    if saveroot:
        savepath = os.path.join(saveroot, f'vs_{baseline}_bubbleplot')
    if savepath:
        save_plot(fig, savepath, dpi=100)
    else:
        plt.show()
    return 

def __sort_markers_and_conditions_by_linkage(pvalues_df:pd.DataFrame)->None:
    # we want our data organzied as marker in the rows and condition in the columns (values are the pvalues)
    marker_pvalue_per_condition = pvalues_df.pivot(index='marker', columns='condition', values='pvalue')
    
    # Perform hierarchical clustering
    marker_linkage = __calculate_hierarchical_clustering(marker_pvalue_per_condition)
    condition_linkage = __calculate_hierarchical_clustering(marker_pvalue_per_condition.T)
    marker_order = __get_order_from_linkage(marker_linkage, np.unique(pvalues_df.marker))
    condition_order = __get_order_from_linkage(condition_linkage, np.unique(pvalues_df.condition))

    pvalues_df['marker'] = pd.Categorical(pvalues_df['marker'], categories=marker_order, ordered=True)
    pvalues_df['condition'] = pd.Categorical(pvalues_df['condition'], categories=condition_order, ordered=True)

    # Sort the DataFrame based on the hierarchical clustering order
    pvalues_df.sort_values(by=['condition','marker'])
    return None

def __convert_labels(plot, config_data:DatasetConfig)->str:
    """
    Given a plot object, edit the marker and condition labels using the config_data.

    Returns:
        str: The edited baseline string.
    """
    name_color_dict = config_data.UMAP_MAPPINGS
    baseline = config_data.BASELINE_CELL_LINE_CONDITION
    if not name_color_dict:
        return baseline
    name_key = config_data.UMAP_MAPPINGS_ALIAS_KEY

    if isinstance(plot, sns.matrix.ClusterGrid): # when plot is a clustermap it's a clustergrid so we need to extract the heatmap ax.
        plot = plot.ax_heatmap

    marker_labels = plot.get_yticklabels()
    marker_labels = [name_color_dict[label.get_text()][name_key] for label in marker_labels]
    ytick_positions = plot.get_yticks()
    plot.set_yticks(ytick_positions)
    plot.set_yticklabels(marker_labels)
    
    condition_labels = plot.get_xticklabels()
    condition_labels = [config_data.UMAP_MAPPINGS_CONDITION_AND_ALS[label.get_text()][name_key] for label in condition_labels]
    xtick_positions = plot.get_xticks()
    plot.set_xticks(xtick_positions)
    plot.set_xticklabels(condition_labels)
    
    baseline = config_data.UMAP_MAPPINGS_CONDITION_AND_ALS[baseline][name_key]
    return baseline

def __calc_pvalue_and_effect(distances:pd.DataFrame, baseline:str, condition:str, metric:str)->dict:
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
        dict: Dictionary containing for each marker its pvalue and effect size (Cohen's d)
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
        d = __calc_cohens_d(condition_distances, baseline_distances)
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

def __plot_boxplot(distances:pd.DataFrame, baseline:str, condition:str, 
                  metric:str, pvalues_df:pd.DataFrame, config_data:DatasetConfig, show_effect_size:bool=False,
                  savepath:str=None)->None:
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
        marker_pval (Dict): Nested dictionary where keys are marker names. Each marker is a dictionary, 
                            with keys for p-values and effect size indicating the statistical significance 
                            of the difference between the condition and baseline.
        show_effect_size (bool, optional): If True, effect sizes are displayed on the plot. Defaults to False.
        savepath (str, optional): File path to save the plot. If None, the plot is shown but not saved. Defaults to None.
    """
    # Filter and sort the data
    cur_distances=distances[distances.condition==condition] # for sorting we want only the condition distances
    median_variance = cur_distances.groupby("marker")[metric].agg(['median', 'var']) # ordering by median, then variance.
    dists_order = median_variance.sort_values(by=['median', 'var'], ascending=[False, False]).index # do the ordering
    cur_distances=distances[distances.condition.isin([baseline,condition])] # after sorting, we can include also the baseline distances

    # Plotting
    name_color_dict = config_data.UMAP_MAPPINGS
    name_key = config_data.UMAP_MAPPINGS_ALIAS_KEY
    fig = plt.figure(figsize=(10,4))
    boxplot=sns.boxplot(data=cur_distances, order=dists_order, hue='condition',
                x='marker', y=metric, fliersize=0)
    
    patches = boxplot.patches[:-2]
    labels = []
    # Add pavlues and effect size
    for i, marker in enumerate(dists_order):
        marker_pvalue = pvalues_df[(pvalues_df.condition==condition)&(pvalues_df.marker==marker)].pvalue.values[0]
        __add_pvalue(marker, i, dists_order, patches,marker_pvalue)
        effect_size_formatted = round(pvalues_df[(pvalues_df.condition==condition)&(pvalues_df.marker==marker)].d.values[0],2)
        label = name_color_dict[marker][name_key] if name_color_dict else marker
        if show_effect_size:
            label = f'{label} (d={effect_size_formatted})'
        labels.append(label)

    boxplot.set_xticklabels(labels)
    
    plt.xticks(rotation=90)
    plt.title(f'{condition} vs {baseline}')
    if savepath:
        save_plot(fig, savepath, dpi=100)
    else:
        plt.show()
    return

def __add_pvalue(marker:str, marker_index:int, dists_order:List[str], patches, pvalue:float)->None:
    # find the highest bar between the baseline and condition for the given marker
    height_1 = max(patches[marker_index].get_path().vertices[:, 1])
    height_2 = max(patches[marker_index+len(dists_order)].get_path().vertices[:, 1])
    pval_loc = max(height_1, height_2)

    if pvalue <= 0.05: # plot significant asterix
        asterisks = __convert_pvalue_to_asterisks(pvalue)
        # Add the asterisks above the box
        plt.text(dists_order.to_list().index(marker), pval_loc, asterisks, 
                    ha='center', va='bottom', fontsize=10)
    else: # plot the full pvalue
        plt.text(dists_order.to_list().index(marker), pval_loc + 0.1*pval_loc, round(pvalue,4), 
                    ha='center', va='bottom', fontsize=7, rotation=90)
            
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

def __calculate_marker_pvalue_per_condition(distances:pd.DataFrame, baseline:str, metric:str)->pd.DataFrame:
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
        metric (str): The metric used to evaluate the distances, e.g., 'ARI_KMeansConstrained', 'dist', etc.

    Returns:
        pd.DataFrame: A DataFrame with calculated p-values and effect sizes as columns,
                      indicating the significance of differences between the baseline and each condition.
    """
    conditions = distances.condition.drop_duplicates().to_list()
    conditions.remove(baseline)
    marker_pvalue_per_condition = None
    for cond in conditions:
        marker_pval = __calc_pvalue_and_effect(distances, baseline=baseline, condition=cond, metric=metric)        
        cur_df = pd.DataFrame(marker_pval).T.reset_index(names='marker')
        cur_df['condition'] = cond
        if marker_pvalue_per_condition is None:
            marker_pvalue_per_condition = cur_df
        else:
            marker_pvalue_per_condition = pd.concat([marker_pvalue_per_condition, cur_df], ignore_index=True)
    return marker_pvalue_per_condition

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
    
    # Create a dictionary to format p-value labels (can remove if don't want to show the original p-values, but only the -log)
    labels_dict = {}
    for l in labels:
        p_value = 10**(-float(l)) # Convert the -log10 p-value back to the original p-value
        # Format the p-value as scientific notation if very small, otherwise as a decimal
        labels_dict[l] = f'{p_value:.0e}' if p_value < 0.001 else f'{p_value:.2f}'
    
    # Add the customized pvalue legend with formatted labels showing -log p-value and actual p-value
    plt.legend(
        handles=handles,
        labels=[f'{round(float(l),1)} ({labels_dict[l]})' for l in labels], # Format legend labels
        title="-log pvalue",
        loc='upper left',
        bbox_to_anchor=(1, 1)  # Place the legend outside the plot
    )

    # Second part of legend: effect size color bar
    cax = plt.gcf().add_axes([0.95, 0.15, 0.03, 0.4])
    # Create a scalar mappable for the colorbar using the effect size colormap
    sm = plt.cm.ScalarMappable(cmap=effect_cmap, norm=norm_d)
    # Add the colorbar and set its label for the effect size (Cohen's d)
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label("Effect Size (Cohen's d)")

def __calculate_hierarchical_clustering(data)->np.ndarray[float]:
    """calculate hierarchical clustering, while supporting for missing values and using optimal_ordering of the clusters.

    Args:
        data: array-like of shape (n_samples_X, n_features). An array where each row is a sample and each column is a feature.

    Returns:
        nd.array: The hierarchical clustering encoded as a linkage matrix of shape (n_samples-1, 4). See scipy.cluster.hierarchy.linkage() for detailed explanations.
    """
    if data.shape[0]==1: # cannoy perform hierarchical clustering if only one sample is in data
        linkage_matrix = None
    else:
        # Compute pairwise distances ignoring NaNs between samples, result is of shape (n_samples, n_samples)
        pairwise_dists:np.ndarray = nan_euclidean_distances(data)
        # Convert to a condensed distance matrix for samples, result is of shape ((n_samples*n_sample-1)/2,)
        condensed_dists:np.ndarray = squareform(pairwise_dists, checks=False)
        # Compute hierarchical clustering of samples
        linkage_matrix:np.ndarray = linkage(condensed_dists, method='average', optimal_ordering=True)
    return linkage_matrix