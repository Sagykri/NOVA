import os 
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from src.common.lib.image_metrics import improve_brightness
from src.common.configs.base_config import BaseConfig


clustermap_colors = LinearSegmentedColormap.from_list("", ["tan", "white", "slategray"])

def load_multiple_vqindhists(batches, embeddings_folder, datasets=['trainset','valset','testset'], embeddings_layer='vqindhist1'):
    
    """Load vqinhist1, labels and paths of tiles in given batches
    Args:        
        batches (list of strings): list of batch folder names (e.g., ['batch6', 'batch7'])
        embeddings_folder (string): full path to stored embeddings
        datasets (list, optional): Defaults to ['trainset','valset','testset'].
        embeddings_layer (str, optional): Defaults to 'vqindhist1'.
    Returns:
        vqindhist: list of np.arrays from shape (# cell lines). each np.array is in shape (# tiles, 2048)
        labels: list of np.arrays from shape (# cell lines). each np.array is in shape (# tiles) and the stored value is full label
        paths: list of strings from shape (# cell lines). each np.array is in shape (# tiles) and the stored value is full path
    """
    
    vqindhist, labels, paths = [] , [], []
    for batch in batches:
        for dataset_type in datasets:
            cur_vqindhist, cur_labels, cur_paths =  np.load(os.path.join(embeddings_folder, batch, f"{embeddings_layer}_{dataset_type}.npy")),\
                                                    np.load(os.path.join(embeddings_folder, batch, f"{embeddings_layer}_labels_{dataset_type}.npy")),\
                                                    np.load(os.path.join(embeddings_folder, batch, f"{embeddings_layer}_paths_{dataset_type}.npy"))
            cur_vqindhist = cur_vqindhist.reshape(cur_vqindhist.shape[0], -1)
            vqindhist.append(cur_vqindhist)
            labels.append(cur_labels)
            paths.append(cur_paths)   
            
    return vqindhist, labels, paths

def create_vqindhists_df(vqindhist, labels, paths, arange_labels=True):
    """Create one DataFrame where columns are 2,048 vqind values, the label and path of each tile 

    Args:
        vqindhist: list of np.arrays from shape (# cell lines). each np.array is in shape (# tiles, 2048)
        labels: list of np.arrays from shape (# cell lines). each np.array is in shape (# tiles) and the stored value is full label
        paths: list of strings from shape (# cell lines). each np.array is in shape (# tiles) and the stored value is full path
        arange_labels (bool, optional): Defaults to True.

    Returns:
        pd.DataFrame: shape is (# tiles, 2050). 
                      rows are tiles, columns are 2048 vqind values and another column for label (string) and columns for path (string)
    """
    vqindhist = np.concatenate(vqindhist)
    labels = np.concatenate(labels)
    paths = np.concatenate(paths)
    
    # Convert to DataFrame
    hist_df = pd.DataFrame(vqindhist)
    # Add the path of each tile
    hist_df['path'] = paths
    # Add label of each tile
    hist_df['label'] = labels
    hist_df['label'] = hist_df['label'].str.replace("_16bit_no_downsample", "")
    hist_df['label'] = hist_df['label'].str.replace("_spd_format", "")
    hist_df['label'] = hist_df['label'].str.replace("_add_brenner_cellpose", "")
    hist_df['label'] = hist_df['label'].str.replace(os.sep, "_")

    def rearrange_string(s):
        parts = s.split('_')
        return f"{parts[4]}_{parts[1]}_{parts[2]}_{parts[0]}_{parts[3]}"

    if arange_labels:
        hist_df['label'] = hist_df['label'].apply(rearrange_string)
    
    return hist_df

def _get_cluster_extend(den):
    """workaround: to support more than 10 unique clusters

    Args:
        den (scipy.cluster.hierarchy.dendrogram): _description_

    Returns:
        defaultdict(list): _description_
    """
    index_to_cluster = defaultdict(list)
    cur_cluster = 1
    last_color = den['leaves_color_list'][0]
    max_color = np.max([int(x.replace("C","")) for x in den['leaves_color_list']])
    
    # Find indices of 'C0' in the list
    indices_to_replace = [i for i, x in enumerate(den['leaves_color_list']) if x == 'C0']
    # Replace 'C0' with unique names
    for index, replacement in zip(indices_to_replace, range(max_color, max_color + len(indices_to_replace))):
        den['leaves_color_list'][index] = f'C{replacement}'
    
    for index, color in zip(den['ivl'], den['leaves_color_list']):
        if color != last_color:
            cur_cluster += 1
            last_color = color
        index_to_cluster[index] = cur_cluster    
    return index_to_cluster
    
def set_num_clusters_by_dendrogram(clustermap, corr, cutoff=14.2):
    """Cut the dendrogram in cutoff to get clusters and clusters's member (codebook vector indices) 
        
    Args:
        clustermap (seaborn.matrix.ClusterGrid): clustermap object (heatmap)
        corr (DataFrame): codebook vectors correlation matrix 
        cutoff (float, optional): value in y-axis of the dendogram. affects num of clusters obtained. Defaults to 14.2.

    Returns:
        DataFrame: corr with number of cluster for each codebook vectors
    """
    # Plot dendogram
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,4))
    den = scipy.cluster.hierarchy.dendrogram(clustermap.dendrogram_col.linkage,
                                             labels = corr.index,
                                             color_threshold=cutoff,
                                             no_labels=True,
                                             ax=axs[0],
                                             distance_sort = False # dist_sort & count_sort affect the sorting of the clusters!
                                             ) 
    
    # Vertical line in "cutoff"         
    axs[0].axhline(cutoff, c='black', linestyle="-")
    
    # Setting cluster number for each codebook vector (i.e., index)
    index_to_cluster = _get_cluster_extend(den)
    new_cluster_id = []
    for i in corr.index:
        new_cluster_id.append(index_to_cluster[i])
    corr["cluster"] = new_cluster_id
    # To show clusters ordered by number
    corr = corr.sort_values(by='cluster')
    corr['cluster'] = corr['cluster'].astype(str)
    corr['cluster'] = 'C' + corr['cluster']
    
    # Histogram of cluster counts: number of codebook vectors in each cluster
    sns.countplot(data=corr, x='cluster', palette='coolwarm', ax=axs[1])
    # Add labels and title
    axs[1].set_xlabel('Cluster', fontsize=24)
    axs[1].set_ylabel('Count', fontsize=24)
    axs[1].set_title('Number of Codebook Vectors per Cluster', fontsize=24)
    plt.tight_layout()
    plt.show()
    
    return corr

def _add_condition_to_label(label, condition):
    l = label.split("_")
    l.insert(2, condition)
    return '_'.join(l)
    
def calc_correlation_codebook_vectors(df, corr_method):
    """
    For every label in df, calc a representative (mean) vqindhist (AKA, group by label and mean).
    Compute correlation on rows and return the correlation of the codebook vectors

    Args:
        df (_type_): _description_
        corr_method (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Average the histograms per each label and save in a new dataframe (mean_spectra_per_marker)
    mean_spectra_per_marker = df.groupby('label').mean()
    print(f"Computing {corr_method} correlation of {mean_spectra_per_marker.shape[0]} labels based on {mean_spectra_per_marker.shape[1]} codebook vectors")
    
    ## Correlate the indices histograms    
    corr = mean_spectra_per_marker.corr(method=corr_method)
    
    # Remove columns or rows that are all nan (which can happen when the data is constant before the correlation)
    corr.dropna(axis=0, how='all', inplace=True)
    corr.dropna(axis=1, how='all', inplace=True)
    return corr

def create_codebook_heatmap(hist_df, save_path=None, to_save=False, filename=None, corr_method='pearson', calc_linkage=False, linkage_method='average'):
    """Compute correlatino between codebook vectors, plot heatmap (clustermap)

    Args:
        hist_df (pd.DataFrame): vqindhist of all tiles
        save_path (string, optional): Defaults to None.
        to_save (bool, optional): Defaults to False.
        filename (string, optional): Defaults to None.
        corr_method (str, optional): Function for calculating correlation. e.g., pearson, spearman, etc.. Defaults to 'pearson'.
        calc_linkage (bool, optional): to control parsms of scipy.cluster.hierarchy.linkage. Defaults to False.
        linkage_method (str, optional): used only if "calc_linkage" is True. Options: 'single', 'complete', 'average', 'weighted', 'centroid', 'median' or 'ward'. Defaults to 'average'.

    Returns:
        corr (DataFrame): the correlation between codebook vectors (2048, 2048)
        clustermap (seaborn.matrix.ClusterGrid): clustermap object
    """
    corr = calc_correlation_codebook_vectors(hist_df, corr_method)
    
    # Calculate linkage matrix
    if calc_linkage:
        linkage = scipy.cluster.hierarchy.linkage(corr, 
                                                  method=linkage_method, # 'complete', 'average', 'median', 'ward'
                                                  metric='euclidean', 
                                                  optimal_ordering=True)
    else:
        # if sets to None, then the linkage is calculated with defualt values of sns.clustermap()
        linkage=None
    
    # Plot correlation heatmap
    kws = dict(cbar_kws=dict(ticks=[-1,0,1]))
    clustermap = sns.clustermap(corr, 
                                center=0, 
                                cmap=clustermap_colors, 
                                vmin=-1, vmax=1, 
                                row_linkage=linkage, 
                                col_linkage=linkage,
                                figsize=(5,5), 
                                xticklabels=False, 
                                col_cluster=True, 
                                row_cluster=True, 
                                **kws)
    
    # Remove the dendrogram on the rows
    clustermap.ax_row_dendrogram.set_visible(False)
    clustermap.ax_cbar.set_position([clustermap.ax_col_dendrogram.get_position().x1+0.01, # x location 
                                     clustermap.ax_col_dendrogram.get_position().y0+0.02, # y location
                                     0.01,                                                # width
                                     clustermap.ax_col_dendrogram.get_position().height-0.04]) #height
    clustermap.ax_cbar.set_title(f'{corr_method} correlation', fontsize=8)
    clustermap.cax.tick_params(axis='y', labelsize=8, length=0, pad=0.1) 
    
    if to_save:
        clustermap.figure.savefig(os.path.join(save_path, filename), bbox_inches='tight', dpi=300)
    
    return clustermap, corr

def _get_cluster_score_per_tile(cluster_assignment, hist_df, norm_by='cluster_size'):
    """
    Return score_per_cluster; for every tile image, a score for every cluster

    Args:
        cluster_assignment (ps.DataFrame): assignment of codebook vectors stored in "cluster" column
        hist_df (pd.DataFrame): tiles vqinhists, with label and path
        norm_by (string, optional): either 'cluster_size' or 'vqind_size'. Defaults to 'cluster_size'.

    Raises:
        ValueError: if norm_by is not 'cluster_size' or 'vqind_size'/

    Returns:
        score_per_cluster (pd.DataFrame): shape is (# tiles, # clusters + 2)
    """
        
    # get unique cluster names
    clusters = np.unique(cluster_assignment.cluster)
    
    # create DataFrame (rows=tiles, columns=[C1, C2,..., label, path])
    score_per_cluster = pd.DataFrame(index=hist_df.index, columns = list(clusters) + ['label','path'])
    score_per_cluster.label = hist_df.label
    score_per_cluster.path = hist_df.path
    
    for cluster_label, cluster_members in cluster_assignment.groupby('cluster'):
        # for each cluster, get the indices (codebook vectors) assigned to it
        cluster_members = hist_df[cluster_members.index]
        # calc the sum of the count values (# times a codebook vector used in a tile)
        score_per_cluster[cluster_label] = cluster_members.sum(axis=1) 
        
        if norm_by=='cluster_size':
            # normalize by the cluster size
            score_per_cluster[cluster_label] = score_per_cluster[cluster_label] / (cluster_members.index.size)  
        elif norm_by=='vqind_size':
            # normalize by the constant 625 (vq1 uses 25x25 vqinds)
            score_per_cluster[cluster_label] = score_per_cluster[cluster_label] / 625
        else:
            raise ValueError(f'{norm_by} is not supperted')

    return score_per_cluster

def find_max_cluster_per_tile(codebook_vec_cluster_assignment, hist_df, norm_by='cluster_size'):
    """Retuns max cluster for each tile (AKA assigns tile images to clusters)

    Args:
        codebook_vec_cluster_assignment (DataFrame): rows = tiles, columns = cluster number
        hist_df (pd.DataFrame): tiles vqinhists, with label and path

    Returns:
        DataFrame: the max cluster for each tile
    """
    
    # Compute score for every tile and every cluster    
    hist_per_cluster = _get_cluster_score_per_tile(cluster_assignment=codebook_vec_cluster_assignment,
                                                    hist_df=hist_df,
                                                    norm_by=norm_by)
    
    # Find the two largest values and corresponding columns (clusters) for each row
    top_clusters = hist_per_cluster.drop(['label', 'path'], axis=1).apply(lambda row: row.nlargest(2).index, axis=1)

    # Assign the first and second max clusters to new columns
    hist_per_cluster['max_cluster'] = top_clusters.apply(lambda x: x[0])
    
    hist_per_cluster.max_cluster = hist_per_cluster.max_cluster.str.replace('C',"").astype(int)
    
    
    return hist_per_cluster

def find_representative_tiles(cluster_id, tile_score_per_cluster, top_images=8, by_conditions=[''], show_other_labels = True):
    """Returns the representative tile (the path to the tile) for a given cluster_id.
        
    Args:
        cluster_id (int): the ID of the cluster of codebook vectors
        tile_score_per_cluster (pd.DataFrame): max cluster for each tile 
        top_images (int, optional): The number of images to show (per cluster and/or condition). Defaults to 8.
        by_conditions (list, optional): if to show representative tiles splitted by condition. Defaults to [''].

    Returns:
        list_of_rep_tiles_path (list of lists): we have a list for every condition. each element in the list is the the full path of the tile. 
                                                length of the list is len("by_conditions"), each containes len("top_images") 
                                          
        
    """
    # Get tiles that were assigned with this cluster as their max cluster 
    max_cluster_group = tile_score_per_cluster[tile_score_per_cluster.max_cluster==cluster_id]
    n_conditions = len(by_conditions)
    list_of_rep_tiles_path = []
    
    if show_other_labels:
        # if we want to see tiles also from other (=not max) labels
        top_images = int(top_images/2)
    for cond in by_conditions:
        
        rep_tiles_path = []
        
        
        # Get all tiles with this condition
        max_cluster_group_cond = max_cluster_group[max_cluster_group.label.str.contains(cond)]
        # Get the top N tiles (where N is defined by "top_images")
        top_rep_tiles = max_cluster_group_cond[[f"C{cluster_id}", 'path']].sort_values(by=f"C{cluster_id}", ascending=False)[:top_images]
        # Get the path of the top representative tiles
        rep_tiles_path.extend(top_rep_tiles.path)
        
        if show_other_labels:
            # Show from the next most common labels
            most_common_marker = top_rep_tiles.path.str.split(os.sep).str[-2].mode().values[0]
            max_cluster_group_cond_others = max_cluster_group_cond[~max_cluster_group_cond.label.str.contains(most_common_marker)]
            top_others = max_cluster_group_cond_others[[f"C{cluster_id}", 'path']].sort_values(by=f"C{cluster_id}", ascending=False)[:top_images]
            rep_tiles_path.extend(top_others.path)
        
        # Alert if no representative tiles were found for this cluster_id 
        if (len(rep_tiles_path) == 0):
            if n_conditions>1:
                print(f'Found no representative tiles for {cluster_id=} and {cond=}!')
            else:
                print(f'Found no representative tiles for {cluster_id=}!')
        
        list_of_rep_tiles_path.append(rep_tiles_path)
    
    return list_of_rep_tiles_path

def _split_tile_path(tile_path):
    cut = tile_path.rfind("_")
    # Cut the path in the last "_"
    real_path = tile_path[:cut]
    tile_number = int(tile_path[cut+1:])
    return real_path, tile_number

def save_representative_tiles(rep_tiles_per_cluster, chosen_idx_dict, save_path=None, to_save=False):
    for cluster in chosen_idx_dict:
        chosen_idx = chosen_idx_dict[cluster]
        for cond, cond_list in enumerate(chosen_idx):
            paths = [rep_tiles_per_cluster[cluster][cond][i] for i in cond_list]
            for path in paths:
                real_path, tile_number = _split_tile_path(path)
                
                fig, ax = plt.subplots()
                # Load the tile (numpy)
                cur_site = np.load(real_path)
                
                # Adjust contrast and brightness
                tile = improve_brightness(img=cur_site[tile_number,:,:,0], 
                                        contrast_factor=1.5, 
                                        brightness_factor=0)
                
                ax.imshow(tile, cmap='cet_linear_ternary_red_0_50_c52',vmin=0,vmax=1) # 
                ax.axis('off')

                # if to_save:
                    #fig.savefig()
                
def plot_representative_tiles(tile_score_per_cluster, top_images=8, by_conditions=[''], show_other_labels=True,
                              figsize=(20,6)):
    
    print(f'Showing more than one label = {show_other_labels}')
    n_conditions = len(by_conditions)
    rep_tiles_per_cluster = {}
    for i, cluster_id in enumerate(np.unique(tile_score_per_cluster[['max_cluster']])):
        
        list_rep_tiles_path = find_representative_tiles(cluster_id, 
                                                        tile_score_per_cluster, 
                                                        top_images, 
                                                        by_conditions, 
                                                        show_other_labels=show_other_labels)
        rep_tiles_per_cluster[cluster_id] = list_rep_tiles_path
        fig, axs = plt.subplots(ncols=top_images, nrows=len(by_conditions),  figsize=figsize)
        
        # If single condition, then we need to make axs 2D manually. Nancy's trick :) 
        if n_conditions<2:
            axs = axs.reshape(1, -1)
        
        for n_rows, condition_rep_tiles_path in enumerate(list_rep_tiles_path):
            for n_cols, tile_path in enumerate(condition_rep_tiles_path):
                
                ax = axs[n_rows, n_cols]
                
                real_path, tile_number = _split_tile_path(tile_path)
                # Load the tile (numpy)
                cur_site = np.load(real_path)
                
                # Adjust contrast and brightness
                tile = improve_brightness(img=cur_site[tile_number,:,:,0], 
                                        contrast_factor=1.5, 
                                        brightness_factor=0)
                
                ax.imshow(tile, cmap='cet_linear_ternary_red_0_50_c52',vmin=0,vmax=1) # 
                ax.axis('off')
                # Set title for each image
                split_path = real_path.split(os.sep)
                marker, condition, cell_line = split_path[-2], split_path[-3], split_path[-4]
                # cluster_score = round(tile_score_per_cluster[tile_score_per_cluster.path==tile_path][f'C{cluster_id}'].values[0]*10**4,3)
                ax.set_title(f"{marker}", fontsize=18)
            
            if by_conditions[0]!='':
                # Set title for condition
                axs[n_rows, 0].text(x=-0.3, y=0, s=f'{condition}', rotation=90, fontsize=24, fontweight='bold', color="orange", transform=axs[n_rows,0].transAxes)
            diff = top_images - (n_cols+1)
            if diff != 0:
                for i in range(n_cols, top_images):
                    axs[n_rows, i].axis('off')
        # Set title for each cluster
        plt.suptitle(f'Cluster {cluster_id}', y=0.9, fontsize=24, fontweight='bold', color="orange")
        plt.tight_layout()
        plt.show()
        
    return rep_tiles_per_cluster
        
def plot_tile_label_pct_in_cluster(tile_score_per_cluster):
    
    colors = ListedColormap(sns.color_palette(cc.glasbey, n_colors=24)) #used for when we have 24 markers
    
    tile_score_per_cluster['short_label'] = tile_score_per_cluster.label.str.split('_').str[0]#.apply(lambda x: "_".join(x)) #include also condition in the label
    
    color_dict = {}
    for i, marker in enumerate(np.unique(tile_score_per_cluster['short_label'])):
        color_dict[marker] = colors(i)
    label_per_cluster = tile_score_per_cluster[['short_label','max_cluster']]
    stack=pd.DataFrame(label_per_cluster.groupby(['max_cluster','short_label']).short_label.count() *100 / label_per_cluster.groupby(['max_cluster']).short_label.count())
    stack = stack.rename(columns={'short_label': 'label_count'})
    stack = stack.reset_index()
    stack = stack.sort_values(by='max_cluster')
    df_pivot = stack.pivot(index='max_cluster', columns='short_label', values='label_count').fillna(0)

    fig = plt.figure(figsize=(10,6))
    for cluster in df_pivot.index:
        row=df_pivot.loc[cluster].sort_values(ascending=False)
        left = 0
        for i,marker in enumerate(row):
            marker_name = row.index[i]
            plt.barh(y=cluster, width=marker, left=left, color=color_dict[marker_name])
            old_left = left
            left = left+marker
            if i <2:
                plt.text(x=(old_left+marker/2), y=cluster-0.1, s=marker_name)
    
    clusters = np.unique(tile_score_per_cluster[['max_cluster']])
    plt.yticks(clusters)
    plt.show()

def plot_histograms(axs, cur_groups, first_cond, second_cond, total_spectra_per_marker_ordered, 
                    color_by_cond, colors, max_per_condition, cluster_counts, plot_delta, plot_cluster_lines=True, 
                    linewidth=1, show_yscale=True, scale_max=True):
    marker_to_organelle = BaseConfig().UMAP_MAPPINGS_MARKERS

    # plot the histograms
    for i, label in enumerate(cur_groups[::-1]):
        if plot_delta:
            label1 = _add_condition_to_label(label, condition=first_cond)
            label2 = _add_condition_to_label(label, condition=second_cond)
            d1 = total_spectra_per_marker_ordered.loc[label1, :]
            d2 = total_spectra_per_marker_ordered.loc[label2, :]
            d = d1 - d2
            axs[i].set_ylim(-10,25) # important!?
        else:
            d = total_spectra_per_marker_ordered.loc[label, :]
        if color_by_cond:
            if first_cond in label: 
                cur_color = colors[first_cond]
            else:
                cur_color = colors[second_cond]
            
            axs[i].fill_between(range(len(d)), d, color=cur_color, label=label, linewidth=linewidth)
            if scale_max:# Set same y label limit for pairs to be compared
                max_limit = max_per_condition.loc[max_per_condition['label']==label, 'max'].values[0]
                axs[i].set_ylim(0, max_limit+(0.25*max_limit))
        else:
            # axs[i].fill_between(range(len(d)), d, color=colors[i], label=label, linewidth=linewidth)
            axs[i].fill_between(range(len(d)), d, color=marker_to_organelle[label]['color'], label=label, linewidth=linewidth)
        axs[i].margins(y=0.25)
        axs[i].set_xticklabels([])
        axs[i].set_xticks([])
        if not show_yscale:
            axs[i].set_yticklabels([])
            axs[i].set_yticks([])
        axs[i].tick_params(axis='y', labelsize=4, length=0, pad=0.1)
        #splitted_label = label.split(sep)
        label_for_plot = label.replace('_', ' ')#'' # comment out - old label editing, when label had also batch, rep, cell line..
        # convert marker name to organelle
        splitted_label = label_for_plot.split(' ')
        marker = splitted_label[0]
        cond = ''
        if color_by_cond:
            cond = f' {splitted_label[1]}'
        label_for_plot = marker_to_organelle[marker]['alias'] + cond
        axs[i].text(1.02, 0.5, label_for_plot, transform=axs[i].transAxes,
                    rotation=0, va='center', ha='left')
        if plot_cluster_lines:
            # add cluster lines to histograms
            prev_count = 0
            for j, cluster in enumerate(cluster_counts.cluster):
                cur_count = cluster_counts.iloc[j]['count']
                cluster_end = cur_count + prev_count

                if cur_count < 15:
                    prev_count = cluster_end
                    continue
                axs[i].axvline(x=cluster_end, color='black',linestyle="--", linewidth=0.4)
                prev_count = cluster_end

        axs[i].spines['bottom'].set_color('lightgray')
        axs[i].spines['top'].set_color('lightgray')
        axs[i].spines['right'].set_color('lightgray')
        axs[i].spines['left'].set_color('lightgray')
        axs[i].margins(x=0)
    return axs

def plot_heatmap_with_clusters_and_histograms(corr_with_clusters, hist_df, labels, save_path=None, to_save=False, color_by_cond=False,
                                              plot_delta=False, sep_histograms = False,
                                              sep = "_", colormap_name = "viridis", plot_hists=True,
                                             filename="codeword_idx_heatmap_and_histograms.tiff",
                                             colors = {"Untreated": "#52C5D5", 'stress': "#F7810F"},
                                             first_cond='stress',second_cond='Untreated',
                                             title=None,calc_linkage=False, linkage_method='average'):
    # calculate linkage matrix
    if calc_linkage:
        linkage = scipy.cluster.hierarchy.linkage(corr_with_clusters.drop(columns=['cluster']), method=linkage_method, metric='euclidean', optimal_ordering=True)
        # methods = single complete average weighted centroid median ward
    else:
        linkage=None
    # colors for the clusters
    n_clusters = np.unique(corr_with_clusters.cluster).shape[0]
    light_gray = [f'C{i}' for i in range(1, n_clusters+1, 2)]
    gray = [f'C{i}' for i in range(2, n_clusters+1, 2)]

    light_gray_dict = {cluster:'lightgray' for cluster in light_gray}
    gray_dict = {cluster:'gray' for cluster in gray}
    color_dict = {**light_gray_dict, **gray_dict}
    # create the heatmap and dendrogram
    kws = dict(cbar_kws=dict(ticks=[-1,0,1]))
    clustermap = sns.clustermap(corr_with_clusters.drop(columns=['cluster']), center=0, cmap=clustermap_colors, vmin=-1, vmax=1, figsize=(7,5), 
                                xticklabels=False, yticklabels=False, row_colors=corr_with_clusters.cluster.map(color_dict), row_linkage=linkage, col_linkage=linkage, **kws)

    # get the indices order from the dendrogram 
    hierarchical_order = clustermap.dendrogram_col.reordered_ind
    
    # prepare labels and filter histograms of wanted labels
    real_labels = []
    for label in labels:
        if label not in np.unique(hist_df.label):
            real_labels += [real_label for real_label in np.unique(hist_df.label) if label in real_label]
        else:
            real_labels.append(label)
    hist_df_cur = hist_df[hist_df.label.isin(real_labels)]
    cur_groups = real_labels
    # Mean the histograms by labels and re-order by the indices order
    total_spectra_per_marker_ordered = hist_df_cur.groupby('label').mean()[hierarchical_order]
    
    if color_by_cond:
        # Set same y label limit for pairs to be compared
        tmp1 = pd.DataFrame(total_spectra_per_marker_ordered.max(axis=1)).reset_index()
        tmp1.columns = ['label', 'max']
        tmp1['label_s'] = tmp1['label'].str.split("_").str[0]
        tmp2 = tmp1.groupby('label_s').max()
        tmp2.columns = ['label', 'max']
        max_per_condition = tmp1[['label', 'label_s']].merge(tmp2['max'], right_index=True, left_on='label_s')
    else:
        max_per_condition = None
    if plot_delta:
        # Pairs to be compared
        list_of_pairs = list(set(total_spectra_per_marker_ordered.reset_index()['label'].str.split("_").str[0]))#.apply(lambda x: '_'.join(x[:2] + x[3:]))))
        cur_groups = list_of_pairs
        # cur_groups = labels
    # calc clusters locations
    cluster_counts = pd.DataFrame(corr_with_clusters.cluster.value_counts()).reset_index()
    cluster_counts.cluster = cluster_counts.cluster.str.replace('C','').astype('int')
    cluster_counts.sort_values(by='cluster', inplace=True)

    if plot_hists:
        # make room for the histograms in the plot
        hist_height = 0.05
        clustermap.fig.subplots_adjust(top=hist_height*len(cur_groups)+1, bottom=hist_height*len(cur_groups))
        
        # add axes for the histograms
        axs=[]
        for i, label in enumerate(cur_groups):
            axs.append(clustermap.fig.add_axes([clustermap.ax_heatmap.get_position().x0, #left
                                                clustermap.ax_heatmap.get_position().y1 +i*hist_height, #0+i*hist_height, #bottom
                                                clustermap.ax_heatmap.get_position().width,  #width
                                                hist_height #height
                                                ]))

        if not color_by_cond:
            # create colors
            colors = sns.color_palette(colormap_name, n_colors=len(cur_groups))
        
        ### PLOT THE HISTOGRAMS ###
        axs = plot_histograms(axs, cur_groups, first_cond, second_cond, total_spectra_per_marker_ordered, 
                        color_by_cond, colors, max_per_condition, cluster_counts, plot_delta)
    
    # fix the cbar appearance 
    clustermap.ax_cbar.set_position([clustermap.ax_col_dendrogram.get_position().x1+0.01, # x location 
                                     clustermap.ax_col_dendrogram.get_position().y0+0.01, # y location
                                     0.01,                                                # width
                                     clustermap.ax_col_dendrogram.get_position().height-0.05]) #height
    clustermap.ax_cbar.set_title('Pearson r',fontsize=6)
    clustermap.cax.tick_params(axis='y', labelsize=6, length=0, pad=0.1)
   
    # add cluster lines to the heatmap
    prev_count = 0
    for j, cluster in enumerate(cluster_counts.cluster):
        cur_count = cluster_counts.iloc[j]['count']

        cluster_end = cur_count + prev_count
        
        if cur_count < 15:
            prev_count = cluster_end
            continue
        clustermap.ax_row_colors.text(y=cluster_end-(cur_count/2), x=0.45, s=cluster, fontsize=6)
        clustermap.ax_heatmap.plot([cluster_end,cluster_end], [prev_count,cluster_end], color='black',linestyle="--", linewidth=1)
        clustermap.ax_heatmap.plot([prev_count,cluster_end], [cluster_end,cluster_end], color='black',linestyle="--", linewidth=1)
        clustermap.ax_heatmap.plot([prev_count,prev_count], [prev_count,cluster_end], color='black',linestyle="--", linewidth=1)
        clustermap.ax_heatmap.plot([prev_count,cluster_end], [prev_count,prev_count], color='black',linestyle="--", linewidth=1)
        prev_count = cluster_end
    # fix dendrogram appearance 
    clustermap.ax_col_dendrogram.set_visible(False)
    orig_row_den_pos = clustermap.ax_row_dendrogram.get_position()
    clustermap.ax_row_dendrogram.set_position([clustermap.ax_heatmap.get_position().x1 + 0.001,
                                               orig_row_den_pos.y0,
                                               orig_row_den_pos.x1-orig_row_den_pos.x0,
                                               orig_row_den_pos.y1-orig_row_den_pos.y0]) #clustermap.ax_col_dendrogram.get_position().x1-clustermap.ax_col_dendrogram.get_position().x0])
    clustermap.ax_row_dendrogram.invert_xaxis()

    if title:
        clustermap.fig.suptitle(title, x = clustermap.ax_col_dendrogram.get_position().x0+(clustermap.ax_col_dendrogram.get_position().x1-clustermap.ax_col_dendrogram.get_position().x0)/2,
                               y = clustermap.ax_col_dendrogram.get_position().y1+0.05)
    if to_save and not sep_histograms:
        clustermap.figure.savefig(os.path.join(save_path, filename),bbox_inches='tight', dpi=300)
    if sep_histograms:
        # fig, axs = plt.subplots(nrows = len(cur_groups), figsize = (6.7, 0.2*len(cur_groups)))
        fig = plt.figure( figsize=(7,5))
        hist_height = 0.05
        axs=[]
        for i, label in enumerate(cur_groups):
            axs.append(fig.add_axes([0, #left
                                     0 +i*hist_height, #0+i*hist_height, #bottom
                                     clustermap.ax_heatmap.get_position().width,  #width
                                     hist_height #height
                                     ]))
        axs = plot_histograms(axs[::-1], cur_groups, first_cond, second_cond, total_spectra_per_marker_ordered, 
                        color_by_cond, colors, max_per_condition, cluster_counts, plot_delta)
        fig.subplots_adjust(hspace=0)
        if to_save:
            fig.savefig(os.path.join(save_path, "only_histograms_" + filename),bbox_inches='tight', dpi=300)
    
    return None

def plot_hists_supp_1A(hist_df, labels, corr_with_clusters, hierarchical_order=None, sort=False, color_by_cond=False, plot_delta=False, 
                       to_save=False, colormap_name='viridis',figsize=(5,5), first_cond='stress', second_cond='Untreated', 
                       save_path=None, filename=None, colors = {"Untreated": "#52C5D5", 'stress': "#F7810F"}, to_mean=True, plot_cluster_lines=True, scale_max=False):
    real_labels = []
    for label in labels:
        if label not in np.unique(hist_df.label):
            real_labels += [real_label for real_label in np.unique(hist_df.label) if label in real_label]
        else:
            real_labels.append(label)
    hist_df_cur = hist_df[hist_df.label.isin(real_labels)]
    cur_groups = real_labels

    # Mean the histograms by labels 
    total_spectra_per_marker_ordered = hist_df_cur.copy()
    total_spectra_per_marker_ordered.set_index('label', inplace=True)
    if to_mean:
        total_spectra_per_marker_ordered = hist_df_cur.groupby('label').mean()

    #re-order by the indices order
    if sort:
        total_spectra_per_marker_ordered=total_spectra_per_marker_ordered[hierarchical_order]

    if color_by_cond and scale_max:
        # Set same y label limit for pairs to be compared
        tmp1 = pd.DataFrame(total_spectra_per_marker_ordered.max(axis=1)).reset_index()
        tmp1.columns = ['label', 'max']
        tmp1['label_s'] = tmp1['label'].str.split("_").str[0]
        tmp2 = tmp1.groupby('label_s').max()
        tmp2.columns = ['label', 'max']
        max_per_condition = tmp1[['label', 'label_s']].merge(tmp2['max'], right_index=True, left_on='label_s')
    else:
        max_per_condition = None
    if plot_delta:
        # Pairs to be compared
        cur_groups = labels
    
    # calc clusters locations
    cluster_counts = pd.DataFrame(corr_with_clusters.cluster.value_counts()).reset_index()
    cluster_counts.cluster = cluster_counts.cluster.str.replace('C','').astype('int')
    cluster_counts.sort_values(by='cluster', inplace=True)


    if to_mean:
        fig, axs = plt.subplots(nrows = len(cur_groups), figsize=figsize)
        if plot_delta:
            colors = sns.color_palette(colormap_name, n_colors=len(cur_groups)*2)[0::3]
        elif not color_by_cond:
            # create colors
            colors = [ListedColormap([colormap_name])(0)]*len(cur_groups)
        axs = plot_histograms(axs[::-1], cur_groups, first_cond, second_cond, total_spectra_per_marker_ordered, False, 
                    color_by_cond, colors, max_per_condition, cluster_counts, plot_delta, plot_cluster_lines, linewidth=2, show_yscale=False, scale_max=False)
    else:
        fig, axs = plt.subplots(nrows = len(cur_groups)*2, figsize=figsize)
        if not color_by_cond:
            # create colors
            colors = [ListedColormap([colormap_name])(0)]*len(cur_groups)*2
        axs = plot_histograms_not_mean(axs[::-1], cur_groups, first_cond, second_cond,total_spectra_per_marker_ordered,
                                 color_by_cond, colors, max_per_condition, cluster_counts, plot_cluster_lines, show_yscale=False)
    
    fig.subplots_adjust(hspace=0)
    if to_save:
        fig.savefig(os.path.join(save_path, filename),bbox_inches='tight', dpi=300)

def plot_histograms_not_mean(axs, cur_groups, first_cond, second_cond, total_spectra_per_marker_ordered, 
                    color_by_cond, colors, max_per_condition, cluster_counts, plot_cluster_lines=True, show_yscale=True):
    # plot the histograms
    for i, label in enumerate(cur_groups[::-1]):
        d = total_spectra_per_marker_ordered.loc[label, :]
        for j in range(d.shape[0]):
            cur_d = d.iloc[j,:]
            ax = axs[i*d.shape[0] + j]
            if color_by_cond:
                if first_cond in label: 
                    cur_color = colors[first_cond]
                else:
                    cur_color = colors[second_cond]
                ax.fill_between(range(len(cur_d)), cur_d, color=cur_color, label=label, linewidth=2)
                # Set same y label limit for pairs to be compared
                # max_limit = max_per_condition.loc[max_per_condition['label']==label, 'max'].values[0]
                # ax.set_ylim(0, max_limit+(0.25*max_limit))
            else:
                ax.fill_between(range(len(cur_d)), cur_d, color=colors[i], label=label, linewidth=2)
        
            ax.margins(y=0.25)
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.tick_params(axis='y', labelsize=4, length=0, pad=0.1)
            ax.text(1.02, 0.5, label.replace('_', ' '), transform=ax.transAxes,
                    rotation=0, va='center', ha='left')
            if plot_cluster_lines:
                # add cluster lines to histograms
                prev_count = 0
                for j, cluster in enumerate(cluster_counts.cluster):
                    cur_count = cluster_counts.iloc[j]['count']
                    cluster_end = cur_count + prev_count

                    if cur_count < 15:
                        prev_count = cluster_end
                        continue
                    ax.axvline(x=cluster_end, color='black',linestyle="--", linewidth=1)
                    prev_count = cluster_end

            ax.spines['bottom'].set_color('lightgray')
            ax.spines['top'].set_color('lightgray')
            ax.spines['right'].set_color('lightgray')
            ax.spines['left'].set_color('lightgray')
            ax.margins(x=0)
            if not show_yscale:
                ax.set_yticklabels([])
                ax.set_yticks([])
    return axs

def plot_heatmap_with_clusters_supp_1A(corr_with_clusters, save_path=None, to_save=False,                                            
                                             filename="codeword_idx_heatmap_and_histograms.tiff",
                                             figsize=(9,3), cmap=None):
    col_colors_df = {'C1': 'olivedrab', 'C2': 'teal', 'C3': 'goldenrod','C4':'purple'}
    # create the heatmap and dendrogram
    kws = dict(cbar_kws=dict(ticks=[-1,0,1]))
    clustermap = sns.clustermap(corr_with_clusters.drop(columns=['cluster']), center=0, cmap=cmap, vmin=-1, vmax=1, 
                                figsize=figsize, xticklabels=False, yticklabels=False, col_colors=corr_with_clusters['cluster'].map(col_colors_df), **kws)
    clustermap.ax_row_dendrogram.set_visible(False)

    # calc clusters locations
    cluster_counts = pd.DataFrame(corr_with_clusters.cluster.value_counts()).reset_index()
    cluster_counts.cluster = cluster_counts.cluster.str.replace('C','').astype('int')
    cluster_counts.sort_values(by='cluster', inplace=True)
    
    # fix the cbar appearance 
    clustermap.ax_cbar.set_position([clustermap.ax_col_dendrogram.get_position().x1+0.01, # x location 
                                     clustermap.ax_col_dendrogram.get_position().y0+0.01, # y location
                                     0.01,                                                # width
                                     clustermap.ax_col_dendrogram.get_position().height-0.05]) #height
    clustermap.ax_cbar.set_title('Pearson r',fontsize=6)
    clustermap.cax.tick_params(axis='y', labelsize=6, length=0, pad=0.1)
   
    # add cluster lines to the heatmap
    prev_count = 0
    for j, cluster in enumerate(cluster_counts.cluster):
        cur_count = cluster_counts.iloc[j]['count']

        cluster_end = cur_count + prev_count
        
        if cur_count < 15:
            prev_count = cluster_end
            continue
        # clustermap.ax_heatmap.axvline(x=cluster_end, color='black',linestyle="--", linewidth=1)
        clustermap.ax_heatmap.plot([cluster_end,cluster_end], [prev_count,cluster_end], color='black',linestyle="--", linewidth=1)
        clustermap.ax_heatmap.plot([prev_count,cluster_end], [cluster_end,cluster_end], color='black',linestyle="--", linewidth=1)
        clustermap.ax_heatmap.plot([prev_count,prev_count], [prev_count,cluster_end], color='black',linestyle="--", linewidth=1)
        clustermap.ax_heatmap.plot([prev_count,cluster_end], [prev_count,prev_count], color='black',linestyle="--", linewidth=1)
        clustermap.ax_col_colors.text(x=cluster_end-(cur_count/2), y=0.8, s=cluster, fontsize=6)
        prev_count = cluster_end

    if to_save:
        clustermap.figure.savefig(os.path.join(save_path, filename),bbox_inches='tight', dpi=300)
    return clustermap.dendrogram_col.reordered_ind

def create_correlation_graph(correlation_matrix, top_positive=True, num_edges=2, scale_factor=10):
    graph = nx.Graph()
    marker_to_organelle = BaseConfig().UMAP_MAPPINGS_MARKERS
    for marker, row in correlation_matrix.iterrows():
        organelle = marker_to_organelle[marker][BaseConfig().UMAP_MAPPINGS_ALIAS_KEY]
        sorted_correlations = row.drop(marker).sort_values(ascending=not top_positive).head(num_edges)
        for other_marker, correlation in sorted_correlations.items():
            other_organelle = marker_to_organelle[other_marker][BaseConfig().UMAP_MAPPINGS_ALIAS_KEY]
            graph.add_edge(organelle, other_organelle, weight=abs(correlation), color=correlation, 
                           scaled_weight=abs(correlation)*scale_factor)
    return graph

def draw_correlation_graph(graph, title, cmap, vmin, vmax, save_path=None, filename="corr_graph.tiff", to_save=False):
    node_size = 10
    # Compute the spring layout with the scaled weights
    pos = nx.spring_layout(graph, weight='scaled_weight', seed=42, k=1)
    # pos = nx.spring_layout(graph, weight='weight', seed=42) #shell_layout
    # pos = nx.kamada_kawai_layout(graph)
    fig, ax = plt.subplots(figsize=(6, 4))
    nx.draw_networkx_nodes(graph, pos, node_color='none', node_size=node_size, edgecolors='none', 
                           linewidths=0.5, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax, 
                            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.05'),
                            verticalalignment='center')
    edge_colors = [graph[u][v]['color'] for u,v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, edge_cmap=cmap, edge_vmin=vmin, 
                           edge_vmax=vmax, ax=ax, node_size=node_size, width=3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, orientation='horizontal', label='Correlation Strength',shrink=0.2,pad=0)
    # cax = plt.axes([0.8, 0.85, 0.1, 0.02])  # Adjust position and size of colorbar axis
    # cbar = plt.colorbar(sm, cax=cax, orientation='horizontal', label='Correlation Strength')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if to_save:
        plt.savefig(os.path.join(save_path, filename),bbox_inches='tight', dpi=300)
    plt.show()
    
def create_lables_heatmap(df, title, save_path=None, to_save=False, filename=None, corr_method='pearson'):
    corrs = df.T.corr(method=corr_method)
    linkage = scipy.cluster.hierarchy.linkage(corrs, 
                                            method='average', # 'complete', 'average', 'median', 'ward'
                                            metric='euclidean', 
                                            optimal_ordering=True)
    clustermap = sns.clustermap(corrs, center=0, cmap='bwr', vmin=-1, vmax=1, figsize=(5,5), row_linkage=linkage, 
                                col_linkage=linkage)

    clustermap.fig.suptitle(title, y=1.05)
    row_order = clustermap.dendrogram_row.reordered_ind
    marker_order = corrs.columns[row_order]

    row_axes = clustermap.ax_heatmap.get_yaxis()
    col_axes = clustermap.ax_heatmap.get_xaxis()
    row_axes.set_ticks(np.arange(len(marker_order)) + 0.5)
    row_axes.set_ticklabels(marker_order, rotation=0, fontsize=6)
    col_axes.set_ticks(np.arange(len(marker_order)) + 0.5)
    col_axes.set_ticklabels(marker_order, rotation=90, fontsize=6)
    
    clustermap.ax_row_dendrogram.set_visible(False)
    clustermap.ax_cbar.set_position([clustermap.ax_col_dendrogram.get_position().x1+0.01, # x location 
                                     clustermap.ax_col_dendrogram.get_position().y0+0.01, # y location
                                     0.01,                                                # width
                                     clustermap.ax_col_dendrogram.get_position().height-0.05]) #height
    clustermap.ax_cbar.set_title(f'{corr_method} r',fontsize=6)
    clustermap.cax.tick_params(axis='y', labelsize=6, length=0, pad=0.1) 
    clustermap.ax_heatmap.set_xlabel('Marker', fontsize=12)
    clustermap.ax_heatmap.set_ylabel('Marker', fontsize=12)
    
    if to_save:
        plt.savefig(os.path.join(save_path, filename),bbox_inches='tight', dpi=300)
    plt.show()
    return marker_order, corrs

def calc_deltas(df, first_cond, second_cond):
    markers = np.unique(df['label'].str.split("_").str[0])
    average_hist = df.groupby('label').mean()
    deltas = pd.DataFrame(index = markers, columns=average_hist.columns)
    # calc delta and save in a new df
    for marker in markers:
        first = average_hist.loc[f'{marker}_{first_cond}']
        second =  average_hist.loc[f'{marker}_{second_cond}']
        deltas.loc[marker] =  first - second

    return deltas

def analyse_deltas(df, first_cond, second_cond, 
                  heatmap_title = None, graph_filename=None, save_path=None, 
                   to_save=False, heatmap_filename=None, plot_network=True):
    
    deltas = calc_deltas(df, first_cond, second_cond)
    markers_order , marker_delta_corr = create_lables_heatmap(deltas, heatmap_title, save_path=save_path, 
                                                  to_save=to_save, filename=heatmap_filename)
    if plot_network:
        positive_graph = create_correlation_graph(marker_delta_corr, top_positive=True, scale_factor=10)
        
        draw_correlation_graph(positive_graph, '', 
                               cmap=plt.get_cmap('Reds'), vmin=0, vmax=1,
                              save_path=save_path, filename=f"Positive_{graph_filename}", to_save=to_save)

    return None

'''
OLD FUNCTIONS TO CREATE AND DRAW GRAPH OF BOTH POSITIVE AND NEGATIVE CORRELATIONS
def create_bicorrelation_graph(correlation_matrix, num_edges=2):
    graph = nx.Graph()
    for marker, row in correlation_matrix.iterrows():
        # first take positive correlations
        sorted_correlations = row.drop(marker).sort_values(ascending=False).head(num_edges)
        for other_marker, correlation in sorted_correlations.items():
            graph.add_edge(marker, other_marker, weight=abs(correlation), color=correlation, label='positive')
        # then take negative correlations
        sorted_correlations = row.drop(marker).sort_values(ascending=True).head(num_edges)
        for other_marker, correlation in sorted_correlations.items():
            graph.add_edge(marker, other_marker, weight=abs(correlation), color=correlation, label='negative')
    return graph
def draw_bicorrelation_graph(graph, title, cmap_pos, cmap_neg, vmin_pos = 0, vmax_pos = 1,
                             vmin_neg=-1, vmax_neg=0, save_path=None, filename="corr_graph.tiff", to_save=False):
    pos = nx.spring_layout(graph, weight='weight', seed=42) #shell_layout
    fig, ax = plt.subplots(figsize=(15, 8))
    nx.draw_networkx_nodes(graph, pos, node_color='white', node_size=1500, edgecolors='black', linewidths=0.5, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    #draw positive edges
    positive_edges = [edge for edge in edge_labels if edge_labels[edge]=='positive']
    positive_edge_colors = [graph[u][v]['color'] for u,v in positive_edges]
    nx.draw_networkx_edges(graph, pos, edge_color=positive_edge_colors, edgelist=positive_edges,
                                   edge_cmap=cmap_pos, edge_vmin=vmin_pos, edge_vmax=vmax_pos, ax=ax)
    #draw negative edges
    negative_edges = [edge for edge in edge_labels if edge_labels[edge]=='negative']
    negative_edge_colors = [graph[u][v]['color'] for u,v in negative_edges]
    nx.draw_networkx_edges(graph, pos, edge_color=negative_edge_colors, edgelist=negative_edges,
                                   edge_cmap=cmap_neg, edge_vmin=vmin_neg, edge_vmax=vmax_neg, ax=ax)
    
    sm_pos = plt.cm.ScalarMappable(cmap=cmap_pos, norm=plt.Normalize(vmin=vmin_pos, vmax=vmax_pos))
    sm_pos.set_array([])
    plt.colorbar(sm_pos, orientation='vertical', label='Positive Correlation Strength',shrink=0.2, pad=0)
    sm_neg = plt.cm.ScalarMappable(cmap=cmap_neg, norm=plt.Normalize(vmin=vmin_neg, vmax=vmax_neg))
    sm_neg.set_array([])
    plt.colorbar(sm_neg, orientation='vertical', label='Negative Correlation Strength',shrink=0.2, pad=0, location='left')
    plt.title(title)
    plt.axis('off')
    if to_save:
        plt.savefig(os.path.join(save_path, filename),bbox_inches='tight', dpi=300)
    plt.show()
'''