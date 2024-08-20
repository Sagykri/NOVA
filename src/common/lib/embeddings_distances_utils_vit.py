import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np
import pandas as pd
import logging
from scipy.spatial.distance import pdist
from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
from src.common.configs.base_config import BaseConfig
import matplotlib.pyplot as plt
import seaborn as sns
###############################################################
# Utils for calculating distances between  labels, based on the full latent space (Embeddings) 
###############################################################


###############################################################
# Distance metrics sklearn: https://scikit-learn.org/stable/modules/metrics.html#metrics
DISTANCE_METRICS = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine']
AGG_FUNCTIONS = [np.mean, np.median]
###############################################################

def create_markers_centroids_df(all_labels, all_embedings_data, exclude_DAPI=True, markers_to_exclude=[]):  
    """Create a pd.DataFrame of centroids embedddings and experimental settings 
    columns are ['batch','cell_line','condition','rep','marker', 'embeddings_centroid'] 

    Args:
        all_labels (np.array): array of strings, each row is in a format of "batch_cell_line_condition_rep_marker"
        all_embedings_data (np.ndarray): latent featuers (each row has 9216 columns)
    """
    assert all_labels.shape[0]==all_embedings_data.shape[0]    
    labels_df = pd.DataFrame(data=all_labels, columns=['label'])
    labels_df['label'] = labels_df.label.str.replace('_16bit_no_downsample','')  #TODO: delete workaround since batch folder have "_"
    labels_df['label'] = labels_df.label.str.replace('_50percent','')  
    # Calculate embeddings centroids (a numpy matrix) + within marker similarities
    marker_centroids = pd.DataFrame()
    #within_marker_similaities = pd.DataFrame()
    centroids = []
    labels = []
    #similarities = [] # used for calculating mean similarty per well
    for label, label_df in labels_df.groupby(['label']): # use indexes of labels to find corresponding embeddings
        logging.info(f'[create_markers_centroids_df] adding label {label[0]}')
        cur_embeddings = all_embedings_data[label_df.index]
        # calc current label centroid
        cur_centroid = np.median(cur_embeddings, axis=0)
        centroids.append(cur_centroid.tolist())
        labels.append(label[0])
        # calc current label mean similarity
        # mean_marker_similarity =  (1/(1+pdist(cur_embeddings, metric='euclidean'))).mean() #using pdist since we don't want a distance matrix here, but just all the pairwise distances (without repeats & without self distances)
        # similarities.append(mean_marker_similarity)

    marker_centroids['label'] = labels
    marker_centroids[['marker','cell_line','condition','batch','rep']] = marker_centroids.label.str.split('_', expand=True)
    marker_centroids['embeddings_centroid'] = centroids
    marker_centroids.drop(columns=['label'], inplace=True)
    logging.info(f'[create_markers_centroids_df] created df with shape {marker_centroids.shape}')

    # within_marker_similaities['label'] = labels
    # within_marker_similaities[['batch','cell_line','condition','rep','marker']] = within_marker_similaities.label.str.split('_', expand=True)
    # within_marker_similaities.drop(columns=['label'], inplace=True)
    # within_marker_similaities['marker_similarity'] = similarities   
    
    # Exclude embeddings of DAPI marker  
    if exclude_DAPI:
        marker_centroids = marker_centroids[marker_centroids['marker']!='DAPI']
        #within_marker_similaities = within_marker_similaities[within_marker_similaities['marker']!='DAPI']
    for m in markers_to_exclude:
        marker_centroids = marker_centroids[marker_centroids['marker']!=m]
    return marker_centroids #, within_marker_similaities

def save_excel_with_sheet_name(filename, input_folders, df):
    with pd.ExcelWriter(filename) as writer:
        for input_folder in input_folders:
            batch_name = input_folder.split(os.sep)[-1].replace('_16bit_no_downsample','')
            df[df.batch==batch_name].to_excel(writer, sheet_name=batch_name, index=False)


def between_cell_lines_sep_rep_dist(marker_centroids, distances_main_folder, dist_func = pairwise_distances, suff=None, compare_identical_reps=True):
    # given batch, how similar are the cell lines? also treating different reps as batches
    # dist_func should compute pairwise distances between all rows of input matrix, such that the output is a distance matrix D such that D_{i, j} is the distance between 
    # the ith and jth vectors of the given matrix X, like in sklearn.metrics.pairwise_distances
    markers = marker_centroids['marker'].unique()
    cell_lines_conditions = marker_centroids['cell_line_condition'].unique()
    if len(cell_lines_conditions) <= 1:
        logging.info(f"Skipping comparison of cell lines since cell lines: {cell_lines_conditions}")
        return None
    elif len(cell_lines_conditions) > 1:
        ## given batch and rep, how similar are the cell lines?
        if compare_identical_reps:
            between_cell_lines_sep_batch_rep = pd.DataFrame(columns=['batch','rep','marker','cell_line_condition'] + cell_lines_conditions.tolist())
            for name, group in marker_centroids.groupby(['batch','rep'])[['cell_line_condition', 'marker','embeddings_centroid']]:
                for marker in markers: 
                    cur_marker = group[group['marker']==marker]
                    if cur_marker.shape[0] == 0:
                        logging.info(f"Skipping comparison of cell_line_conds for {marker} in {name} since cell_line_conds: {cur_marker.shape[0]}")
                        continue
                    x = np.stack(cur_marker['embeddings_centroid'].values, axis=0)
                    cell_lines_conds_similarities = pd.DataFrame(dist_func(x, metric='euclidean', n_jobs=-1), 
                                                                columns=cur_marker.cell_line_condition.values, 
                                                                index=cur_marker.cell_line_condition).reset_index()
                    num_rows, _ = cell_lines_conds_similarities.shape

                    # Nullify the diagonal elements
                    for i in range(num_rows):
                        cell_lines_conds_similarities.iat[i, i+1] = np.nan

                    # combine marker similiarites in one df
                    cell_lines_conds_similarities_for_df = cell_lines_conds_similarities
                    cell_lines_conds_similarities_for_df['batch'] = name[0]
                    cell_lines_conds_similarities_for_df['rep'] = name[1]
                    cell_lines_conds_similarities_for_df['marker'] = marker
                    between_cell_lines_sep_batch_rep = pd.concat([between_cell_lines_sep_batch_rep, cell_lines_conds_similarities_for_df])
            #save_excel_with_sheet_name(os.path.join(distances_main_folder,'between_cell_lines_conds_similarities_rep_rep.xlsx'), input_folders, between_cell_lines_sep_batch_rep)
            between_cell_lines_sep_batch_rep.to_csv(os.path.join(distances_main_folder,f'between_cell_lines_conds_similarities_rep{suff}.csv'), index=False)
            return between_cell_lines_sep_batch_rep
        else:
            between_cell_lines = pd.DataFrame(columns=['label_1','label_2','dist','marker'])
            for marker in markers: 
                cur_marker = marker_centroids[marker_centroids['marker']==marker].reset_index(drop=True)
                if cur_marker.shape[0] == 0:
                    logging.info(f"Skipping comparison of cell_line_conds for {marker} since cell_line_conds: {cur_marker.shape[0]}")
                    continue
                x = np.stack(cur_marker['embeddings_centroid'].values, axis=0)
                labels = cur_marker.cell_line_condition + '_' + cur_marker.batch + '_' + cur_marker.rep
                dists = pairwise_distances(x, metric='euclidean', n_jobs=-1)
                triu_indices = np.triu_indices_from(dists, k=0) # change to k=1 to ignore the diagonal!
                cell_lines_conds_similarities=pd.DataFrame(data={'label_1':labels[triu_indices[0]].reset_index(drop=True),
                                'label_2':labels[triu_indices[1]].reset_index(drop=True),
                                'dist':dists[triu_indices]})
                cell_lines_conds_similarities['marker']=marker
                between_cell_lines = pd.concat([between_cell_lines, cell_lines_conds_similarities])
            between_cell_lines[['cell_line_1','condition_1', 'batch_1','rep_1']] = between_cell_lines['label_1'].str.split('_', expand=True)
            between_cell_lines[['cell_line_2','condition_2', 'batch_2','rep_2']] = between_cell_lines['label_2'].str.split('_', expand=True)
            between_cell_lines['cell_line_condition_1'] = between_cell_lines['cell_line_1'] + '_' + between_cell_lines['condition_1']
            between_cell_lines['cell_line_condition_2'] = between_cell_lines['cell_line_2'] + '_' + between_cell_lines['condition_2']
            between_cell_lines.drop(columns=['label_1','label_2','cell_line_1','cell_line_2','condition_1','condition_2'], inplace=True)
            between_cell_lines.to_csv(os.path.join(distances_main_folder,f'between_cell_lines_conds_distances{suff}.csv'), index=False)
            return between_cell_lines

def calc_embeddings_distances_for_vit(embeddings, labels, config_data, 
                                      distances_main_folder, suff=None,
                                      compare_identical_reps=True):
    """Main function to calculate embeddings distances
    """
    # ------------------------------------------------------------------------------------------ 
    marker_centroids = create_markers_centroids_df(labels, embeddings, exclude_DAPI=False, markers_to_exclude=config_data.MARKERS_TO_EXCLUDE)
    # ------------------------------------------------------------------------------------------  
    marker_centroids['cell_line_condition'] = marker_centroids['cell_line'] + '_' + marker_centroids['condition']
    
    return between_cell_lines_sep_rep_dist(marker_centroids, distances_main_folder,
                                     suff=suff,compare_identical_reps=compare_identical_reps)

    return None

def plot_distances_plot(distances_main_folder, 
                        convert_markers_names_to_organelles=False, suff=None,
                        compare_identical_reps=True):
    if compare_identical_reps:
        dists = pd.read_csv(os.path.join(distances_main_folder,f'between_cell_lines_conds_similarities_rep{suff}.csv'))
        cell_line_conds = np.unique(dists.cell_line_condition)
    else:
        dists = pd.read_csv(os.path.join(distances_main_folder,f'between_cell_lines_conds_distances{suff}.csv'))
        cell_line_conds = np.unique(dists.cell_line_condition_1)

    if convert_markers_names_to_organelles:
        logging.info("Converting markers names to organelles")
        # Convert markers names to organelles
        base_config = BaseConfig()
        marker_aliases = base_config.UMAP_MAPPINGS_MARKERS

        alias_mapping = {marker: details[base_config.UMAP_MAPPINGS_ALIAS_KEY] for marker, details in marker_aliases.items()}
        logging.info(f'markers before: {dists["marker"].drop_duplicates()}')
        dists['marker'] = dists['marker'].map(alias_mapping)
        logging.info(f'markers after: {dists["marker"].drop_duplicates()}')


    for base, target in combinations(cell_line_conds, 2):
        logging.info(f"{base} Vs. {target}")
        dists_cond, dists_cond_order = get_dists_between_baseline_and_target(dists, base, target,compare_identical_reps=compare_identical_reps)
        plot_distances_boxplot(dists_cond, dists_cond_order,
                    base, target,
                    transpose=False, figsize = (20,5), 
                    savefolder=distances_main_folder,suff=suff,
                    compare_identical_reps=compare_identical_reps)


def get_dists_between_baseline_and_target(dists, baseline_label, target_label, 
                                          scale=True, compare_identical_reps=True):
    logging.info(f"baseline_label={baseline_label}, target_label={target_label}")
    dists_filtered = dists.copy()
    if compare_identical_reps:
        # Filter by condition
        # Filter column by cond
        filtered_columns = [target_label, 'cell_line_condition', 'marker', 'batch','rep']
        dists_filtered = dists_filtered[filtered_columns]
        # Filter row by cond
        dists_filtered = dists_filtered[dists_filtered['cell_line_condition'].str.endswith(f'{baseline_label}')]

        if len(dists_filtered) == 0:
            logging.info(f"No values for {baseline_label} Vs. {target_label}")
            return None, None

        if scale:        
            # min max scale for each group of batch-rep distances:
            for name, group in dists_filtered.groupby(['batch','rep'])[target_label]:
                group_min = group.min()
                group_max = group.max()
                scaled_group = (group-group_min) / (group_max - group_min)
                dists_filtered.loc[group.index, target_label] = scaled_group
        dists_filtered.drop(columns=['batch','rep'], inplace=True)
        # dists_filtered_order = dists_filtered.groupby("marker")[f'{target_label}'].mean().sort_values(ascending=False).index
        median_variance = dists_filtered.groupby("marker")[target_label].agg(['median', 'var'])
        dists_filtered_order = median_variance.sort_values(by=['median', 'var'], ascending=[False, False]).index
    else:
        dists_filtered = dists_filtered[((dists_filtered['cell_line_condition_1']==target_label) & \
                       (dists_filtered['cell_line_condition_2']==baseline_label)) | 
                       ((dists_filtered['cell_line_condition_1']==baseline_label) & \
                       (dists_filtered['cell_line_condition_2']==target_label))]
        # dists_filtered_order = dists_filtered.groupby("marker")['dist'].mean().sort_values(ascending=False).index
        median_variance = dists_filtered.groupby("marker")['dist'].agg(['median', 'var'])
        dists_filtered_order = median_variance.sort_values(by=['median', 'var'], ascending=[False, False]).index
        if len(dists_filtered) == 0:
            logging.info(f"No values for {baseline_label} Vs. {target_label}")
            return None, None
        
    
    logging.info(f"dists_filtered.shape: {dists_filtered.shape}")
    return dists_filtered, dists_filtered_order

def plot_distances_boxplot(dists, dists_order, baseline_label, target_label,
                           title=None, savefolder=None, ax=None, transpose=False,
                        figsize=(6,4), fontsize=13, title_fontsize=20, suff=None,
                        compare_identical_reps=True):
    axis_label = "Distances"
    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if not transpose:
        x = 'marker'
        y = target_label if compare_identical_reps else 'dist'
        sns.boxplot(data=dists,x=x,y=y ,ax=ax,fliersize=0,
                    order=dists_order,linewidth=1, color='gray') 
        sns.stripplot(data=dists, x=x, y=y, color='black',
              dodge=True, order=dists_order, size=2, jitter=0, marker='o', ax=ax, edgecolor='k',legend=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsize)
            
        ax.set_ylabel(axis_label)
        ax.set_xlabel('')
    else:
        y = 'marker'
        x = target_label if compare_identical_reps else 'dist'
        sns.boxplot(data=dists,x=x,y=y ,ax=ax,fliersize=0,
                    order=dists_order,linewidth=1, color='gray') 
        sns.stripplot(data=dists, x=x, y=y, color='black',
              dodge=True, order=dists_order, size=2, jitter=0, marker='o', ax=ax, edgecolor='k',legend=False)
        ax.set_xlabel(axis_label)
        ax.set_ylabel('')
        
    title = title if title is not None else f'{baseline_label} VS. {target_label}'
    ax.set_title(title, fontsize=title_fontsize)
    
    plt.tight_layout()
    if savefolder is not None:
        folderpath = os.path.join(savefolder)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        savepath = os.path.join(folderpath, f'{baseline_label}_VS_{target_label}{suff}')
        if not compare_identical_reps:
            savepath += '_all_reps_batches'
        plt.savefig(f"{savepath}.png", dpi=300)
        # plt.savefig(f"{savepath}.eps", dpi=300, format='eps')
        
    
    return ax