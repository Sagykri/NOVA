import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np
import pandas as pd
import logging
from copy import deepcopy
from scipy.spatial.distance import cdist, pdist
from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
from src.common.lib.synthetic_multiplexing import __get_multiplexed_embeddings, __embeddings_to_df
from src.common.lib.utils import load_config_file, get_if_exists
from src.common.lib.embeddings_utils import load_embeddings, load_indhists
from src.common.configs.base_config import BaseConfig
import seaborn as sns
import matplotlib.pyplot as plt
###############################################################
# Utils for calculating distances between  labels, based on the full latent space (Embeddings) 
###############################################################


###############################################################
# Distance metrics sklearn: https://scikit-learn.org/stable/modules/metrics.html#metrics
DISTANCE_METRICS = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine']
AGG_FUNCTIONS = [np.mean, np.median]
###############################################################

def fetch_saved_embeddings(config_model, config_data, embeddings_type):
    """Couple embedding vector with it's corresponding label (label == batch-cellline-condition-rep-marker)

    Args:
        exclude_DAPI (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: pd.DataFrame
    """
    embeddings_layer = get_if_exists(config_data, 'EMBEDDINGS_LAYER', None)
    if 'vqvec' in embeddings_layer:
        all_embedings_data, all_labels = load_embeddings(config_model=config_model, 
                                                     config_data= config_data,
                                                     embeddings_type=embeddings_type)
        all_embedings_data = all_embedings_data.reshape(all_embedings_data.shape[0], -1)
    elif 'vqindhist' in embeddings_layer:
        all_embedings_data, all_labels = load_indhists(config_model=config_model, config_data=config_data, embeddings_type=embeddings_type)
    logging.info(f"[load_embeddings] {all_embedings_data.shape}, {all_labels.shape}, example label: {all_labels[0]}")    
    return all_embedings_data, all_labels

def calc_cellprofiler_distances():
    #load labels and features
    cellprofiler_features_folder = "/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/cell_profiler"
    batches_folders = ['batch6','batch7','batch8','batch9']
    batch_labels, batch_features = [],[]
    for batch in batches_folders:
        batch_df = pd.read_csv(os.path.join(cellprofiler_features_folder, batch, 'combined', f'stress_all_markers_concatenated-by-object-type_{batch}.csv'))
        batch_features.append(batch_df.drop(columns=['marker','replicate','treatment','cell_line','Unnamed: 0']))
        cur_batch_labels = batch_df[['cell_line','treatment','replicate','marker']]
        cur_batch_labels['batch'] = batch
        cur_batch_labels = cur_batch_labels[['batch','cell_line','treatment','replicate','marker']]
        batch_labels.append(cur_batch_labels)
    batch_labels = pd.concat(batch_labels)
    batch_features = pd.concat(batch_features)

    # feature filtering and scaling
    constant_cols = batch_features.columns[batch_features.nunique() == 1]
    batch_features = batch_features.drop(columns=constant_cols) # remove columns with constant values
    batch_features = batch_features.dropna(axis=1, how='any') # remove features with nan values
    for feature in batch_features.columns:
        cur_min = batch_features[feature].min()
        cur_max = batch_features[feature].max()
        batch_features[feature] = (batch_features[feature] - cur_min) / (cur_max-cur_min) # min max each feature (it's really not the best solution but it ~works)

    features = np.array(batch_features)
    labels = np.array(batch_labels.apply(lambda row: '_'.join(map(str, row)), axis=1))
    marker_centroids = create_markers_centroids_df(labels, features, exclude_DAPI=False)
    markers = marker_centroids['marker'].unique()
    reps = marker_centroids['rep'].unique()
    marker_centroids['cell_line_condition'] = marker_centroids['cell_line'] + '_' + marker_centroids['condition']
    cell_lines_conditions = marker_centroids['cell_line_condition'].unique()
    output_folder = "/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/cell_profiler/"
    between_cell_lines_sep_batch_rep = between_cell_lines_sep_rep_dist(marker_centroids, cell_lines_conditions, markers,
                                                                        distances_main_folder=output_folder, batch_name="CellProfiler")

def create_markers_centroids_df(all_labels, all_embedings_data, config_data, exclude_DAPI=True):  
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
    if config_data.EMBEDDINGS_LAYER in ['vqindhist1','vqindhist2']:
        marker_centroids[['marker','cell_line','condition','batch','rep']] = marker_centroids.label.str.split('_', expand=True)
    elif config_data.EMBEDDINGS_LAYER in ['vqvec2']:
        marker_centroids[['batch','cell_line','condition','rep','marker']] = marker_centroids.label.str.split('_', expand=True)
    else:
        logging.warning(f'EMBEDDINGS_LAYER {config_data.EMBEDDINGS_LAYER} is not supported!')
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
    return marker_centroids #, within_marker_similaities

def save_excel_with_sheet_name(filename, input_folders, df):
    with pd.ExcelWriter(filename) as writer:
        for input_folder in input_folders:
            batch_name = input_folder.split(os.sep)[-1].replace('_16bit_no_downsample','')
            df[df.batch==batch_name].to_excel(writer, sheet_name=batch_name, index=False)


def average_batches_distances(config_path_model, config_path_data):
    """ Read distances files that are separated into batches, calculate the average per batch, and save the results
    """
    # ------------------------------------------------------------------------------------------ 
    # Get configs of model (trained model) 
    config_model = load_config_file(config_path_model, 'model')
    
    # Get dataset configs (as to be used in the desired UMAP)
    config_data = load_config_file(config_path_data, 'data')
    
    experiment_type = get_if_exists(config_data, 'EXPERIMENT_TYPE', None)
    assert experiment_type is not None, "EXPERIMENT_TYPE can't be None"    
    embeddings_layer = get_if_exists(config_data, 'EMBEDDINGS_LAYER', 'vqvec2')
    input_folders = config_data.INPUT_FOLDERS
    distances_main_folder = os.path.join(config_model.MODEL_OUTPUT_FOLDER, 'distances', experiment_type, embeddings_layer)
    batches_names = [folder.split(os.sep)[-1].replace('_16bit_no_downsample','') for folder in input_folders]
    # -------------------------------------------------------------------------------------------------------------------
    # calc the average similarity of markers between reps (across batches)
    between_reps_similaities = pd.read_excel(os.path.join(distances_main_folder,'between_rep_similarities.xlsx'), 
                   sheet_name=batches_names)
    batches_dfs = [between_reps_similaities.get(batch_name) for batch_name in batches_names]
    all_batches = pd.concat(batches_dfs)   
    mean_reps_similarities = all_batches.groupby(['cell_line', 'condition', 'marker'])['rep_similiarity'].agg(mean_rep_similiarity='mean').reset_index()
    mean_reps_similarities.to_csv(os.path.join(distances_main_folder,'mean_reps_similarities.csv'), index=False)

    mean_between_cell_lines_conds_similarities = pd.read_excel(os.path.join(distances_main_folder,'mean_between_cell_lines_conds_similarities.xlsx'), 
                   sheet_name=batches_names)
    batches_dfs = [mean_between_cell_lines_conds_similarities.get(batch_name) for batch_name in batches_names]
    all_batches = pd.concat(batches_dfs)   
    batch_mean_between_cell_lines_conds_similarities = all_batches.groupby(['cell_line_condition'])[all_batches.columns.drop('batch').drop('cell_line_condition')].mean().reset_index()
    batch_mean_between_cell_lines_conds_similarities.to_csv(os.path.join(distances_main_folder,'batch_mean_between_cell_lines_conds_similarities.csv'), index=False)
    return None

def multiplex_embeddings(all_embedings_data, all_labels, dataset_conf):
    all_labels = np.asarray(all_labels).reshape(-1,)
    embeddings_df = __embeddings_to_df(all_embedings_data, all_labels, dataset_conf)
    all_embedings_data, all_labels, unique_groups = __get_multiplexed_embeddings(embeddings_df, random_state=dataset_conf.SEED)
    return all_embedings_data, all_labels

def calc_embeddings_distances_for_SM(config_model, config_data, distances_main_folder, embeddings_type, dist_func = cdist):
    # dist_func should Compute distance between each pair of the two collections of inputs, like scipy.spatial.distance.cdist
    # ------------------------------------------------------------------------------------------ 
    input_folders = config_data.INPUT_FOLDERS
    cell_lines = config_data.CELL_LINES
    # we need to load embeddings of each batch and each cell line at a time to avoid too much memory usage
    for input_folder in input_folders:
        sm_df = pd.DataFrame()
        for cell_line in cell_lines:
            batch_name = input_folder.split(os.sep)[-1].replace('_16bit_no_downsample','')
            cur_config_data = deepcopy(config_data)
            cur_config_data.INPUT_FOLDERS = [input_folder]
            cur_config_data.CELL_LINES = [cell_line]
            # Load saved embeddings
            #cur_marker_centroids_splitted, cur_marker_centroids, cur_within_marker_similaities = fetch_saved_embeddings(config_model = config_model, config_data = cur_config_data,
            #                                                                            embeddings_type=embeddings_type, EMBEDDINGS_NAMES=EMBEDDINGS_NAMES, exclude_DAPI=True)
            all_embedings_data, all_labels = fetch_saved_embeddings(config_model, cur_config_data, embeddings_type)
            if cur_config_data.EXPERIMENT_TYPE == 'neurons':
                all_labels_df = pd.DataFrame(all_labels, columns=['label'])
                if batch_name == 'batch6' and cell_line == 'SCNA':
                    bad_label = 'SCNA_Untreated_rep1'
                elif batch_name == 'batch9' and cell_line == 'FUSRevertant':
                    bad_label = 'FUSRevertant_Untreated_rep1'
                else:
                    bad_label = None
                if bad_label is not None:
                    logging.info(f'Filtering out {bad_label} for {batch_name}, {cell_line}')
                    filtered = all_labels_df[~all_labels_df.label.str.contains(bad_label)]
                    all_labels = np.array(filtered)
                    all_embedings_data = all_embedings_data[filtered.index]
            all_embedings_data, all_labels = multiplex_embeddings(all_embedings_data, all_labels, config_data)
            #cur_marker_centroids = create_markers_centroids_df(all_labels, all_embedings_data, EMBEDDINGS_NAMES, exclude_DAPI=True)
            cur_df = pd.DataFrame()
            cur_df['label'] = all_labels.reshape(-1)
            cur_df['sm_embeddings'] = list(all_embedings_data)
            del(all_embedings_data)

            logging.info(f'created sm embeddings for {batch_name}, {cell_line}')
            sm_df = pd.concat([sm_df, cur_df])
            del(cur_df)

        # ------------------------------------------------------------------------------------------
        # we want to calc similarites on all data from the current batch together 
        labels = np.unique(sm_df.label)
        similarities = pd.DataFrame(index=labels, columns=list(labels))
        for label_A, label_B in combinations(labels,2):
            # if label_A[:-5] == label_B[:-5]: # if the different is only in the rep, we don't want to calc the similarity
            #     continue
            sm_A = np.stack(sm_df[sm_df.label==label_A].sm_embeddings.values, axis=0)
            sm_B = np.stack(sm_df[sm_df.label==label_B].sm_embeddings.values, axis=0)
            cur_similarity = 1/(1+dist_func(sm_A, sm_B)).mean()
            similarities.loc[label_A, label_B] = cur_similarity
            similarities.loc[label_B, label_A] = cur_similarity
        similarities = similarities.reset_index(names='label')
        similarities.to_csv(os.path.join(distances_main_folder,f'between_cell_lines_conds_similarities_rep_{batch_name}.csv'), index=False)
        logging.info(f'Calculated similarities for {batch_name}')
        # ------------------------------------------------------------------------------------------          

    return None

def load_batch_embeddings(input_folder, cell_lines, config_data, config_model, embeddings_type, batch_name):
    marker_centroids, within_marker_similaities = pd.DataFrame(), pd.DataFrame()
    for cell_line in cell_lines:
        cur_config_data = deepcopy(config_data)
        cur_config_data.INPUT_FOLDERS = [input_folder]
        cur_config_data.CELL_LINES = [cell_line]
        # Load saved embeddings
        logging.info(f"[load_batch_embeddings] loading {cell_line} from {batch_name}")
        all_embedings_data, all_labels = fetch_saved_embeddings(config_model, cur_config_data, embeddings_type)
        cur_marker_centroids = create_markers_centroids_df(all_labels, all_embedings_data, config_data, exclude_DAPI=False)
        del(all_embedings_data)

        logging.info(f'Calculated marker_centroids_vectors for {batch_name}, {cell_line}')
        marker_centroids = pd.concat([marker_centroids, cur_marker_centroids])
        del(cur_marker_centroids)
        #within_marker_similaities = pd.concat([within_marker_similaities, cur_within_marker_similaities])

        # save current results
        #save_excel_with_sheet_name(os.path.join(distances_main_folder,f'within_marker_similaities_{batch_name}_{rep}.xlsx'), input_folders, within_marker_similaities)
    # within_marker_similaities.to_csv(os.path.join(distances_main_folder,f'within_marker_similaities_{batch_name}.csv'))
    # del(within_marker_similaities)
    return marker_centroids

def within_reps_dist(marker_centroids, markers, distances_main_folder, batch_name, dist_func = pairwise_distances):
    # Similarity between markers within reps, for a given batch-cellline-condition-rep
    # dist_func should compute pairwise distances between all rows of input matrix, such that the output is a distance matrix D such that D_{i, j} is the distance between 
    # the ith and jth vectors of the given matrix X, like in sklearn.metrics.pairwise_distances

    within_reps = pd.DataFrame(columns= ['batch','cell_line','condition','rep','marker'] + markers.tolist())
    for name, group in marker_centroids.groupby(['batch','cell_line','condition','rep'])[['marker', 'embeddings_centroid']]:

        group.set_index('marker', inplace=True)
        # Convert list of 1D arrays to a 2D array
        cur_markers = np.stack(group['embeddings_centroid'].values, axis=0)
        # Use pairwise_distances to calcualte similarities
        between_marker_similarities = pd.DataFrame(1/(1+dist_func(cur_markers, metric='euclidean', n_jobs=-1)), 
                                                columns=group.index.values, 
                                                index=group.index).reset_index()
        
        # combine marker similiarites in one df
        for i, column_name in enumerate(['batch','cell_line','condition','rep']):
            between_marker_similarities[column_name] = name[i]
        within_reps = pd.concat([within_reps, between_marker_similarities])
    # save combined df
    #save_excel_with_sheet_name(os.path.join(distances_main_folder,'within_reps_similarities.xlsx'), input_folders, within_reps)
    within_reps.to_csv(os.path.join(distances_main_folder,f'within_reps_similarities_{batch_name}.csv'), index=False)
    del(within_reps)
    logging.info(f'calculated between marker distances for {batch_name}')
    return None

def between_reps_dist(marker_centroids, reps, distances_main_folder, batch_name, EMBEDDINGS_NAMES, dist_func = pairwise_distances):
    # How similiar are the reps, given batch-cellline-condition?
    # dist_func should compute pairwise distances between all rows of input matrix, such that the output is a distance matrix D such that D_{i, j} is the distance between 
    # the ith and jth vectors of the given matrix X, like in sklearn.metrics.pairwise_distances

    if len(reps)<=1:
        logging.info(f"Skipping comparison of reps since reps: {reps}")
        return None

    elif len(reps)>1:
        between_reps = pd.DataFrame(columns=['batch','cell_line','condition','marker','rep_similiarity'])
        reps_centroids = pd.DataFrame(columns=['batch','cell_line','condition','marker'] + EMBEDDINGS_NAMES)
        marker_reps = marker_centroids.groupby(['batch','cell_line','condition','marker'])[['embeddings_centroid']]
        for name, group in marker_reps: 
            # calc cur batch_cell_line_condition_marker centroid 
            cur_centroid = np.median(np.stack(group['embeddings_centroid']), axis=0)
            reps_centroids.loc[len(reps_centroids)] = [name[0], name[1], name[2],name[3]] +  cur_centroid.tolist()
            # check if current marker has to reps, else we don't want to calc distances
            if group.shape[0] < 2:
                logging.info(f"Skipping comparison of reps for {name} since reps: {group.shape[0]}")
                between_reps.loc[len(between_reps)] = [name[0], name[1], name[2], name[3], None]
                continue
            # Pairwise ditances between reps centroids
            marker_similarity_between_reps = 1 / (1 + dist_func(np.stack(group['embeddings_centroid']), metric="euclidean"))[0,1]
            # combine rep similiarites in one df
            between_reps.loc[len(between_reps)] = [name[0], name[1], name[2], name[3], marker_similarity_between_reps]
        # save combined df
        #save_excel_with_sheet_name(os.path.join(distances_main_folder,'between_rep_similarities.xlsx'), input_folders, between_reps)
        between_reps.to_csv(os.path.join(distances_main_folder,f'between_rep_similarities_{batch_name}.csv'), index=False)
        # save reps centroids
        #save_excel_with_sheet_name(os.path.join(distances_main_folder,'reps_centroids.xlsx'), input_folders, reps_centroids)
        reps_centroids.set_index(['batch', 'cell_line', 'condition', 'marker'], inplace=True)
        logging.info(f'calculated between reps distances for {batch_name}')  
        return reps_centroids
    
def between_cell_lines_dist(marker_centroids, reps_centroids, cell_lines_conditions, markers, distances_main_folder, batch_name, EMBEDDINGS_NAMES, dist_func = pairwise_distances):
    # given batch, how similar are the cell lines? 
    # dist_func should compute pairwise distances between all rows of input matrix, such that the output is a distance matrix D such that D_{i, j} is the distance between 
    # the ith and jth vectors of the given matrix X, like in sklearn.metrics.pairwise_distances

    if len(cell_lines_conditions) <= 1:
        logging.info(f"Skipping comparison of cell lines since cell lines: {cell_lines_conditions}")
        return None

    elif len(cell_lines_conditions) > 1:
        between_cell_lines = pd.DataFrame(columns=['batch','marker','cell_line_condition'] + cell_lines_conditions.tolist())
        for name, group in marker_centroids.groupby(['batch'])[['batch', 'cell_line_condition', 'marker']]:
            def fetch_centroids(x, centroids_data):
                x.drop_duplicates(inplace=True)
                batch, cell_line_condition, marker = x['batch'], x['cell_line_condition'], x['marker']
                cell_line, condition = cell_line_condition.str.split("_").values[0]
                return np.stack(centroids_data.loc[batch, cell_line, condition, marker].values)
            group_cell_line_condition_centroids = group.groupby(['marker', 'cell_line_condition']).apply(lambda x: fetch_centroids(x, reps_centroids)).reset_index(name='embeddings_centroid')

            for marker in markers:
                cur_marker = group_cell_line_condition_centroids[group_cell_line_condition_centroids['marker']==marker]
                if cur_marker.shape[0] < 2:
                    logging.info(f"Skipping comparison of cell_line_conds for {marker} in {name} since cell_line_conds: {cur_marker.shape[0]}")
                    continue
                x = np.stack(cur_marker['embeddings_centroid'].values, axis=0)[:,0,:]
                cell_lines_conds_similarities = pd.DataFrame(1/(1+dist_func(x, metric='euclidean', n_jobs=-1)), 
                                        columns=cur_marker.cell_line_condition.values, 
                                        index=cur_marker.cell_line_condition).reset_index()

                # combine marker similiarites in one df
                cell_lines_conds_similarities_for_df = cell_lines_conds_similarities
                cell_lines_conds_similarities_for_df['batch'] = name
                cell_lines_conds_similarities_for_df['marker'] = marker
                between_cell_lines = pd.concat([between_cell_lines, cell_lines_conds_similarities_for_df])
        #save_excel_with_sheet_name(os.path.join(distances_main_folder,'between_cell_lines_conds_similarities.xlsx'), input_folders, between_cell_lines)
        between_cell_lines.to_csv(os.path.join(distances_main_folder,f'between_cell_lines_conds_similarities_{batch_name}.csv'), index=False)
        mean_between_cell_lines = between_cell_lines.groupby(['batch', 'cell_line_condition'])[cell_lines_conditions.tolist()].mean().reset_index()
        #save_excel_with_sheet_name(os.path.join(distances_main_folder,'mean_between_cell_lines_conds_similarities.xlsx'), input_folders, mean_between_cell_lines)
        mean_between_cell_lines.to_csv(os.path.join(distances_main_folder,f'mean_between_cell_lines_conds_similarities_{batch_name}.csv'), index=False)
        logging.info(f'calculated between cell_lines_conds distances for {batch_name}')
        return None

def between_cell_lines_sep_rep_dist(marker_centroids, distances_main_folder, 
                                    dist_func = pairwise_distances, compare_identical_reps=True, dist_metric='euclidean'):
    # given batch, how similar are the cell lines? also treating different reps as batches
    # dist_func should compute pairwise distances between all rows of input matrix, such that the output is a distance matrix D such that D_{i, j} is the distance between 
    # the ith and jth vectors of the given matrix X, like in sklearn.metrics.pairwise_distances
    cell_lines_conditions = marker_centroids.cell_line_condition.unique()
    markers = marker_centroids.marker.unique()
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
                    cell_lines_conds_similarities = pd.DataFrame(dist_func(x, metric=dist_metric, n_jobs=-1), 
                                                                columns=cur_marker.cell_line_condition.values, 
                                                                index=cur_marker.cell_line_condition).reset_index()
                    num_rows, _ = cell_lines_conds_similarities.shape

                    # # Nullify the diagonal elements
                    # for i in range(num_rows):
                    #     cell_lines_conds_similarities.iat[i, i+1] = np.nan

                    # combine marker similiarites in one df
                    cell_lines_conds_similarities_for_df = cell_lines_conds_similarities
                    cell_lines_conds_similarities_for_df['batch'] = name[0]
                    cell_lines_conds_similarities_for_df['rep'] = name[1]
                    cell_lines_conds_similarities_for_df['marker'] = marker
                    between_cell_lines_sep_batch_rep = pd.concat([between_cell_lines_sep_batch_rep, cell_lines_conds_similarities_for_df])
            between_cell_lines_sep_batch_rep.to_csv(os.path.join(distances_main_folder,f'between_cell_lines_conds_similarities_rep.csv'), index=False)
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
                dists = pairwise_distances(x, metric=dist_metric, n_jobs=-1)
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
            between_cell_lines.to_csv(os.path.join(distances_main_folder,f'between_cell_lines_conds_distances.csv'), index=False)

        #save_excel_with_sheet_name(os.path.join(distances_main_folder,'between_cell_lines_conds_similarities_rep_rep.xlsx'), input_folders, between_cell_lines_sep_batch_rep)
        # logging.info(f'calculated between cell_lines_conds distances separated into reps for {batch_name}')

def calc_embeddings_distances(config_model, config_data, distances_main_folder, embeddings_type, compare_identical_reps=True):
    """Main function to calculate embeddings distances
    """
    # ------------------------------------------------------------------------------------------ 
    train_batches = get_if_exists(config_data, 'TRAIN_BATCHES', None)
    assert train_batches is not None, "train_batches can't be None"    

    input_folders = config_data.INPUT_FOLDERS
    cell_lines = config_data.CELL_LINES
    # ------------------------------------------------------------------------------------------ 
    # we need to load embeddings and calc dists of each batch at a time to avoid too much memory usage
    orig_embeddings_type = embeddings_type
    all_batches_marker_centroids = pd.DataFrame()
    for input_folder in input_folders:
        batch_name = input_folder.split(os.sep)[-1].replace('_16bit_no_downsample','')
        if batch_name in train_batches:
            embeddings_type = 'testset'
            logging.info(f'setting embeddings_type:{embeddings_type}')
        marker_centroids = load_batch_embeddings(input_folder, cell_lines, config_data, config_model, embeddings_type, batch_name)
        # ------------------------------------------------------------------------------------------  
        marker_centroids['cell_line_condition'] = marker_centroids['cell_line'] + '_' + marker_centroids['condition']
        # Similarity between markers within reps, for a given batch-cellline-condition-rep
        # within_reps_dist(marker_centroids, markers, distances_main_folder, batch_name)
        # ------------------------------------------------------------------------------------------ 
        # How similiar are the reps, given batch-cellline-condition?
        #reps_centroids = between_reps_dist(marker_centroids, reps, distances_main_folder, batch_name, EMBEDDINGS_NAMES)
        # ------------------------------------------------------------------------------------------   
        ## given batch, how similar are the cell lines? 
        #between_cell_lines_dist(marker_centroids, reps_centroids, cell_lines_conditions, markers, distances_main_folder, batch_name, EMBEDDINGS_NAMES)
        if all_batches_marker_centroids.shape[0]==0:
            all_batches_marker_centroids = marker_centroids
        else:
            all_batches_marker_centroids = pd.concat([all_batches_marker_centroids, marker_centroids], ignore_index=True)
        embeddings_type = orig_embeddings_type
    between_cell_lines_sep_rep_dist(all_batches_marker_centroids, distances_main_folder, compare_identical_reps=compare_identical_reps)

    return None

def unite_batches(distances_main_folder, input_folders, files = ['within_reps_similarities','between_rep_similarities',
                                                                'between_cell_lines_conds_similarities','mean_between_cell_lines_conds_similarities',
                                                                'between_cell_lines_conds_similarities_rep']
                                                                ):
    """ Read batch similarites files and unite
    """
    # ------------------------------------------------------------------------------------------ 
    batches_names = [folder.split(os.sep)[-1].replace('_16bit_no_downsample','') for folder in input_folders]
    for file in files:
        # old method
        df = pd.DataFrame()
        for batch in batches_names:
            cur_df = pd.read_csv(os.path.join(distances_main_folder, f'{file}_{batch}.csv'))
            if 'batch' not in cur_df.columns:
                cur_df['batch'] = batch
            df = pd.concat([df, cur_df])
        save_excel_with_sheet_name(os.path.join(distances_main_folder,f'{file}.xlsx'), input_folders, df)

        new_df = pd.DataFrame()
        for batch in batches_names:
            cur_df = pd.read_csv(os.path.join(distances_main_folder, f'{file}_{batch}.csv'))
            if 'batch' not in cur_df.columns:
                cur_df['batch'] = batch
            new_df = pd.concat([new_df, cur_df])
        new_df.to_csv(os.path.join(distances_main_folder,f'{file}_new.csv'), index=False)
    return None


def plot_distances_plot(distances_main_folder, convert_markers_names_to_organelles=False, compare_identical_reps=True):
    if compare_identical_reps:
        dists = pd.read_csv(os.path.join(distances_main_folder,'between_cell_lines_conds_similarities_rep.csv'))
        cell_line_conds = np.unique(dists.cell_line_condition)
    else:
        dists = pd.read_csv(os.path.join(distances_main_folder,'between_cell_lines_conds_distances.csv'))
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
        dists_cond, dists_cond_order = get_dists_between_baseline_and_target(dists, base, target, compare_identical_reps=compare_identical_reps)
        plot_distances_boxplot(dists_cond, dists_cond_order,
                    base, target,
                    transpose=False, figsize = (20,5), savefolder=distances_main_folder, compare_identical_reps=compare_identical_reps)


def get_dists_between_baseline_and_target(dists, baseline_label, target_label, scale=True, compare_identical_reps=True):
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
        # cannot do scaling per batch/rep because the distances are not calculcated separatly for each batch/rep!!

    logging.info(f"dists_filtered.shape: {dists_filtered.shape}")
    return dists_filtered, dists_filtered_order

def plot_distances_boxplot(dists, dists_order, baseline_label, target_label,
                           title=None, savefolder=None, ax=None, transpose=False, figsize=(6,4), fontsize=13, title_fontsize=20,
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
        savepath = os.path.join(folderpath, f'{baseline_label}_VS_{target_label}')
        if not compare_identical_reps:
            savepath += '_all_reps_batches'
        plt.savefig(f"{savepath}.png", dpi=300)
        # plt.savefig(f"{savepath}.eps", dpi=300, format='eps')
        
    
    return ax

if __name__ == "__main__":
    calc_cellprofiler_distances()
    # if len(sys.argv) != 4:
    #     raise ValueError("Invalid config path. Must supply model config and data config and embedding type.")
    # try:
    #     calc_embeddings_distances(config_path_model= sys.argv[1], config_path_data=sys.argv[2], embeddings_type=sys.argv[3])  
    #     average_batches_distances(config_path_model= sys.argv[1], config_path_data=sys.argv[2])
    # except Exception as e:
    #     logging.exception(str(e))
    #     raise e
    # logging.info("Done!")
        
# ./bash_commands/run_py.sh ./src/common/lib/embeddings_distances_utils -m 40000 -a ./src/models/neuroself/configs/model_config/TLNeuroselfB78NoDSModelConfig ./src/datasets/configs/embeddings_data_config/EmbeddingsB9DatasetConfig all        
