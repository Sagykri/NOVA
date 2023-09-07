import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np
import pandas as pd
import logging
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics.pairwise import pairwise_distances

#from src.common.lib.image_sampling_utils import find_marker_folders
from src.common.lib.utils import load_config_file, init_logging, get_if_exists
#from src.common.lib.model import Model
#from src.common.lib.data_loader import get_dataloader
#from src.datasets.dataset_spd import DatasetSPD
from src.common.lib.embeddings_utils import load_embeddings
import datetime
###############################################################
# Utils for calculating distances between  labels, based on the full latent space (Embeddings) 
# (run from MOmaps/src/runables/calc_distances.py) # TODO: 
###############################################################


###############################################################
# Distance metrics sklearn: https://scikit-learn.org/stable/modules/metrics.html#metrics
DISTANCE_METRICS = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine']
AGG_FUNCTIONS = [np.mean, np.median]
###############################################################

def fetch_saved_embeddings(config_model, config_data, embeddings_type, EMBEDDINGS_NAMES, exclude_DAPI=True):
    """Couple embedding vector with it's corresponding label (label == batch-cellline-condition-rep-marker)

    Args:
        exclude_DAPI (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: pd.DataFrame
    """
    all_embedings_data, all_labels = load_embeddings(config_model=config_model, 
                                                     config_data= config_data,
                                                     embeddings_type=embeddings_type)
        
    all_embedings_data = all_embedings_data.reshape(all_embedings_data.shape[0], -1)
    logging.info(f"[load_embeddings] {all_embedings_data.shape}, {all_labels.shape}")    
    
    
    marker_centroids_splitted, marker_centroids, within_marker_similaities = create_markers_centroids_df(all_labels, all_embedings_data, EMBEDDINGS_NAMES)
    del(all_embedings_data)
    # Exclude embeddings of DAPI marker  
    if exclude_DAPI:
        marker_centroids = marker_centroids[marker_centroids['marker']!='DAPI']
        within_marker_similaities = within_marker_similaities[within_marker_similaities['marker']!='DAPI']
        marker_centroids_splitted = marker_centroids_splitted[marker_centroids_splitted['marker']!='DAPI']
    return marker_centroids_splitted, marker_centroids, within_marker_similaities
        
def create_markers_centroids_df(all_labels, all_embedings_data, EMBEDDINGS_NAMES):
    """Create a pd.DataFrame of centroids embedddings and experimental settings 
    columns are ['batch','cell_line','condition','rep','marker', 'embeddings_centroid'] 

    Args:
        all_labels (np.array): array of strings, each row is in a format of "batch_cell_line_condition_rep_marker"
        all_embedings_data (np.ndarray): latent featuers (each row has 9216 columns)
    """
    
    assert all_labels.shape[0]==all_embedings_data.shape[0]    
    labels_df = pd.DataFrame(data=all_labels, columns=['label'])
    labels_df['label'] = labels_df.label.str.replace('_16bit_no_downsample','')  #TODO: delete workaround since batch folder have "_"
    
    # Calculate embeddings centroids (a numpy matrix) + within marker similarities
    marker_centroids = pd.DataFrame()
    within_marker_similaities = pd.DataFrame()
    centroids = []
    labels = []
    similarities = []
    for label, label_df in labels_df.groupby(['label']): # use indexes of labels to find corresponding embeddings
        logging.info(f'[create_markers_centroids_df] adding label {label[0]}')
        cur_embeddings = all_embedings_data[label_df.index]
        # calc current label centroid
        cur_centroid = np.median(cur_embeddings, axis=0)
        centroids.append(cur_centroid.tolist())
        labels.append(label[0])
        # calc current label mean similarity
        mean_marker_similarity =  (1/(1+pdist(cur_embeddings, metric='euclidean'))).mean() #using pdist since we don't want a distance matrix here, but just all the pairwise distances (without repeats & without self distances)
        similarities.append(mean_marker_similarity)

    marker_centroids['label'] = labels
    marker_centroids[['batch','cell_line','condition','rep','marker']] = marker_centroids.label.str.split('_', expand=True)
    marker_centroids['embeddings_centroid'] = centroids
    marker_centroids.drop(columns=['label'], inplace=True)
    logging.info(f'[create_markers_centroids_df] created df with shape {marker_centroids.shape}')

    split_embeddings = marker_centroids['embeddings_centroid'].apply(pd.Series)
    split_embeddings.columns = EMBEDDINGS_NAMES
    split_embeddings = pd.concat([marker_centroids[['batch','cell_line','condition','rep','marker']], split_embeddings], axis=1)

    within_marker_similaities['label'] = labels
    within_marker_similaities[['batch','cell_line','condition','rep','marker']] = within_marker_similaities.label.str.split('_', expand=True)
    within_marker_similaities.drop(columns=['label'], inplace=True)
    within_marker_similaities['marker_similarity'] = similarities
    
    return split_embeddings, marker_centroids, within_marker_similaities


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

def calc_embeddings_distances(config_path_model, config_path_data, embeddings_type):
    """Main function to calculate embeddings distances
    """
    # ------------------------------------------------------------------------------------------ 
    # Get configs of model (trained model) 
    config_model = load_config_file(config_path_model, 'model')
    logging.info('[Calc Embeddings Distances]')
    # Get dataset configs (as to be used in the desired UMAP)
    config_data = load_config_file(config_path_data, 'data')
    
    experiment_type = get_if_exists(config_data, 'EXPERIMENT_TYPE', None)
    assert experiment_type is not None, "EXPERIMENT_TYPE can't be None"    
    embeddings_layer = get_if_exists(config_data, 'EMBEDDINGS_LAYER', 'vqvec2')
    EMBEDDINGS_SHAPE = 9216 if embeddings_layer == 'vqvec2' else 40000
    EMBEDDINGS_NAMES = [f'embeddings_{i}' for i in range(EMBEDDINGS_SHAPE)]
    input_folders = config_data.INPUT_FOLDERS
    distances_main_folder = os.path.join(config_model.MODEL_OUTPUT_FOLDER, 'distances', experiment_type, embeddings_layer)
    os.makedirs(distances_main_folder, exist_ok=True)
    logging.info(f'Saving results in {distances_main_folder}')
    # ------------------------------------------------------------------------------------------ 
    marker_centroids, marker_centroids_splitted, within_marker_similaities = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # we need to load embeddings of each batch at a time to avoid too much memory usage
    for input_folder in input_folders:
        batch_name = input_folder.split(os.sep)[-1].replace('_16bit_no_downsample','')
        cur_config_data = deepcopy(config_data)
        cur_config_data.INPUT_FOLDERS = [input_folder]
        # Load saved embeddings
        cur_marker_centroids_splitted, cur_marker_centroids, cur_within_marker_similaities = fetch_saved_embeddings(config_model = config_model, config_data = cur_config_data,
                                                                                     embeddings_type=embeddings_type, EMBEDDINGS_NAMES=EMBEDDINGS_NAMES, exclude_DAPI=True)
        logging.info(f'Calculated marker_centroids_vectors for {batch_name}')
        marker_centroids_splitted = pd.concat([marker_centroids_splitted, cur_marker_centroids_splitted])
        marker_centroids = pd.concat([marker_centroids, cur_marker_centroids])
        within_marker_similaities = pd.concat([within_marker_similaities, cur_within_marker_similaities])
    
    # save current results
    save_excel_with_sheet_name(os.path.join(distances_main_folder,'marker_centroids.xlsx'), input_folders, marker_centroids_splitted)
    del(marker_centroids_splitted)
    save_excel_with_sheet_name(os.path.join(distances_main_folder,'within_marker_similaities.xlsx'), input_folders, within_marker_similaities)
    del(within_marker_similaities)
    # ------------------------------------------------------------------------------------------  
    markers = marker_centroids['marker'].unique()
    reps = marker_centroids['rep'].unique()
    marker_centroids['cell_line_condition'] = marker_centroids['cell_line'] + '_' + marker_centroids['condition']
    cell_lines_conditions = marker_centroids['cell_line_condition'].unique()
    
    # Similarity between markers within reps, for a given batch-cellline-condition-rep
    within_reps = pd.DataFrame(columns= ['batch','cell_line','condition','rep','marker'] + markers.tolist())
    for name, group in marker_centroids.groupby(['batch','cell_line','condition','rep'])[['marker', 'embeddings_centroid']]:
        
        group.set_index('marker', inplace=True)
        # Convert list of 1D arrays to a 2D array
        cur_markers = np.stack(group['embeddings_centroid'].values, axis=0)
        # Use pairwise_distances to calcualte similarities
        between_marker_similarities = pd.DataFrame(1/(1+pairwise_distances(cur_markers, metric='euclidean', n_jobs=-1)), 
                                        columns=group.index.values, 
                                        index=group.index).reset_index()

        # combine marker similiarites in one df
        for i, column_name in enumerate(['batch','cell_line','condition','rep']):
            between_marker_similarities[column_name] = name[i]
        within_reps = pd.concat([within_reps, between_marker_similarities])

    # save combined df
    save_excel_with_sheet_name(os.path.join(distances_main_folder,'within_reps_similarities.xlsx'), input_folders, within_reps)
    del(within_reps)
    logging.info('calculated between marker distances')
    # ------------------------------------------------------------------------------------------ 
    # How similiar are the reps, given batch-cellline-condition?  -> take 50 marker vectors, calc distance between two instances of the same marker
    if len(reps)<=1:
        logging.info(f"Skipping comparison of reps since reps: {reps}")
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
           marker_similarity_between_reps = 1 / (1 + pairwise_distances(np.stack(group['embeddings_centroid']), metric="euclidean"))[0,1]
           # combine rep similiarites in one df
           between_reps.loc[len(between_reps)] = [name[0], name[1], name[2], name[3], marker_similarity_between_reps]
        # save combined df
        save_excel_with_sheet_name(os.path.join(distances_main_folder,'between_rep_similarities.xlsx'), input_folders, between_reps)
        # save reps centroids
        save_excel_with_sheet_name(os.path.join(distances_main_folder,'reps_centroids.xlsx'), input_folders, reps_centroids)
        reps_centroids.set_index(['batch', 'cell_line', 'condition', 'marker'], inplace=True)
        logging.info('calculated between reps distances')        
    # ------------------------------------------------------------------------------------------   
    ## given batch, how similar are the cell lines? for each cell_line and marker, we need to create a vector
    # for every pair of cell lines, take all 'marker' vectors, and calc the distances -> resulting in 25 distances
    if len(cell_lines_conditions) <= 1:
        logging.info(f"Skipping comparison of cell lines since cell lines: {cell_lines_conditions}")

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
                    cell_lines_conds_similarities = pd.DataFrame(1/(1+pairwise_distances(x, metric='euclidean', n_jobs=-1)), 
                                            columns=cur_marker.cell_line_condition.values, 
                                            index=cur_marker.cell_line_condition).reset_index()
                                        
                    # combine marker similiarites in one df
                    cell_lines_conds_similarities_for_df = cell_lines_conds_similarities
                    cell_lines_conds_similarities_for_df['batch'] = name
                    cell_lines_conds_similarities_for_df['marker'] = marker
                    between_cell_lines = pd.concat([between_cell_lines, cell_lines_conds_similarities_for_df])
        save_excel_with_sheet_name(os.path.join(distances_main_folder,'between_cell_lines_conds_similarities.xlsx'), input_folders, between_cell_lines)
        mean_between_cell_lines = between_cell_lines.groupby(['batch', 'cell_line_condition'])[cell_lines_conditions.tolist()].mean().reset_index()
        save_excel_with_sheet_name(os.path.join(distances_main_folder,'mean_between_cell_lines_conds_similarities.xlsx'), input_folders, mean_between_cell_lines)

        logging.info('calculated between cell_lines_conds distances')
    
    return None
    
if __name__ == "__main__":
    
    if len(sys.argv) != 4:
        raise ValueError("Invalid config path. Must supply model config and data config and embedding type.")
    try:
        calc_embeddings_distances(config_path_model= sys.argv[1], config_path_data=sys.argv[2], embeddings_type=sys.argv[3])  
        average_batches_distances(config_path_model= sys.argv[1], config_path_data=sys.argv[2])
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done!")
        