import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np
import pandas as pd
import logging

from multiprocessing.pool import ThreadPool
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics.pairwise import pairwise_distances

#from src.common.lib.image_sampling_utils import find_marker_folders
#from src.common.lib.utils import load_config_file, init_logging
#from src.common.lib.model import Model
#from src.common.lib.data_loader import get_dataloader
#from src.datasets.dataset_spd import DatasetSPD
from src.common.lib.embeddings_utils import load_embeddings

###############################################################
# Utils for calculating distances between  labels, based on the full latent space (Embeddings) 
# (run from MOmaps/src/runables/calc_distances.py) # TODO: 
###############################################################


###############################################################
# Distance metrics sklearn: https://scikit-learn.org/stable/modules/metrics.html#metrics
DISTANCE_METRICS = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine']
AGG_FUNCTIONS = [np.mean, np.median]
###############################################################

def fetch_saved_embeddings(exclude_DAPI=True):
    """Couple embedding vector with it's corresponding label (label == batch-cellline-condition-rep-marker)

    Args:
        exclude_DAPI (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: pd.DataFrame
    """
    all_embedings_data, all_labels = load_embeddings(config_path_model='./src/models/neuroself/configs/model_config/TLNeuroselfB78ModelConfig', 
                                                     config_path_data='./src/datasets/configs/embeddings_data_config/EmbeddingsB9DatasetConfig',
                                                     embeddings_type='all')
        
    all_embedings_data = all_embedings_data.reshape(all_embedings_data.shape[0], -1)
    logging.info(f"[load_embeddings] {all_embedings_data.shape}, {all_labels.shape}")    
    print(f"[load_embeddings] {all_embedings_data.shape}, {all_labels.shape}")     #TODO: delete
    
    
    df = create_df(all_labels, all_embedings_data)
    
    # Exclude embeddings of DAPI marker  
    id exclude_DAPI:
        df = df[df['marker']!='DAPI']
    
    return df
        
def create_df(all_labels, all_embedings_data):
    """Create a pd.DataFrame of embedddings and experimental settings 
    columns are ['batch','cell_line','condition','rep','marker', 'embeddings] 
    where the ndarray embeddings are saved as list of lists (in a single column)

    Args:
        all_labels (np.array): array of strings, each row is in a format of "batch_cell_line_condition_rep_marker"
        all_embedings_data (np.ndarray): latent featuers (each row has 9216 columns)
    """
    
    assert all_labels.shape[0]==all_embedings_data.shape[0]
    
    # Split to experimental settings to different columns 
    df = pd.DataFrame(data=all_labels, columns=['label'])
    df['label'] = df.label.str.replace('_16bit_no_downsample','')  #TODO: delete workaround since batch folder have "_"
    df[['batch','cell_line','condition','rep','marker']] = df.label.str.split('_', expand=True)
    df.drop(columns=['label'], inplace=True)
    
    
    # Conversion of embeddings (a numpy matrix) into a single column 
    df['embeddings'] = pd.Series(all_embedings_data.tolist())
    
    return df

def calc_embeddings_centroids(df, groups_by, agg_func):
    """_summary_

    Args:
        df (ps.DataFrame): _description_
        groups_by (list): _description_
        agg_func (numpy function): np.mean / np.median / etc.
    """
    groups_series = df.groupby(groups_by)['embeddings'].apply(lambda x: agg_func([*x], axis=0))
    group_representatives = pd.DataFrame(groups_series).reset_index()
    group_representatives.rename(columns={"embeddings": "embeddings_centroid"}, inplace=True)
    
    return group_representatives



#def main(config_path_model, config_path_data, embeddings_type): #TODO: sys.args?
def main():
    
    
    # ------------------------------------------------------------------------------------------ 
    # Load saved embeddings
    df = fetch_saved_embeddings(exclude_DAPI=True)
    
    # ------------------------------------------------------------------------------------------ 
    # Aggregate embeddings for all tiles under same label -> get one vector per label 
    marker_centroids_vectors = calc_embeddings_centroids(df,
                                                         groups_by=['batch','cell_line','condition','rep','marker'], 
                                                         agg_func = np.median)

    print("\n\nXX", marker_centroids_vectors, "\n\n")
    
    # ------------------------------------------------------------------------------------------  
    # Similarity between markers, for a given batch-cellline-condition-rep 
    for name, group in marker_centroids_vectors.groupby(['batch','cell_line','condition','rep'])[['marker', 'embeddings_centroid']]:
        
        group.set_index('marker', inplace=True)
        # Convert list of 1D arrays to a 2D array
        x = np.stack(group['embeddings_centroid'].values, axis=0)
        # Use pairwise_distances to calcualte similarities
        marker_similarities = pd.DataFrame(1/(1+pairwise_distances(x, metric='euclidean', n_jobs=-1)), 
                                        columns=group.index.values, 
                                        index=group.index).reset_index()
        
        print(name, ":\n", marker_similarities)
        logging.info(f"\n[marker_similarities] {name}, {marker_similarities}")    
    
    # ------------------------------------------------------------------------------------------ 
    # How similiar are the reps, given batch-cellline-condition?  -> take 50 marker vectors, calc distance between two instances of the same marker
    markers = marker_centroids_vectors['marker'].unique()
    for marker in markers:
        tmp = marker_centroids_vectors[marker_centroids_vectors['marker']==marker]
        marker_reps = tmp.groupby(['batch','cell_line','condition','rep'])[['rep', 'embeddings_centroid']]
        
        # Collect centroids of each rep 
        marker_reps_centroid = []
        for name, group in marker_reps: 
            x = np.stack(group['embeddings_centroid'].values, axis=0)
            marker_reps_centroid.append(x[0])
        
        # Pairwise ditances between reps centroids
        marker_similarity_between_reps = 1 / (1 + pdist(np.stack(marker_reps_centroid), metric="euclidean"))
        print(f"\nmarker_similarity_between_reps {marker}: {marker_similarity_between_reps}")
        logging.info(f"\nmarker_similarity_between_reps {marker}: {marker_similarity_between_reps}}")
        
        # ------------------------------------------------------------------------------------------ 
        
        # Nancy: stopped here...
        ## given Batch-cellline, which markers create most sepration between conditions? -> for each cond, for every marker, average its two reps -> got 25 vectors for each cond. then calc distances between those 25 vectors and any 25 vectors in other conditions.
        
        # TBD: need to store intermidiate results and distances 
        
        
        
        ## given batch, how similar are the cell lines? for each cell_line and marker, we need to create a vector
        # for every pair of cell lines, take all 'marker' vectors, and calc the distances -> resulting in 25 distances


    
        # # marker_reps.apply(lambda x: pdist(np.array(list(zip(x.x, x.y)))).mean())
    
    
    return None
    
if __name__ == "__main__":
    
    try:
        # TODO: get configs as args... 
        #if len(sys.argv) != 3:
            #raise ValueError("Invalid config path. Must supply model config and data config.")
        #config_path_model, config_path_data, embeddings_type = sys.argv[0], sys.argv[1], sys.argv[3]
        
        main()
        
        
        # with ThreadPool(4) as pool:
        #     # prepare args
        #     args = [(a,b) for a,b in zip(data1,data2)]
        #     # calculate distances on chunks of vectors
        #     distances = pool.starmap(dis_func_on_x,y, args)
        #     # cdist(X, Y, metric='euclidean')
        #     print(distances.shape)
        
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done!")
        
