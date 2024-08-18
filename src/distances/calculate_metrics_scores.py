import datetime
import itertools
import logging
import os
import subprocess
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

from src.common.lib.utils import load_config_file, get_if_exists, init_logging
from src.runables.calculate_embeddings_distances_vit import __load_vit_features
from sklearn.metrics import silhouette_score, adjusted_rand_score
from src.common.lib.metrics import cluster_without_outliers
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

def __handle_log(model_output_folder):
    # logs
    jobid = os.getenv('LSB_JOBID')
    jobname = os.getenv('LSB_JOBNAME')
    username = 'UnknownUser'
    if jobid:
        # Run the bjobs command to get job details
        result = subprocess.run(['bjobs', '-o', 'user', jobid], capture_output=True, text=True, check=True)
        # Extract the username from the output
        username = result.stdout.replace('USER', '').strip()
            
    __now = datetime.datetime.now()
    logs_folder = os.path.join(model_output_folder, "logs")
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder, exist_ok=True)
    log_file_path = os.path.join(logs_folder, __now.strftime("%d%m%y_%H%M%S_%f") + f'_{jobid}_{username}_{jobname}.log')
    init_logging(log_file_path)
    ##

def calculate_metrics_scores(config_path_data, model_output_folder):
    __handle_log(model_output_folder)

    config_data = load_config_file(config_path_data, 'data')
    
    output_folder_path = os.path.join(model_output_folder, 'figures', config_data.EXPERIMENT_TYPE, 'distances')
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)

    train_batches = get_if_exists(config_data, 'TRAIN_BATCHES', None)
    assert train_batches is not None, "train_batches can't be None for distances" 

    embeddings, labels = __load_vit_features(model_output_folder, config_data, train_batches)

    baseline_cell_line_cond = get_if_exists(config_data, 'BASELINE_CELL_LINE_CONDITION', None)
    assert baseline_cell_line_cond is not None, "BASELINE_CELL_LINE_CONDITION is None. You have to specify the baseline to calculate the silhouette agains (for example: WT_Untreated or TDP43_Untreated)"
    
    conditions = np.unique(['_'.join(l.split("_")[1:3]) for l in labels]) # Extract the list of cell_line_conds
    conditions = np.delete(conditions, np.where(conditions == baseline_cell_line_cond)[0]) # Remove the WT_Untreated
    markers = np.unique([label.split('_')[0] for label in np.unique(labels)])
    batches = [input_folder.split(os.sep)[-1] for input_folder in config_data.INPUT_FOLDERS] 
    
    logging.info(f"model_output_folder: {model_output_folder}, config_path_data: {config_path_data}")
    logging.info(f"config_data: {config_data.__dict__}")
    logging.info(f"Conditions: {conditions}, markers: {markers}, batches: {batches}, baseline_cell_line_cond: {baseline_cell_line_cond}")

    scores = pd.DataFrame(columns=['silhouette', 'ARI', 'ARI_constrained', 'marker', 'condition', 'repA', 'repB', 'batch'])
    for batch in batches:
        logging.info(f"batch: {batch}")
        for marker in markers:   
            np.random.seed(1)
            logging.info(f"marker: {marker}")

            logging.info(f"cell line (baseline): {baseline_cell_line_cond}")
            marker_baseline_idx_r1 = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_rep1')>-1)[0]
            marker_baseline_idx_r2 = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_rep2')>-1)[0]

            if len(marker_baseline_idx_r1) == 0 and len(marker_baseline_idx_r2) == 0:
                logging.warn(f"Marker {marker} couldn't be found in batch {batch}. Skipping this marker..")
                continue

            r1_size1 = len(marker_baseline_idx_r1) // 2
            r2_size1 = len(marker_baseline_idx_r2) // 2

            r1_part1 = np.random.choice(marker_baseline_idx_r1, size=r1_size1, replace=False)
            r1_part2 = marker_baseline_idx_r1[~np.isin(marker_baseline_idx_r1, r1_part1)]

            r2_part1 = np.random.choice(marker_baseline_idx_r2, size=r2_size1, replace=False)
            r2_part2 = marker_baseline_idx_r2[~np.isin(marker_baseline_idx_r2, r2_part1)]

            logging.info(f"|r1_part1| = {len(r1_part1)}, |r1_part2| = {len(r1_part2)}, |r2_part1| = {len(r2_part1)}, |r2_part2| = {len(r2_part2)}")

            d = {'rep1_part1': r1_part1, 'rep1_part2': r1_part2, 'rep2_part1': r2_part1, 'rep2_part2': r2_part2}
            partial_reps = itertools.product(['rep1_part1', 'rep1_part2'], ['rep2_part1', 'rep2_part2'])

            for ra,rb in partial_reps:
                d_ra, d_rb = d[ra], d[rb]
                cur_labels = np.concatenate([[ra]*len(d_ra),[rb]*len(d_rb)])
                cur_embeddings = np.concatenate([embeddings[d_ra],embeddings[d_rb]])
                
                n_clusters = 2
                kmeans_labels = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(cur_embeddings)
                kmeans_constrained_labels = cluster_without_outliers(cur_embeddings, n_clusters=n_clusters, outliers_fraction=0.1, n_init=10, random_state=1)
                gmm_labels = GaussianMixture(n_components=n_clusters, random_state=1).fit_predict(cur_embeddings)
                spectral_labels = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=1).fit_predict(cur_embeddings)

                
                score = pd.DataFrame({'silhouette':[silhouette_score(cur_embeddings, cur_labels)],
                                      'ARI_KMeans': adjusted_rand_score(cur_labels, kmeans_labels),
                                      'ARI_KMeansConstrained': [adjusted_rand_score(cur_labels, kmeans_constrained_labels)],
                                      'ARI_GMM': [adjusted_rand_score(cur_labels, gmm_labels)],
                                      'ARI_Spectral': [adjusted_rand_score(cur_labels, spectral_labels)],
                                        'marker':[marker],
                                        'condition': [baseline_cell_line_cond],
                                        'repA':[ra],
                                        'repB': [rb],
                                        'batch': [batch]})
                if scores.shape[0]==0:
                    scores = score
                else:
                    scores = pd.concat([scores, score])
        
            
            
            for als in conditions:
                logging.info(f"cell line: {als}")
                reps = itertools.product(['rep1', 'rep2'], repeat=2)
                for repA,repB in reps:
                    marker_als_idx = np.where(np.char.find(labels.astype(str), f'{marker}_{als}_{batch}_{repA}')>-1)[0]
                    marker_baseline_idx = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_{repB}')>-1)[0]
                    
                    logging.info(f"|{marker}_{als}_{batch}_{repA}| = {len(marker_als_idx)}")
                    logging.info(f"|{marker}_{baseline_cell_line_cond}_{batch}_{repB}| = {len(marker_baseline_idx)}")
                    if len(marker_als_idx) == 0:
                        logging.info(f"No samples for {marker}_{als}_{batch}_{repA}")
                        continue
                    if len(marker_baseline_idx) == 0:
                        logging.info(f"No samples for {marker}_{baseline_cell_line_cond}_{repB}")
                        continue
                    
                    cur_labels = np.concatenate([labels[marker_baseline_idx],labels[marker_als_idx]])
                    cur_embeddings = np.concatenate([embeddings[marker_baseline_idx],embeddings[marker_als_idx]])
                    
                    n_clusters = 2
                    kmeans_constrained_labels = cluster_without_outliers(cur_embeddings, n_clusters=n_clusters, outliers_fraction=0.1, n_init=10, random_state=1)
                    kmeans_labels = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(cur_embeddings)
                    gmm_labels = GaussianMixture(n_components=n_clusters, random_state=1).fit_predict(cur_embeddings)
                    spectral_labels = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=1).fit_predict(cur_embeddings)
                    
                    score = pd.DataFrame({'silhouette':[silhouette_score(cur_embeddings, cur_labels)],
                                          'ARI_KMeans': adjusted_rand_score(cur_labels, kmeans_labels),
                                        'ARI_KMeansConstrained': [adjusted_rand_score(cur_labels, kmeans_constrained_labels)],
                                        'ARI_GMM': [adjusted_rand_score(cur_labels, gmm_labels)],
                                        'ARI_Spectral': [adjusted_rand_score(cur_labels, spectral_labels)],
                                        'marker':[marker],
                                        'condition': [als],
                                        'repA':[repA],
                                        'repB': [repB],
                                        'batch': [batch]})
                    if scores.shape[0]==0:
                        scores = score
                    else:
                        scores = pd.concat([scores, score])
        
    savepath = os.path.join(output_folder_path, f"metrics_score_updated_{'_'.join(batches)}_{baseline_cell_line_cond}.csv")
    logging.info(f"Saving scores to {savepath}")
    scores.to_csv(savepath, index=False)


if __name__ == "__main__":
    try:
        config_data_path = sys.argv[1]
        model_output_folder = sys.argv[2]
        calculate_metrics_scores(config_data_path, model_output_folder)
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done!")
    
# example: 
# ./bash_commands/run_py.sh ./src/distances/calculate_metrics_scores -m 40000 -a ./src/distances/model_comparisons_distances_config/NeuronsDistanceConfig /home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/transfer_b78_freeze_least_changed -q long  -j calc_silhouette