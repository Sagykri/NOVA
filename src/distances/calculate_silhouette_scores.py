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
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

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

def calculate_silhouette_scores(config_path_data, model_output_folder):
    __handle_log(model_output_folder)

    config_data = load_config_file(config_path_data, 'data')
    
    output_folder_path = os.path.join(model_output_folder, 'figures', config_data.EXPERIMENT_TYPE, 'distances')
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)

    train_batches = get_if_exists(config_data, 'TRAIN_BATCHES', None)
    assert train_batches is not None, "train_batches can't be None for distances" 

    embeddings, labels = __load_vit_features(model_output_folder, config_data, train_batches)


    # ALS_lines = ['FUSHomozygous','FUSHeterozygous','FUSRevertant', 'OPTN','TBK1','TDP43']
    # conditions = [als+'_Untreated' for als in ALS_lines] + ['WT_stress']

    baseline_cell_line_cond = get_if_exists(config_data, 'BASELINE_CELL_LINE_CONDITION', None)
    assert baseline_cell_line_cond is not None, "BASELINE_CELL_LINE_CONDITION is None. You have to specify the baseline to calculate the silhouette against (for example: WT_Untreated or TDP43_Untreated)"
    
    conditions = np.unique(['_'.join(l.split("_")[1:3]) for l in labels]) # Extract the list of cell_line_conds
    conditions = np.delete(conditions, np.where(conditions == baseline_cell_line_cond)[0]) # Remove the WT_Untreated
    
    markers = np.unique([label.split('_')[0] for label in np.unique(labels)])
    batches = [input_folder.split(os.sep)[-1] for input_folder in config_data.INPUT_FOLDERS] #['batch6', 'batch9']
    
    logging.info(f"model_output_folder: {model_output_folder}, config_path_data: {config_path_data}")
    logging.info(f"config_data: {config_data.__dict__}")
    logging.info(f"Conditions: {conditions}, markers: {markers}, batches: {batches}")

    scores = pd.DataFrame(columns=['silhouette','marker', 'condition', 'repA', 'repB', 'batch'])
    distances = pd.DataFrame(columns=['dist','marker', 'condition', 'repA', 'repB', 'batch'])
    for batch in batches:
        logging.info(f"batch: {batch}")
        for marker in markers:   
            
            logging.info(f"marker: {marker}")
            
            logging.info(f"cell line (baseline): {baseline_cell_line_cond}")
            marker_wt_idx_r1 = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_rep1')>-1)[0]
            marker_wt_idx_r2 = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_rep2')>-1)[0]
            if len(marker_wt_idx_r1) == 0:
                logging.info(f"No samples for {marker}_{baseline_cell_line_cond}_{batch}_rep1")
                continue
            if len(marker_wt_idx_r2) == 0:
                logging.info(f"No samples for {marker}_{baseline_cell_line_cond}_{batch}_rep2")
                continue
            # r1_size1 = len(marker_wt_idx_r1) // 2
            # r2_size1 = len(marker_wt_idx_r2) // 2

            # r1_part1 = np.random.choice(len(marker_wt_idx_r1), r1_size1, replace=False)
            # r1_part2 = np.delete(marker_wt_idx_r1, r1_part1)

            # r2_part1 = np.random.choice(len(marker_wt_idx_r2), r2_size1, replace=False)
            # r2_part2 = np.delete(marker_wt_idx_r2, r2_part1)
            
            # logging.info(f"|r1_part1| = {len(r1_part1)}, |r1_part2| = {len(r1_part2)}, |r2_part1| = {len(r2_part1)}, |r2_part2| = {len(r2_part2)}")

            # d = {'rep1_part1': r1_part1, 'rep1_part2': r1_part2, 'rep2_part1': r2_part1, 'rep2_part2': r2_part2}
            # partial_reps = itertools.product(['rep1_part1', 'rep1_part2'], ['rep2_part1', 'rep2_part2'])

            # for ra,rb in partial_reps:
            #     d_ra, d_rb = d[ra], d[rb]
            #     cur_labels = np.concatenate([[ra]*len(d_ra),[rb]*len(d_rb)])
            #     cur_embeddings = np.concatenate([embeddings[d_ra],embeddings[d_rb]])
            #     score = pd.DataFrame({'silhouette':[silhouette_score(cur_embeddings, cur_labels)],
            #                             'marker':[marker],
            #                             'condition': [baseline_cell_line_cond],
            #                             'repA':[ra],
            #                             'repB': [rb],
            #                             'batch': [batch]})
            #     if scores.shape[0]==0:
            #         scores = score
            #     else:
            #         scores = pd.concat([scores, score])

            cur_labels = np.concatenate([labels[marker_wt_idx_r1],labels[marker_wt_idx_r2]])
            cur_embeddings = np.concatenate([embeddings[marker_wt_idx_r1],embeddings[marker_wt_idx_r2]])
            score = pd.DataFrame({'silhouette':[silhouette_score(cur_embeddings, cur_labels)],
                                    'marker':[marker],
                                    'condition': [baseline_cell_line_cond],
                                    'repA':['rep1'],
                                    'repB': ['rep2'],
                                    'batch': [batch]})
            if scores.shape[0]==0:
                scores = score
            else:
                scores = pd.concat([scores, score])
            if marker!='DAPI':
                
                dists=pairwise_distances(X=embeddings[marker_wt_idx_r1],
                                        Y = embeddings[marker_wt_idx_r2],
                                        metric='euclidean', n_jobs=-1)
                dists = pd.DataFrame({'dist':dists.flatten(),
                                    'marker':marker,
                                    'condition':baseline_cell_line_cond,
                                    'repA':'rep1',
                                    'repB':'rep2',
                                    'batch':batch,})

                if distances.shape[0]==0:
                    distances = dists
                else:
                    distances = pd.concat([distances, dists])
            
            for als in conditions:
                logging.info(f"cell line: {als}")
                # reps = itertools.product(['rep1', 'rep2'], repeat=2)
                # for repA,repB in reps:
                reps = ['rep1','rep2']
                for rep in reps:
                    marker_als_idx = np.where(np.char.find(labels.astype(str), f'{marker}_{als}_{batch}_{rep}')>-1)[0]
                    marker_wt_idx = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_{rep}')>-1)[0]
                    
                    logging.info(f"|{marker}_{als}_{batch}_{rep}| = {len(marker_als_idx)}")
                    logging.info(f"|{marker}_{baseline_cell_line_cond}_{batch}_{rep}| = {len(marker_wt_idx)}")
                    if len(marker_als_idx) == 0:
                        logging.info(f"No samples for {marker}_{als}_{batch}_{rep}")
                        continue
                    if len(marker_wt_idx) == 0:
                        logging.info(f"No samples for {marker}_{baseline_cell_line_cond}_{rep}")
                        continue
                    
                    cur_labels = np.concatenate([labels[marker_wt_idx],labels[marker_als_idx]])
                    cur_embeddings = np.concatenate([embeddings[marker_wt_idx],embeddings[marker_als_idx]])
                    score = pd.DataFrame({'silhouette':[silhouette_score(cur_embeddings, cur_labels)],
                                        'marker':[marker],
                                        'condition': [als],
                                        'repA':[rep],
                                        'repB': [rep],
                                        'batch': [batch]})
                    if scores.shape[0]==0:
                        scores = score
                    else:
                        scores = pd.concat([scores, score])
                    if marker!='DAPI':
                        dists=pairwise_distances(X=embeddings[marker_wt_idx],
                                            Y =embeddings[marker_als_idx],
                                            metric='euclidean', n_jobs=-1)

                        dists = pd.DataFrame({'dist':dists.flatten(),
                                'marker':marker,
                                'condition':als,
                                'repA':'rep1',
                                'repB':'rep2',
                                'batch':batch,})
                        if distances.shape[0]==0:
                            distances = dists
                        else:
                            distances = pd.concat([distances, dists])
        
    savepath = os.path.join(output_folder_path, f"silhouette_score_matching_reps_{'_'.join(batches)}_vs_{baseline_cell_line_cond}.csv")
    logging.info(f"Saving scores to {savepath}")
    scores.to_csv(savepath, index=False)

    savepath = os.path.join(output_folder_path, f"pairwise_distance_matching_reps_{'_'.join(batches)}_vs_{baseline_cell_line_cond}.csv")
    logging.info(f"Saving distances to {savepath}")
    distances.to_csv(savepath, index=False)


if __name__ == "__main__":
    try:
        config_data_path = sys.argv[1]
        model_output_folder = sys.argv[2]
        calculate_silhouette_scores(config_data_path, model_output_folder)
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done!")
    
# example: 
# ./bash_commands/run_py.sh ./src/distances/calculate_silhouette_scores -m 40000 -a ./src/distances/model_comparisons_distances_config/NeuronsDistanceConfig /home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/transfer_b78_freeze_least_changed -q long  -j calc_silhouette