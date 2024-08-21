from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig ## TODO SAGY CHANGE

from src.Analysis.analyzer_distances import AnalyzerDistances
from src.common.lib.utils import handle_log, get_if_exists

import logging
import os
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import adjusted_rand_score
from src.common.lib.metrics import cluster_without_outliers
from sklearn.cluster import KMeans

class AnalyzerDistancesARI(AnalyzerDistances):
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        super().__init__(trainer_conf, data_conf)


    def __calculate_metrices_for_batch_and_marker(embeddings, labels, baseline_cell_line_cond,
                                                  conditions, batch, marker, scores):
        np.random.seed(1)
        # We start with calculating the score of difference between the baseline and itself
        logging.info(f"cell line (baseline): {baseline_cell_line_cond}")
        marker_baseline_idx_r1 = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_rep1')>-1)[0]
        marker_baseline_idx_r2 = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_rep2')>-1)[0]

        if len(marker_baseline_idx_r1) == 0 and len(marker_baseline_idx_r2) == 0:
            logging.warn(f"Marker {marker} couldn't be found in batch {batch}. Skipping this marker..")
            return scores

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
            # gmm_labels = GaussianMixture(n_components=n_clusters, random_state=1).fit_predict(cur_embeddings)
            # spectral_labels = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=1).fit_predict(cur_embeddings)

            score = pd.DataFrame({'ARI_KMeans': adjusted_rand_score(cur_labels, kmeans_labels),
                                    'ARI_KMeansConstrained': [adjusted_rand_score(cur_labels, kmeans_constrained_labels)],
                                #   'ARI_GMM': [adjusted_rand_score(cur_labels, gmm_labels)],
                                #   'ARI_Spectral': [adjusted_rand_score(cur_labels, spectral_labels)],
                                #   'silhouette':[silhouette_score(cur_embeddings, cur_labels)],
                                    'marker':[marker],
                                    'condition': [baseline_cell_line_cond],
                                    'repA':[ra],
                                    'repB': [rb],
                                    'batch': [batch]})
            if scores.shape[0]==0:
                scores = score
            else:
                scores = pd.concat([scores, score])
    
        
        # Then we calculate the difference score between the conditions and the baseline:
        for cond in conditions:
            logging.info(f"cell line: {cond}")
            reps = itertools.product(['rep1', 'rep2'], repeat=2)
            for repA,repB in reps:
                marker_als_idx = np.where(np.char.find(labels.astype(str), f'{marker}_{cond}_{batch}_{repA}')>-1)[0]
                marker_baseline_idx = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_{repB}')>-1)[0]
                
                logging.info(f"|{marker}_{cond}_{batch}_{repA}| = {len(marker_als_idx)}")
                logging.info(f"|{marker}_{baseline_cell_line_cond}_{batch}_{repB}| = {len(marker_baseline_idx)}")
                if len(marker_als_idx) == 0:
                    logging.info(f"No samples for {marker}_{cond}_{batch}_{repA}")
                    continue
                if len(marker_baseline_idx) == 0:
                    logging.info(f"No samples for {marker}_{baseline_cell_line_cond}_{repB}")
                    continue
                
                cur_labels = np.concatenate([labels[marker_baseline_idx],labels[marker_als_idx]])
                cur_embeddings = np.concatenate([embeddings[marker_baseline_idx],embeddings[marker_als_idx]])
                
                n_clusters = 2
                kmeans_constrained_labels = cluster_without_outliers(cur_embeddings, n_clusters=n_clusters, outliers_fraction=0.1, n_init=10, random_state=1)
                kmeans_labels = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(cur_embeddings)
                # gmm_labels = GaussianMixture(n_components=n_clusters, random_state=1).fit_predict(cur_embeddings)
                # spectral_labels = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=1).fit_predict(cur_embeddings)
                
                score = pd.DataFrame({'ARI_KMeans': adjusted_rand_score(cur_labels, kmeans_labels),
                                    'ARI_KMeansConstrained': [adjusted_rand_score(cur_labels, kmeans_constrained_labels)],
                                    # 'ARI_GMM': [adjusted_rand_score(cur_labels, gmm_labels)],
                                    # 'ARI_Spectral': [adjusted_rand_score(cur_labels, spectral_labels)],
                                    # 'silhouette':[silhouette_score(cur_embeddings, cur_labels)],
                                    'marker':[marker],
                                    'condition': [cond],
                                    'repA':[repA],
                                    'repB': [repB],
                                    'batch': [batch]})
                if scores.shape[0]==0:
                    scores = score
                else:
                    scores = pd.concat([scores, score])
        return scores

    def calculate(self, embeddings, labels):
        model_output_folder = self.output_folder_path
        handle_log(model_output_folder)
    
        baseline_cell_line_cond = get_if_exists(self.data_conf, 'BASELINE_CELL_LINE_CONDITION', None)
        assert baseline_cell_line_cond is not None, "BASELINE_CELL_LINE_CONDITION is None. You have to specify the baseline to calculate the silhouette agains (for example: WT_Untreated or TDP43_Untreated)"

        conditions = np.unique(['_'.join(l.split("_")[1:3]) for l in labels]) # Extract the list of cell_line_conds
        conditions = np.delete(conditions, np.where(conditions == baseline_cell_line_cond)[0]) # Remove the WT_Untreated
        markers = np.unique([label.split('_')[0] for label in np.unique(labels)])
        batches = [input_folder.split(os.sep)[-1] for input_folder in self.input_folders] 

        logging.info(f"Conditions: {conditions}, markers: {markers}, batches: {batches}, baseline_cell_line_cond: {baseline_cell_line_cond}")

        scores = pd.DataFrame(columns=['silhouette', 'ARI', 'ARI_constrained', 'marker', 'condition', 'repA', 'repB', 'batch'])

        for batch in batches:
            logging.info(f"batch: {batch}")
            for marker in markers:   
                logging.info(f"marker: {marker}")
                self.__calculate_metrices_for_batch_and_marker(embeddings, labels, baseline_cell_line_cond,
                                                  conditions, batch, marker, scores)
            

        self.features = scores
        return None