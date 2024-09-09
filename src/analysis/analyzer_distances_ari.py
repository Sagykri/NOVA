import sys
import os
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np
from typing import Tuple
# from sklearn.metrics import adjusted_rand_score

from src.analysis.analyzer_distances import AnalyzerDistances
from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig
from src.common.lib.metrics import calc_clustering_validation_metric
from src.common.lib.utils import get_if_exists
# from src.datasets.label_utils import get_batches_from_input_folders

class AnalyzerDistancesARI(AnalyzerDistances):
    def __init__(self, trainer_config: TrainerConfig, data_config: DatasetConfig):
        super().__init__(trainer_config, data_config)  

    def _compute_score(self, embeddings: np.ndarray[float], labels: np.ndarray[float]) -> Tuple[float,str]:
        """Compute ARI scores
        Args:
            embeddings (np.ndarray[float]): embeddings to calculate scores on
            labels (np.ndarray[str]): labels of the embeddings to calculate scores on; should contain only 2 unique labels

        Returns:
            float: the ari with kmeans constrained score 
            str: the score name
        """
        try:
            assert np.unique(labels).shape[0] == 2, "There must be exactly 2 unique labels."
        except AssertionError as e:
            raise ValueError(f"Label validation failed: {e}, np.unique(labels):{np.unique(labels)}")
    
        # n_clusters = 2
        # kmeans_constrained_labels = cluster_without_outliers(embeddings, n_clusters=n_clusters, outliers_fraction=0.1, n_init=10, random_state=1)

        # score = adjusted_rand_score(labels, kmeans_constrained_labels)
        score_dict = calc_clustering_validation_metric(embeddings, labels)
        return score_dict['ARI'], 'ARI_KMeansConstrained'
    
    def _get_save_path(self, output_folder_path:str)->str:
        
        baseline_cell_line_cond = get_if_exists(self.data_config, 'BASELINE_CELL_LINE_CONDITION', None)

        savepath = os.path.join(output_folder_path, f"metrics_score_{baseline_cell_line_cond}.csv")
        return savepath

