import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

import numpy as np
from typing import Tuple

from src.analysis.analyzer_utils import calc_ari_with_kmeans
from src.analysis.analyzer_distances import AnalyzerDistances
from src.datasets.dataset_config import DatasetConfig
from src.common.utils import get_if_exists

class AnalyzerDistancesARI(AnalyzerDistances):
    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        super().__init__(data_config, output_folder_path)  

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
    
        score = calc_ari_with_kmeans(embeddings, labels)
        return score, 'ARI_KMeansConstrained'
    
    def _get_save_path(self, output_folder_path:str)->str:
        
        baseline_cell_line_cond = get_if_exists(self.data_config, 'BASELINE_CELL_LINE_CONDITION', None)

        savepath = os.path.join(output_folder_path, f"metrics_score_{baseline_cell_line_cond}.csv")
        return savepath

