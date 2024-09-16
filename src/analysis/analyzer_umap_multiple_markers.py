import sys
import os
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.dataset_config import DatasetConfig

from src.analysis.analyzer_umap import AnalyzerUMAP

import logging
import numpy as np
import logging
from typing import Tuple, Dict

class AnalyzerUMAPMultipleMarkers(AnalyzerUMAP):
    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        super().__init__(data_config, output_folder_path)


    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str])->Tuple[np.ndarray[float],np.ndarray[str], Dict[str,float]]:
        """Calculate UMAP embeddings from the given embeddings and store the results in the `self.features` attribute. 

        Args:
            embeddings (np.ndarray[float]): The input embeddings with shape (n_samples, n_features).
            labels (np.ndarray[str]): The labels associated with each embedding.
        Returns:
            Tuple[np.ndarray[float], np.ndarray[str]]: 
                - The UMAP embeddings after dimensionality reduction with shape (n_samples, n_components).
                - The corresponding labels preserving the association with the UMAP embeddings.
                - A dictionary with 'ari' as key and the ari score as value
        """
        logging.info(f"[AnalyzerUMAP1.calculate] Calulating UMAP")
        umap_embeddings = self._compute_umap_embeddings(embeddings)

        if self.data_config.SHOW_ARI:
            ari = self._compute_ari(embeddings, labels)
            ari_score = {'ari':ari}
        
        else:
            ari_score = {}
        
        self.features = umap_embeddings
        self.labels = labels
        self.ari_scores = ari_score

        return umap_embeddings, labels, ari_score