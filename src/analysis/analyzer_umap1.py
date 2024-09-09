import sys
import os
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig

from src.analysis.analyzer_umap import AnalyzerUMAP

import logging
import numpy as np
import logging
from typing import Tuple

class AnalyzerUMAP1(AnalyzerUMAP):
    def __init__(self, trainer_config: TrainerConfig, data_config: DatasetConfig):
        super().__init__(trainer_config, data_config)


    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str])->Tuple[np.ndarray[float],np.ndarray[str]]:
        """Calculate UMAP embeddings from the given embeddings and store the results in the `self.features` attribute. 

        Args:
            embeddings (np.ndarray[float]): The input embeddings with shape (n_samples, n_features).
            labels (np.ndarray[str]): The labels associated with each embedding.
        Returns:
            Tuple[np.ndarray[float], np.ndarray[str]]: 
                - The UMAP embeddings after dimensionality reduction with shape (n_samples, n_components).
                - The corresponding labels preserving the association with the UMAP embeddings.
        """
        logging.info(f"[AnalyzerUMAP1.calculate] Calulating UMAP")
        umap_embeddings = self._compute_umap_embeddings(embeddings)

        self.features = umap_embeddings
        self.labels = labels
        return umap_embeddings, labels