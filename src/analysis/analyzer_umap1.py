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
        """Calculate UMAP embeddings from given embeddings and save in the self.features attribute

        Args:
            embeddings (np.ndarray[float]): The embeddings
            labels (np.ndarray[str]): The corresponding labels of the embeddings
        Returns:#TODO
        """
        logging.info(f"[AnalyzerUMAP1.calculate] Calulating UMAP")
        umap_embeddings = self._compute_umap_embeddings(embeddings)

        self.features = umap_embeddings
        self.labels = labels
        return umap_embeddings, labels