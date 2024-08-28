from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig ## TODO SAGY CHANGE

from src.Analysis.analyzer_umap import AnalyzerUMAP

import logging
import numpy as np
import logging

class AnalyzerUMAP1(AnalyzerUMAP):
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        super().__init__(trainer_conf, data_conf)


    def calculate(self, embeddings:np.ndarray, labels:np.ndarray)->None:
        """Calculate UMAP embeddings from given embeddings and save in the self.features attribute

        Args:
            embeddings (np.ndarray): The embeddings
            labels (np.ndarray): The corresponding labels of the embeddings
        """
        logging.info(f"[AnalyzerUMAP1.calculate] Calulating UMAP")
        umap_embeddings = self._compute_umap_embeddings(embeddings)

        umap_embeddings = np.hstack([umap_embeddings, labels.reshape(-1,1)])
        self.features = umap_embeddings