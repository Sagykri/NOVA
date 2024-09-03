import sys
import os
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig

from src.analysis.analyzer_umap import AnalyzerUMAP
from src.common.lib.synthetic_multiplexing import __embeddings_to_df as embeddings_to_df, __get_multiplexed_embeddings as get_multiplexed_embeddings
import logging
import numpy as np
from typing import Tuple

class AnalyzerUMAP2(AnalyzerUMAP):
    def __init__(self, trainer_config: TrainerConfig, data_config: DatasetConfig):
        super().__init__(trainer_config, data_config)


    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str])->Tuple[np.ndarray[float],np.ndarray[str]]:
        """Calculate UMAP embeddings from the concatenation of given embeddings, grouping by the cell line and condition.
        Then save in the self.features attribute

        Args:
            embeddings (np.ndarray[float]): The embeddings
            labels (np.ndarray[str]): The corresponding labels of the embeddings
        """
                
        logging.info(f"[Before concat] Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")
    
        df = embeddings_to_df(embeddings, labels, self.data_config)
        embeddings, label_data, _ = get_multiplexed_embeddings(df, random_state=self.data_config.SEED)
        logging.info(f"[After concat] Embeddings shape: {embeddings.shape}, Labels shape: {label_data.shape}")
        label_data = label_data.reshape(-1)
        logging.info(f"[AnalyzerUMAP2.calculate] Calulating UMAP of multiplex embeddings")
        umap_embeddings = self._compute_umap_embeddings(embeddings)
        label_data = label_data.astype(object)
        self.features = umap_embeddings
        self.labels = label_data