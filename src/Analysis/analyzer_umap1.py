from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig ## TODO SAGY CHANGE

from src.Analysis.analyzer_umap import AnalyzerUMAP
from src.common.lib.utils import handle_log, get_if_exists

import logging
import numpy as np

class AnalyzerUMAP1(AnalyzerUMAP):
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        super().__init__(trainer_conf, data_conf)


    def calculate(self, embeddings, labels):
        model_output_folder = self.output_folder_path
        handle_log(model_output_folder)
    
        umap_embeddings = self.__compute_umap_embeddings(embeddings)

        umap_embeddings = np.hstack([umap_embeddings, labels.reshape(-1,1)])
        self.features = umap_embeddings