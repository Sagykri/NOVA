from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig ## TODO SAGY CHANGE

from src.Analysis.analyzer_umap import AnalyzerUMAP
from src.common.lib.utils import handle_log, get_if_exists

import logging
import numpy as np

class AnalyzerUMAP0(AnalyzerUMAP):
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        super().__init__(trainer_conf, data_conf)


    def calculate(self, embeddings, labels):
        model_output_folder = self.output_folder_path
        handle_log(model_output_folder)

        markers = np.unique([m.split('_')[0] if '_' in m else m for m in np.unique(labels.reshape(-1,))]) 
        logging.info(f"Detected markers: {markers}")
        
        umap_embeddings = None
        for c in markers:
            logging.info(f"Marker: {c}")
            logging.info(f"[{c}] Selecting indexes of marker")
            c_indexes = np.where(np.char.startswith(labels.astype(str), f"{c}_"))[0]
            logging.info(f"[{c}] {len(c_indexes)} indexes have been selected")

            if len(c_indexes) == 0:
                logging.info(f"[{c}] Not exists in embedding. Skipping to the next one")
                continue

            embeddings_c, labels_c = np.copy(embeddings[c_indexes]), np.copy(labels[c_indexes].reshape(-1,))
            
            logging.info(f"[{c}] calc umap...")
            
            c_umap_embeddings = self.__compute_umap_embeddings(embeddings_c)
            c_features = np.hstack([c_umap_embeddings, labels_c.reshape(-1,1)])
            if umap_embeddings is None:
                umap_embeddings = c_features
            else:
                umap_embeddings = np.concatenate([umap_embeddings, c_features])
            
        self.features = umap_embeddings