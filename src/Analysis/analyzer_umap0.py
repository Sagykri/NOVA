from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig ## TODO SAGY CHANGE

from src.Analysis.analyzer_umap import AnalyzerUMAP

import logging
import numpy as np

class AnalyzerUMAP0(AnalyzerUMAP):
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        super().__init__(trainer_conf, data_conf)


    def calculate(self, embeddings:np.ndarray, labels:np.ndarray)->None:
        """Calculate UMAP embeddings from given embeddings, separatly for each marker in the given embeddings.
        Then save all together in the self.features attribute

        Args:
            embeddings (np.ndarray): The embeddings
            labels (np.ndarray): The corresponding labels of the embeddings
        """

        markers = np.unique([m.split('_')[0] if '_' in m else m for m in np.unique(labels.reshape(-1,))]) 
        logging.info(f"[AnalyzerUMAP0.calculate] Detected markers: {markers}")
        
        umap_embeddings = None
        for marker in markers:
            logging.info(f"Marker: {marker}")
            indices = np.where(np.char.startswith(labels.astype(str), f"{marker}_"))[0]
            logging.info(f"{len(indices)} indexes have been selected")

            if len(indices) == 0:
                logging.info(f"No data for marker {marker}, skipping.")
                continue

            marker_embeddings, marker_labels = np.copy(embeddings[indices]), np.copy(labels[indices].reshape(-1,))
                        
            marker_umap_embeddings = self._compute_umap_embeddings(marker_embeddings)
            marker_umap_embeddings = np.hstack([marker_umap_embeddings, marker_labels.reshape(-1,1)])
            if umap_embeddings is None:
                umap_embeddings = marker_umap_embeddings
            else:
                umap_embeddings = np.concatenate([umap_embeddings, marker_umap_embeddings])
            
        self.features = umap_embeddings