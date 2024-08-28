from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig ## TODO SAGY CHANGE

from src.Analysis.analyzer_umap import AnalyzerUMAP
from src.common.lib.synthetic_multiplexing import __embeddings_to_df as embeddings_to_df, __get_multiplexed_embeddings as get_multiplexed_embeddings
import logging
import numpy as np

class AnalyzerUMAP2(AnalyzerUMAP):
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        super().__init__(trainer_conf, data_conf)


    def calculate(self, embeddings:np.ndarray, labels:np.ndarray)->None:
        """Calculate UMAP embeddings from the concatenation of given embeddings, grouping by the cell line and condition.
        Then save in the self.features attribute

        Args:
            embeddings (np.ndarray): The embeddings
            labels (np.ndarray): The corresponding labels of the embeddings
        """
                
        logging.info(f"[Before concat] Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")
    
        df = embeddings_to_df(embeddings, labels, self.data_conf)
        embeddings, label_data, _ = get_multiplexed_embeddings(df, random_state=self.data_conf.SEED)
        logging.info(f"[After concat] Embeddings shape: {embeddings.shape}, Labels shape: {label_data.shape}")
        label_data = label_data.reshape(-1)
        logging.info(f"[AnalyzerUMAP2.calculate] Calulating UMAP of multiplex embeddings")
        umap_embeddings = self._compute_umap_embeddings(embeddings)
        label_data = label_data.astype(object)
        umap_embeddings = np.hstack([umap_embeddings, label_data.reshape(-1,1)])
        self.features = umap_embeddings