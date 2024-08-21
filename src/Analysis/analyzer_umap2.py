from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig ## TODO SAGY CHANGE

from src.Analysis.analyzer_umap import AnalyzerUMAP
from src.common.lib.utils import handle_log, get_if_exists
from src.common.lib.synthetic_multiplexing import __embeddings_to_df, __get_multiplexed_embeddings

import logging
import numpy as np

class AnalyzerUMAP2(AnalyzerUMAP):
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        super().__init__(trainer_conf, data_conf)


    def calculate(self, embeddings, labels):
        model_output_folder = self.output_folder_path
        handle_log(model_output_folder)

        logging.info(f"[Before concat] Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")
    
        df = __embeddings_to_df(embeddings, labels, self.data_conf,  vq_type='vqindhist1')
        embeddings, label_data, unique_groups = __get_multiplexed_embeddings(df, random_state=self.data_conf.SEED)
        logging.info(f"[After concat] Embeddings shape: {embeddings.shape}, Labels shape: {label_data.shape}")
        label_data = label_data.reshape(-1)
        map_labels_function = get_if_exists(self.data_conf, 'MAP_LABELS_FUNCTION', None)
        if map_labels_function is not None:
            logging.info("Applyging map_labels_function from the config on the unique_groups")
            logging.info(f"unique groups before function: {unique_groups}")

            map_labels_function = eval(map_labels_function)(self.data_conf)
            unique_groups = map_labels_function(unique_groups)    
            logging.info(f"unique groups after function: {unique_groups}")

            label_data = map_labels_function(label_data)

        logging.info('[SM] computing umap')
        umap_embeddings = self.__compute_umap_embeddings(embeddings)

        umap_embeddings = np.hstack([umap_embeddings, label_data.reshape(-1,1)])
            
        self.features = umap_embeddings