from abc import abstractmethod

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig ## TODO SAGY CHANGE

from src.Analysis.analyzer import Analyzer
from src.common.lib.utils import handle_log

import logging
import os
import numpy as np
import umap

class AnalyzerUMAP(Analyzer):
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        super().__init__(trainer_conf, data_conf)


    @abstractmethod
    def calculate(self, embeddings, labels):
        pass
    
    
    def load(self, umap_type):
        model_output_folder = self.output_folder_path
        handle_log(model_output_folder)

        output_folder_path = os.path.join(model_output_folder, 'figures', self.experiment_type,'UMAP', umap_type)
        if not os.path.exists(output_folder_path):
            logging.info(f"{output_folder_path} doesn't exists. Can't load!")
            return None

        title = f"{'_'.join([os.path.basename(f) for f in self.input_folders])}_{'_'.join(self.reps)}"
        saveroot = os.path.join(output_folder_path,f'{title}')
        if not os.path.exists(saveroot):
            logging.info(f"{saveroot} doesn't exists. Can't load!")
            return None
        
        self.features = np.load(f'{saveroot}_{umap_type}.npy')
    
    def save(self, umap_type):
        model_output_folder = self.output_folder_path
        handle_log(model_output_folder)

        output_folder_path = os.path.join(model_output_folder, 'figures', self.experiment_type,'UMAP', umap_type)
        if not os.path.exists(output_folder_path):
            logging.info(f"{output_folder_path} doesn't exists. Creating it")
            os.makedirs(output_folder_path, exist_ok=True)

        title = f"{'_'.join([os.path.basename(f) for f in self.input_folders])}_{'_'.join(self.reps)}"
        saveroot = os.path.join(output_folder_path,f'{title}')
        if not os.path.exists(saveroot):
            os.makedirs(saveroot, exist_ok=True)
        
        np.save(f'{saveroot}_{umap_type}.npy', self.features)

    def __compute_umap_embeddings(self, embeddings, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                            n_components=n_components, random_state=random_state)
        umap_embeddings = reducer.fit_transform(embeddings)
        return umap_embeddings