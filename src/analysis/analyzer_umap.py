import sys
import os
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from abc import abstractmethod

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig
from src.analysis.analyzer import Analyzer

import logging
import os
import numpy as np
import umap
from typing import Tuple

class AnalyzerUMAP(Analyzer):
    def __init__(self, trainer_config: TrainerConfig, data_config: DatasetConfig):
        """Get an instance

        Args:
            trainer_config (TrainerConfig): The trainer configuration
            data_config (DatasetConfig): the dataset configuration
        """
        super().__init__(trainer_config, data_config)
        self.labels = None


    @abstractmethod
    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str])->Tuple[np.ndarray[float],np.ndarray[str]]:
        """Calculate UMAP embeddings from given embeddings, save in the self.features attribute, and return.
        Also save in self.labels the labels of the embeddings, and return them.

        Args:
            embeddings (np.ndarray[float]): The embeddings
            labels (np.ndarray[str]): The corresponding labels of the embeddings
        Returns:
            np.ndarray[float]: the UMAP embeddings
            np.ndarray[str]: the labels
        """
        pass
    
    
    def load(self, umap_type:str)->Tuple[np.ndarray[float], np.ndarray[str]]:
        """load the saved UMAP embeddings into the self.features attribute,
        load the saved labels into the self.labels attribute

        Args:
            umap_type (str): string indicating the umap type ('umap0','umap1','umap2')
        """
        model_output_folder = self.output_folder_path

        output_folder_path = os.path.join(model_output_folder, 'figures', self.data_config.EXPERIMENT_TYPE,'UMAP', umap_type)
        if not os.path.exists(output_folder_path):
            logging.info(f"{output_folder_path} doesn't exists. Can't load!")
            return None

        title = f"{'_'.join([os.path.basename(f) for f in self.data_config.INPUT_FOLDERS])}_{'_'.join(self.data_config.REPS)}"
        saveroot = os.path.join(output_folder_path,f'{title}')
        if not os.path.exists(saveroot):
            logging.info(f"{saveroot} doesn't exists. Can't load!")
            return None
        
        self.features = np.load(f'{saveroot}_{umap_type}.npy')
        self.labels = np.load(f'{saveroot}_{umap_type}_labels.npy')
    
    def save(self, umap_type:str)->None:
        """save the calculated UMAP embeddings and labels in path derived from self.output_folder_path

        Args:
            umap_type (str): string indicating the umap type ('umap0','umap1','umap2')
        """
        model_output_folder = self.output_folder_path

        output_folder_path = os.path.join(model_output_folder, 'figures', self.data_config.EXPERIMENT_TYPE,'UMAP', umap_type)
        if not os.path.exists(output_folder_path):
            logging.info(f"{output_folder_path} doesn't exists. Creating it")
            os.makedirs(output_folder_path, exist_ok=True)

        title = f"{'_'.join([os.path.basename(f) for f in self.data_config.INPUT_FOLDERS])}_{'_'.join(self.data_config.REPS)}"
        saveroot = os.path.join(output_folder_path,f'{title}')
        if not os.path.exists(saveroot):
            os.makedirs(saveroot, exist_ok=True)
        
        np.save(f'{saveroot}_{umap_type}.npy', self.features)
        np.save(f'{saveroot}_{umap_type}_labels.npy', self.labels)

    def _compute_umap_embeddings(self, embeddings:np.ndarray[float], **kwargs)->np.ndarray:
        """Protected method to calculate UMAP reduction given embeddings

        Args:
            embeddings (np.ndarray[float]): embeddings to calculate UMAP on
            **kwargs TODO add

        Returns:
            np.ndarray[float]: the UMAP embeddings
        """
        if 'random_state' not in kwargs:
            kwargs['random_state'] = self.data_config.SEED
        reducer = umap.UMAP(**kwargs)
        umap_embeddings = reducer.fit_transform(embeddings)
        return umap_embeddings