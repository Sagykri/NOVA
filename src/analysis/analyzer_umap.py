import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))


from src.analysis.analyzer import Analyzer
from src.datasets.dataset_config import DatasetConfig
from src.analysis.analyzer_utils import calc_ari_with_kmeans

from abc import abstractmethod
from enum import Enum
import numpy as np
import umap
from typing import Tuple, Dict
import inspect
class AnalyzerUMAP(Analyzer):
    class UMAPType(Enum):
        SINGLE_MARKERS = 0
        MULTIPLE_MARKERS = 1
        MULTIPLEX_MARKERS = 2

    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        """Get an instance

        Args:
            data_config (DatasetConfig): the dataset configuration
            output_folder_path (str): path to output folder

        """
        super().__init__(data_config, output_folder_path)
        self.labels = None


    @abstractmethod
    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str], paths:np.ndarray[str])->Tuple[np.ndarray[float],np.ndarray[str], 
                                                                                                            np.ndarray[str], Dict[str,float]]:
        """Calculate UMAP embeddings from given embeddings, save in the self.features attribute, and return.
        Also save in self.labels the labels of the embeddings, and return them.

        Args:
            embeddings (np.ndarray[float]): The embeddings
            labels (np.ndarray[str]): The corresponding labels of the embeddings
        Returns:
            np.ndarray[float]: the UMAP embeddings
            np.ndarray[str]: the labels
            np.ndarray[str]: the paths
            Dict[str,float] : ari scores if needed
        """
        pass
    
    
    def load(self, umap_idx:int)->None:
        """load the saved UMAP embeddings into the self.features attribute,
        load the saved labels into the self.labels attribute

        Args:
            umap_idx (int): int indicating the umap type (0,1,2)
        """
        umap_type = self.UMAPType(umap_idx).name
        self.save_path = os.path.join('UMAPs', self.UMAPType(umap_idx).name)
        saveroot = self.save_path
        self.features = np.load(os.path.join(saveroot,f'{umap_type}.npy'))
        self.labels = np.load(os.path.join(saveroot,f'{umap_type}_labels.npy'))
    
    def save(self, umap_idx:int)->None:
        """save the calculated UMAP embeddings and labels in path derived from self.output_folder_path

        Args:
            umap_idx (int): int indicating the umap type (0,1,2)
        """
        umap_type = self.UMAPType(umap_idx).name
        self.save_path = os.path.join('UMAPs', self.UMAPType(umap_idx).name)
        saveroot = self.save_path
        os.makedirs(saveroot, exist_ok=True)
        np.save(os.path.join(saveroot,f'{umap_type}.npy'), self.features)
        np.save(os.path.join(saveroot,f'{umap_type}_labels.npy'), self.labels)

    def _compute_umap_embeddings(self, embeddings:np.ndarray[float], **kwargs)->np.ndarray:
        """Protected method to calculate UMAP dimensionality reduction on the provided embeddings.

        Args:
            embeddings (np.ndarray[float]): The input embeddings to perform UMAP on. 
            The shape should be (n_samples, n_features), where each row corresponds 
            to a sample and each column represents a feature.
            **kwargs: Additional keyword arguments to pass to the UMAP constructor, 
            such as `n_neighbors`, `min_dist`, `metric`, etc. If `random_state` 
            is not provided, it will be set to `self.data_config.SEED` to ensure 
            reproducibility.

        Returns:
            np.ndarray[float]: The reduced UMAP embeddings with shape (n_samples, n_components),
                                where `n_components` is typically 2 but can be modified via kwargs.
        """
        if 'random_state' not in kwargs:
            kwargs['random_state'] = self.data_config.SEED
        reducer = umap.UMAP(**kwargs)
        umap_embeddings = reducer.fit_transform(embeddings)
        return umap_embeddings
    
    def _compute_ari(self, embeddings:np.ndarray[float], labels:np.ndarray[str], **kwargs)->float:
        """Function to compute ari score given embeddings and true labels.

        Args:
            embeddings (np.ndarray[float]): data to calculate the ari for
            labels (np.ndarray[str]): corresponding labels
            **kwargs: Additional keyword arguments passed to the `calc_ari_with_kmeans`

        Returns:
            float: the ari score
        """
        clustering_kwargs = {a: kwargs[a] for a in inspect.signature(calc_ari_with_kmeans).parameters if a in kwargs}
        score = calc_ari_with_kmeans(embeddings, labels, **clustering_kwargs)
        return score

