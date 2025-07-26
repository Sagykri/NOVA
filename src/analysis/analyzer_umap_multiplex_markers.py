import sys
import os
from typing import Dict, Tuple

sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.datasets.label_utils import map_labels
from src.datasets.dataset_config import DatasetConfig

from src.analysis.analyzer_multiplex_markers import AnalyzerMultiplexMarkers
from src.analysis.analyzer_umap import AnalyzerUMAP
import logging
import numpy as np

class AnalyzerUMAPMultiplexMarkers(AnalyzerUMAP):

    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        super().__init__(data_config, output_folder_path)

        self.analyzer_multiplex_markers = AnalyzerMultiplexMarkers(data_config, output_folder_path)


    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str], paths:np.ndarray[str]=None)->Tuple[np.ndarray[float],np.ndarray[str], np.ndarray[str], Dict[str,float]]:
        """Calculate UMAP embeddings for multiplexed embeddings from the given embeddings and store the results in the `self.features` attribute. 
         This method takes in embeddings and their associated labels, and computes multiplexed embeddings by grouping the data based on shared phenotypes.

        Args:
            embeddings (np.ndarray[float]): The input embeddings with shape (n_samples, n_features).
            labels (np.ndarray[str]): The labels associated with each embedding.
            paths (np.ndarray[str], Optional): The image paths associated with each embedding. Defaults to None.
        Returns:
            Tuple[np.ndarray[float], np.ndarray[str]]: 
                - The UMAP embeddings after dimensionality reduction with shape (n_mutliplexed_samples, n_components).
                - The corresponding phenotypes labels preserving the association with the UMAP embeddings.
                - A dictionary with 'ari' as key and the ari score as value
                - The corresponding paths preserving the association with the UMAP embeddings.
        """
        
        multiplexed_embeddings, multiplexed_labels, multiplexed_paths = self.analyzer_multiplex_markers.calculate(embeddings, labels, paths)

        logging.info(f"[AnalyzerUMAPMultiplexMarkers.calculate] Calculating UMAP of multiplex embeddings")
        umap_embeddings = self._compute_umap_embeddings(multiplexed_embeddings)     
        
        if self.data_config.SHOW_ARI:
            multiplexed_labels_for_ari = map_labels(multiplexed_labels, self.data_config, self.data_config, config_function_name='ARI_LABELS_FUNC')
            ari = self._compute_ari(multiplexed_embeddings, multiplexed_labels_for_ari)
            ari_score = {'ari':ari}

        else:
            ari_score = {}

        self.features = umap_embeddings
        self.labels = multiplexed_labels
        self.ari_scores = ari_score
        self.paths = multiplexed_paths

        return umap_embeddings, multiplexed_labels, multiplexed_paths, ari_score
