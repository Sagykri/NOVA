import sys
import os
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.datasets.label_utils import get_markers_from_labels
from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig
from src.analysis.analyzer_umap import AnalyzerUMAP
from src.datasets.label_utils import get_unique_parts_from_labels, get_markers_from_labels

import logging
import numpy as np
from typing import Tuple

class AnalyzerUMAP0(AnalyzerUMAP):
    def __init__(self, trainer_config: TrainerConfig, data_config: DatasetConfig):
        super().__init__(trainer_config, data_config)


    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str])->Tuple[np.ndarray[float],np.ndarray[str]]:
        """Calculate UMAP embeddings separately for each marker in the given embeddings 
            and store the results in the `self.features` attribute. For each unique marker, the function extracts the corresponding embeddings and 
            labels, computes the UMAP embeddings, and concatenates them into a final result.

        Args:
            embeddings (np.ndarray[float]): The input embeddings with shape (n_samples, n_features).
            labels (np.ndarray[str]): The labels associated with each embedding. These labels are used 
                to group embeddings by marker.
        Returns:
            Tuple[np.ndarray[float], np.ndarray[str]]: 
                - The UMAP embeddings after dimensionality reduction with shape (n_samples, n_components).
                - The corresponding labels after concatenation, preserving the association with the UMAP embeddings.
        """

        markers = get_unique_parts_from_labels(labels, get_markers_from_labels) 
        logging.info(f"[AnalyzerUMAP0.calculate] Detected markers: {markers}")
        umap_embeddings = None
        for marker in markers:
            logging.info(f"Marker: {marker}")
            marker_of_labels = get_markers_from_labels(labels)
            indices = np.where(marker_of_labels == marker)[0]
            logging.info(f"{len(indices)} indexes have been selected")

            if len(indices) == 0:
                logging.info(f"No data for marker {marker}, skipping.")
                continue

            marker_embeddings, marker_labels = embeddings[indices], labels[indices]
            marker_umap_embeddings = self._compute_umap_embeddings(marker_embeddings)
            if umap_embeddings is None:
                umap_embeddings = marker_umap_embeddings
                umap_labels = marker_labels
            else:
                umap_embeddings = np.concatenate([umap_embeddings, marker_umap_embeddings])
                umap_labels = np.concatenate([umap_labels, marker_labels])
        self.features = umap_embeddings
        self.labels = umap_labels

        return umap_embeddings, umap_labels