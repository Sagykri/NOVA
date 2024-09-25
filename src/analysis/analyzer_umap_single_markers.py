import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.datasets.label_utils import get_markers_from_labels
from src.datasets.dataset_config import DatasetConfig
from src.analysis.analyzer_umap import AnalyzerUMAP
from src.datasets.label_utils import get_unique_parts_from_labels, get_markers_from_labels

import logging
import numpy as np
from typing import Tuple

class AnalyzerUMAPSingleMarkers(AnalyzerUMAP):
    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        super().__init__(data_config, output_folder_path)


    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str])->Tuple[np.ndarray[float],np.ndarray[str]]:
        """Calculate UMAP embeddings separately for each marker in the given embeddings 
            and store the results in the `self.features` attribute. For each unique marker, the function extracts the corresponding embeddings and 
            labels, computes the UMAP embeddings, and concatenates them into a final result.

        Args:
            embeddings (np.ndarray[float]): The input embeddings with shape (n_samples, n_features).
            labels (np.ndarray[str]): The labels associated with each embedding. These labels are used 
                to group embeddings by marker.
        Returns:
            Tuple[np.ndarray[float], np.ndarray[str], Dict[str,float]]: 
                - The UMAP embeddings after dimensionality reduction with shape (n_samples, n_components).
                - The corresponding labels after concatenation, preserving the association with the UMAP embeddings.
                - Dictionary with marker as keys and ari scores as values
        """

        unique_markers = get_unique_parts_from_labels(labels, get_markers_from_labels) 
        logging.info(f"[AnalyzerUMAPSingleMarkers.calculate] Detected markers: {unique_markers}")
        marker_of_labels = get_markers_from_labels(labels)
        umap_embeddings, umap_labels = [], []
        ari_scores = {}
        for marker in unique_markers:
            logging.info(f"Marker: {marker}")
            indices = np.where(marker_of_labels == marker)[0]
            logging.info(f"{len(indices)} indexes have been selected")

            if len(indices) == 0:
                logging.info(f"No data for marker {marker}, skipping.")
                continue

            marker_embeddings, marker_labels = embeddings[indices], labels[indices]
            marker_umap_embeddings = self._compute_umap_embeddings(marker_embeddings)
            umap_embeddings.append(marker_umap_embeddings)
            umap_labels.append(marker_labels)

            if self.data_config.SHOW_ARI:
                # If we want to change the ari to be calculated on the embeddings themself and not the umap, just send here the embeddings of the marker
                ari = self._compute_ari(marker_embeddings, marker_labels)
                ari_scores[marker] = ari

        umap_embeddings = np.concatenate(umap_embeddings)
        umap_labels = np.concatenate(umap_labels)
        self.features = umap_embeddings
        self.labels = umap_labels
        self.ari_scores = ari_scores

        return umap_embeddings, umap_labels, ari_scores