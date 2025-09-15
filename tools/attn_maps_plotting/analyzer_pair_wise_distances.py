import sys
import os

sys.path.insert(1, os.getenv("NOVA_HOME"))

from abc import abstractmethod
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Iterable, Dict
import itertools

from src.datasets.label_utils import get_unique_parts_from_labels, get_cell_lines_conditions_from_labels, get_markers_from_labels, get_batches_from_labels
from src.datasets.dataset_config import DatasetConfig
from src.analysis.analyzer import Analyzer
from src.common.utils import get_if_exists
from src.datasets.label_utils import get_batches_from_labels, get_unique_parts_from_labels, get_markers_from_labels
from tools.attn_maps_plotting.analyzer_pairwise_dist_utils import filter_by_labels, compute_distances, extract_pairs


class AnalyzerPairwiseDistances(Analyzer):
    """
    AnalyzerPairwiseDistances is responsible for calculating distances between t types of embedding conditions (for example: control vs. stress). 

    """
    def __init__(self, data_config: DatasetConfig, pairwise_config, output_folder_path:str):
        """Get an instance

        Args:
            data_config (DatasetConfig): The dataset configuration object. 
            output_folder_path (str): path to output folder
        """
        super().__init__(data_config, output_folder_path)
        self.pairwise_config = pairwise_config


    def calculate(self, embeddings: np.ndarray, labels: np.ndarray, paths: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate features from given embeddings, save in the self.features attribute and return them.

        Args:
            embeddings (np.ndarray): embedding vector for each sample
            labels (np.ndarray): The corresponding labels of embeddings
            paths (np.ndarray): The corresponding paths of embeddings

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping marker name to raw pairwise distances.
        """
        self.raw_distances: Dict[str, np.ndarray] = {}
        self.pairs_df: Dict[str, pd.DataFrame] = {}

        #marker_names = get_unique_parts_from_labels(labels, get_markers_from_labels, self.pairwise_config)
        grouped_labels_by_markers = get_markers_from_labels(labels, self.data_config)
        self.marker_names = np.unique(grouped_labels_by_markers)
        for marker in self.marker_names:
            # Filter by marker
            mask = grouped_labels_by_markers == marker
            marker_labels = labels[mask]
            marker_embeddings = embeddings[mask]
            marker_paths = paths[mask]

            # Compute distances
            marker_distances, unique_conditions, paths_c1, paths_c2 = compute_distances(
                marker_embeddings, marker_labels, marker_paths, self.pairwise_config.METRIC, self.data_config
            )
            self.raw_distances[marker] = marker_distances

            # Extract subset DataFrame
            distances_df = extract_pairs(
                marker_distances, unique_conditions, paths_c1, paths_c2, self.pairwise_config
            )
            #"pair_type","{config.METRIC}_distance,"path_{unique_conditions[0]}","path_{unique_conditions[1]}"
            self.pairs_df[marker] = distances_df

        return self.pairs_df

    def load(self) -> None:
        """
        Load the saved features (raw_distances and pairs_df) from disk into the corresponding attributes.
        Stacks data per set into dictionaries indexed by marker names.
        """

        output_folder_path = self.get_saving_folder(feature_type='pairwise_distances', main_folder='figures')
        self.raw_distances = {}
        self.pairs_df = {}

        logging.info(f"Loading scores from {output_folder_path}")

        for marker in self.marker_names:
            raw_dist_path = os.path.join(output_folder_path, f"{marker}_raw_distances.npy")
            pairs_df_path = os.path.join(output_folder_path, f"{marker}_pairs.csv")

            if os.path.exists(raw_dist_path):
                self.raw_distances[marker] = np.load(raw_dist_path)
            else:
                logging.warning(f"Missing file: {raw_dist_path}")

            if os.path.exists(pairs_df_path):
                self.pairs_df[marker] = pd.read_csv(pairs_df_path)
            else:
                logging.warning(f"Missing file: {pairs_df_path}")
        
        return None


    def save(self)->None:
        """"
        Save the calculated distances to a specified file.
        """
        output_folder_path = self.get_saving_folder(feature_type='pairwise_distances', main_folder = 'figures')
        os.makedirs(output_folder_path, exist_ok=True)
        logging.info(f"Saving scores to {output_folder_path}")

        for marker in self.marker_names:
            np.save(os.path.join(output_folder_path, f"{marker}_raw_distances.npy"), self.raw_distances[marker])
            self.pairs_df[marker].to_csv(os.path.join(output_folder_path, f"{marker}_pairs.csv"), index=False)
            
            # save all the pairwise subset paths
            path_columns = [col for col in self.pairs_df[marker].columns if col.startswith("path_")]
            all_paths = self.pairs_df[marker][path_columns].values.flatten().tolist()
            np.save(os.path.join(output_folder_path, f"{marker}_paths.npy"), all_paths)
        
        return None

    @abstractmethod    
    def _compute_score(self, embeddings: np.ndarray[float], labels: np.ndarray[str]) -> Tuple[float,str]:
        """
        Abstract method to compute the score between two sets of embeddings.

        Args:
            embeddings (np.ndarray[float]): The embeddings to compute scores on.
            labels (np.ndarray[str]): Corresponding labels; should contain only 2 unique labels.

        Returns:
            float: The calculated score.
            str: Name of the score metric.
        """
        pass
    
    

    def _get_save_path(self, output_folder_path:str, file_type:str)->str: #TODO:ask sagy where to save
        
        savepath = os.path.join(output_folder_path, f"{file_type}.npy")
        return savepath

    