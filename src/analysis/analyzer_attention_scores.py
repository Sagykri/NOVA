import sys
import os

sys.path.insert(1, os.getenv("NOVA_HOME"))

from abc import abstractmethod
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Iterable, Dict
import itertools
import torch

from src.datasets.label_utils import get_unique_parts_from_labels, get_cell_lines_conditions_from_labels,\
                                         get_markers_from_labels, get_batches_from_labels, get_batches_from_input_folders
from src.datasets.dataset_config import DatasetConfig
from src.analysis.analyzer import Analyzer
from src.common.utils import get_if_exists
from src.analysis.analyzer_attn_scores_utils import compute_attn_correlations
from src.analysis.attention_scores_config import AttnScoresBaseConfig

class AnalyzerAttnScore(Analyzer):
    """
    AnalyzerAttnScore is responsible for calculating correlation scores between attention maps and their corresponding input images. 
    The correlation scores are computed for each marker and batch.
    """
    def __init__(self, data_config: DatasetConfig, output_folder_path:str, corr_config:AttnScoresBaseConfig):
        """Get an instance

        Args:
            data_config (DatasetConfig): The dataset configuration object. 
            output_folder_path (str): path to output folder
        """
        super().__init__(data_config, output_folder_path)
        self.corr_config = corr_config
        self.corr_method = corr_config.CORR_METHOD

    def calculate(self, processed_attn_maps:np.ndarray[float], labels:np.ndarray[str], paths: np.ndarray[str])->List[np.ndarray[torch.Tensor]]:
        """Calculate features from given embeddings, save in the self.features attribute and return them as well

        Args:
            processed_attn_maps (np.ndarray[float]): The processed attention maps, already in the img shape (H,W)
            labels (np.ndarray[str]): The corresponding labels of attention maps
            paths (np.ndarray[str]): The corresponding paths of attention maps
        Return:
            The calculated correlation data
        """
        corr_data = compute_attn_correlations(processed_attn_maps, labels, paths, data_config = self.data_config, corr_config= self.corr_config)
        self.features = corr_data
        self.labels = labels
        self.paths = paths
        return corr_data
    
    def load(self) -> None:
        """
        Load the saved features, labels, and paths into the corresponding attributes.
        Stacks data per set into arrays with shape (num_sets, ...).
        """
        output_folder_path = self.get_saving_folder(feature_type='attn_scores')
        logging.info(f"[load scores]: output_folder_path: {output_folder_path}")

        if self.data_config.SPLIT_DATA:
            data_set_types = ['trainset', 'valset', 'testset']
        else:
            data_set_types = ['testset']

        features = []
        labels = []
        paths = []

        for set_type in data_set_types:
            logging.info(f"[AnalyzerAttnScores] loading from: {output_folder_path}")
            features.append(np.load(self._get_save_path(output_folder_path, "corrs", set_type)))
            labels.append(np.load(self._get_save_path(output_folder_path, "labels", set_type)))
            paths.append(np.load(self._get_save_path(output_folder_path, "paths", set_type)))

        self.features = features
        self.labels = labels
        self.paths = paths

        return None

    def save(self)->None:
        """"
        Save the calculated distances to a specified file.
        """
        output_folder_path = self.get_saving_folder(feature_type='attn_scores')
        os.makedirs(output_folder_path, exist_ok=True)
        logging.info(f"Saving scores to {output_folder_path}")

        if self.data_config.SPLIT_DATA:
            data_set_types = ['trainset','valset','testset']
        else:
            data_set_types = ['testset']
        
        for i, set_type in enumerate(data_set_types):
            np.save(self._get_save_path(output_folder_path, "corrs", set_type), self.features[i])
            np.save(self._get_save_path(output_folder_path, "labels",set_type), self.labels[i])
            np.save(self._get_save_path(output_folder_path, "paths", set_type), self.paths[i])
        
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
    
    def get_saving_folder(self, feature_type:str, main_folder:str = 'figures')->str:
        """Get the path to the folder where the features and figures can be saved
        Args:
            feature_type (str): string indicating the feature type ('distances','UMAP')
        """
        model_output_folder = self.output_folder_path
        feature_folder_path = os.path.join(model_output_folder, main_folder, self.data_config.EXPERIMENT_TYPE, feature_type)
        os.makedirs(feature_folder_path, exist_ok=True)
        
        input_folders = get_batches_from_input_folders(self.data_config.INPUT_FOLDERS)
        reps = self.data_config.REPS if self.data_config.REPS else ['all_reps']
        cell_lines = self.data_config.CELL_LINES if self.data_config.CELL_LINES else ["all_cell_lines"]
        conditions = self.data_config.CONDITIONS if self.data_config.CONDITIONS else ["all_conditions"]
        markers = get_if_exists(self.data_config, 'MARKERS', None)
        if markers is not None and len(markers)<=3:
            title = f"{'_'.join(input_folders)}_{'_'.join(reps)}_{'_'.join(cell_lines)}_{'_'.join(conditions)}_{'_'.join(markers)}"
        else:
            excluded_markers = self.data_config.MARKERS_TO_EXCLUDE.copy() if self.data_config.MARKERS_TO_EXCLUDE else ["all_markers"]
            if excluded_markers != ['all_markers']:
                excluded_markers.insert(0,"without")
            title = f"{'_'.join(input_folders)}_{'_'.join(reps)}_{'_'.join(cell_lines)}_{'_'.join(conditions)}_{'_'.join(excluded_markers)}"
        saveroot = os.path.join(feature_folder_path,f'{title}')
        return saveroot


    def _get_save_path(self, output_folder_path:str, file_type:str, set_type:str = "testset")->str: 
        savepath = os.path.join(output_folder_path, f"{set_type}_{self.corr_method}_{file_type}.npy")
        return savepath

    