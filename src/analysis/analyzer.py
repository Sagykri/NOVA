
from abc import abstractmethod

import sys
import os
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig
from src.datasets.label_utils import get_batches_from_input_folders
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional

class Analyzer():
    """Analyzer class is used to analyze the embeddings of a model.
    It's three main methods are:
    1. calculate(): to calculate the wanted features from the embeddings, such as UMAP or distances.
    2. load(): to load the calculated features that were previusly saved.
    3. save(): to save the calculated features.
    """
    def __init__(self, trainer_config: TrainerConfig, data_config: DatasetConfig):
        """Get an instance

        Args:
            trainer_config (TrainerConfig): The trainer config object.
            data_config (DatasetConfig): The dataset config object. 
        """
        self.__set_params(trainer_config, data_config)
        self.features:np.ndarray = None

    def __set_params(self, trainer_config: TrainerConfig, data_config: DatasetConfig)->None:
        """Extracting params from the configuration

        Args:
            trainer_config (TrainerConfig): trainer configuration
            data_config (DatasetConfig): data configuration
        """       
        self.data_config = data_config
        self.output_folder_path = trainer_config.OUTPUTS_FOLDER

    @abstractmethod
    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str])->Union[pd.DataFrame,Tuple[np.ndarray[float],np.ndarray[str]]]:
        """Calculate features from given embeddings, save in the self.features attribute and return them as well

        Args:
            embeddings (np.ndarray[float]): The embeddings
            labels (np.ndarray[str]): The corresponding labels of the embeddings
        Return:
            The calculated features
        """
        pass

    @abstractmethod
    def load(self)->None:
        """load the saved features into the self.features attribute
        """
        pass

    @abstractmethod
    def save(self)->None:
        """save the calculated features in path derived from self.output_folder_path
        """
        pass

    def _get_saving_folder(self, feature_type:str, umap_type:Optional[str]='')->str:
        """Get the path to the folder where the features and figures can be saved
        Args:
            feature_type (str): string indicating the feature type ('distances','UMAP')
            umap_type (str): string indicating the umap type ('umap0','umap1','umap2'), optional (default is None)
        """
        model_output_folder = self.output_folder_path
        feature_folder_path = os.path.join(model_output_folder, 'figures', self.data_config.EXPERIMENT_TYPE, feature_type, umap_type)
        os.makedirs(feature_folder_path, exist_ok=True)
        
        input_folders = get_batches_from_input_folders(self.data_config.INPUT_FOLDERS)
        reps = self.data_config.REPS if self.data_config.REPS else ['all_reps']
        cell_lines = self.data_config.CELL_LINES if self.data_config.CELL_LINES else ["all_cell_lines"]
        conditions = self.data_config.CONDITIONS if self.data_config.CONDITIONS else ["all_conditions"]
        title = f"{'_'.join(input_folders)}_{'_'.join(reps)}_{'_'.join(cell_lines)}_{'_'.join(conditions)}"
        saveroot = os.path.join(feature_folder_path,f'{title}')
        # saveroot = feature_folder_path
        os.makedirs(saveroot, exist_ok=True)

        return saveroot

