
from abc import abstractmethod

import sys
import os
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig
import numpy as np
import pandas as pd
from typing import Union, Tuple

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

    @abstractmethod
    def _get_saving_folder(self)->str:
        """Get the path to the folder where the features and figures can be saved
        """
        pass

