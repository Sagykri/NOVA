
from abc import abstractmethod

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig
import numpy as np

class Analyzer():
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        self.__set_params(trainer_conf, data_conf)
        self.features:np.ndarray = None

    def __set_params(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig)->None:
        """Extracting params from the configuration

        Args:
            trainer_conf (TrainerConfig): trainer configuration
            data_conf (DatasetConfig): data configuration
        """
        self.input_folders = data_conf.INPUT_FOLDERS
        self.experiment_type = data_conf.EXPERIMENT_TYPE
        self.reps = data_conf.REPS
        self.markers = data_conf.MARKERS
        self.markers_to_exclude = data_conf.MARKERS_TO_EXCLUDE
        self.cell_lines = data_conf.CELL_LINES
        self.conditions = data_conf.CONDITIONS
        self.train_part = data_conf.TRAIN_PCT
        self.shuffle = data_conf.SHUFFLE

        self.to_split_data = data_conf.SPLIT_DATA
        self.data_set_type = data_conf.DATA_SET_TYPE
        
        self.data_conf = data_conf

        self.output_folder_path = trainer_conf.OUTPUTS_FOLDER

    @abstractmethod
    def calculate(self, embeddings:np.ndarray, labels:np.ndarray)->None:
        """Calculate features from given embeddings, save in the self.features attribute

        Args:
            embeddings (np.ndarray): The embeddings
            labels (np.ndarray): The corresponding labels of the embeddings
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

