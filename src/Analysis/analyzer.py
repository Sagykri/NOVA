
from abc import ABC, abstractmethod

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig ## TODO SAGY CHANGE


class Analyzer(ABC):
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        self.__set_params(trainer_conf, data_conf)
        self.features = None

    def __set_params(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        
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

        self.output_folder_path = trainer_conf.OUTPUT_FOLDER_PATH

    @abstractmethod
    def calculate(self, embeddings):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

