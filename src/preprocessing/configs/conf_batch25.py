import os
from preprocessing.configs.preprocessor_conf_config import ConfPreprocessingConfig

class Conf_Batch25(ConfPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.__folders = ["batch_2_5"]
        
        self.input_folders = [os.path.join(".", "data", "raw", f) for f in self.__folders]
        self.output_folders = [os.path.join(".", "data", "processed", f) for f in self.__folders]