import os
from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingConfig


class SPD_Batch5(SPDPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.input_folders = [os.path.join(".", "data", "raw", "SpinningDisk", "batch5")]
        self.output_folders = [os.path.join(".", "data", "processed", "spd2", "SpinningDisk", "batch5")]