import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingConfig


class SPD_Batch3(SPDPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(".", "data", "raw", "SpinningDisk", "batch3")]
        self.OUTPUT_FOLDERS = [os.path.join(".", "data", "processed", "spd2", "SpinningDisk", "batch3")]
        print( self.INPUT_FOLDERS)
        print(self.OUTPUT_FOLDERS)
        if os.name == 'nt': # ie windows - so cut .\\
            self.INPUT_FOLDERS = list([self.INPUT_FOLDERS[0][2:]])
            self.OUTPUT_FOLDERS = list([self.OUTPUT_FOLDERS[0][2:]])
