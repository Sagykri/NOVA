import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
sys.path.insert(1,'/home/labs/hornsteinlab/Collaboration/MOmaps/') # Nancy

from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingConfig


class SPD_Batch6(SPDPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(".", "input", "images", "raw", "SpinningDisk", "batch6")]
        self.OUTPUT_FOLDERS = [os.path.join(".", "input", "images", "processed", "spd2", "SpinningDisk", "batch6")]