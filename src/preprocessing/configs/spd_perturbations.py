import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingConfig


class SPD_Perturbations(SPDPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(".", "input", "images", "raw", "SpinningDisk", "Perturbations")]
        self.OUTPUT_FOLDERS = [os.path.join(".", "input", "images", "processed", "spd2", "SpinningDisk", "Perturbations")]