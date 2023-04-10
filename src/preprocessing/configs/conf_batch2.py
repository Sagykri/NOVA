import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.preprocessing.configs.preprocessor_conf_config import ConfPreprocessingConfig

class Conf_Batch2(ConfPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        __folders = ["220814_neurons",
                        "220818_neurons",
                        "220831_neurons",
                        "220908", "220914"]
        
        self.INPUT_FOLDERS = [os.path.join(".", "data", "raw", f) for f in __folders]
        self.OUTPUT_FOLDERS = [os.path.join(".", "data", "processed", f) for f in __folders]