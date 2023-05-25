import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingConfig

class SPD_Test(SPDPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        __folders = ["test_spd"]
        
        home_subfolder = os.path.join(self.HOME_FOLDER, "tests", "test_preprocessing")
        
        self.INPUT_FOLDERS = [os.path.join(home_subfolder, "input", "images", "raw", f) for f in __folders]
        self.OUTPUT_FOLDERS = [os.path.join(home_subfolder, "input", "images", "processed", f) for f in __folders]
        
        self.LOGS_FOLDER = os.path.join(home_subfolder, 'logs')
        
        # self.TO_SHOW = True