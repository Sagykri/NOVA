import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.model_config import ModelConfig


class CytoselfConfig(ModelConfig):
    def __init__(self):
        super().__init__()
                
        self.HOME_SUBFOLDER = os.path.join(self.MODELS_HOME_FOLDER, "cytoself")
        self.INPUT_FOLDERS = os.path.join(self.PROCESSED_FOLDER_ROOT, "220714") 
        self.LOGS_FOLDER = os.path.join(self.HOME_SUBFOLDER, 'logs')
        
