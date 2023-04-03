import os
from common.configs.model_config import ModelConfig


class CytoselfConfig(ModelConfig):
    def __init__(self):
        super(CytoselfConfig, self).__init__()
                
        self.HOME_SUBFOLDER = os.path.join(self.HOME_FOLDER, "cytoself")
        self.INPUT_FOLDERS = os.path.join(self.PROCESSED_FOLDER_ROOT, "220714") 
        self.LOGS_FOLDER = os.path.join(self.HOME_SUBFOLDER, 'logs')
        
