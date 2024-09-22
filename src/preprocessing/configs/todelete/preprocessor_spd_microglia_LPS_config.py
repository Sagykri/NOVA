import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))



from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingBaseConfig

class SPDPreprocessingConfigMicrogliaLPS(SPDPreprocessingBaseConfig):
    def __init__(self):
        super().__init__()
        
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_SUBSUBFOLDER, 'logs', 'microglia_LPS')
