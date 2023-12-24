import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.preprocessing.configs.preprocessor_spd_dnls_config import SPDPreprocessingConfigdNLS


class SPD_Batch3dnls(SPDPreprocessingConfigdNLS):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "deltaNLS_sort", "batch3")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "deltaNLS","batch3")]
        
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_SUBSUBFOLDER, 'logs', "preprocessing_Dec2023", "dNLS", 'batch3')        
