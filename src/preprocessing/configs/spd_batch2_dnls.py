import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.preprocessing.configs.preprocessor_spd_dnls_config import SPDPreprocessingConfigdNLS


class SPD_Batch2dnls(SPDPreprocessingConfigdNLS):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "deltaNLS_sort", "batch2")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "deltaNLS","batch2")]
        