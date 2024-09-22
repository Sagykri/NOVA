import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.preprocessing.configs.preprocessor_spd_dnls_config import SPDPreprocessingConfigdNLS


class SPD_Batch5dnls_Test(SPDPreprocessingConfigdNLS):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "deltaNLS_sort", "batch5")]
        self.OUTPUT_FOLDERS = None#[os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "batch6")]
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_SUBSUBFOLDER, 'logs', "testing_180924", 'dNLS_batch5_test')
