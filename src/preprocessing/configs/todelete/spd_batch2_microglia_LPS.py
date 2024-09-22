import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.preprocessing.configs.preprocessor_spd_microglia_LPS_config import SPDPreprocessingConfigMicrogliaLPS


class SPD_Batch2MicrogliaLPS(SPDPreprocessingConfigMicrogliaLPS):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "microglia_LPS_sort", "batch2")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "microglia_LPS","batch2")]      