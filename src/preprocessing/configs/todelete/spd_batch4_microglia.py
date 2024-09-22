import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.preprocessing.configs.preprocessor_spd_microglia_config import SPDPreprocessingConfigMicroglia


class SPD_Batch4Microglia(SPDPreprocessingConfigMicroglia):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "microglia_sort", "batch4")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "microglia","batch4")]      