import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.preprocessing.configs.preprocessor_spd_dnls_config import SPDPreprocessingConfigdNLS


class SPD_Batch4dnls(SPDPreprocessingConfigdNLS):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "deltaNLS_sort", "batch4")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "deltaNLS","batch4")]


class SPD_Batch4dnlsNODS(SPDPreprocessingConfigdNLS):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "deltaNLS_sort", "batch4")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT,"deltaNLS", "batch4_16bit_no_downsample")]
        self.TO_DOWNSAMPLE = False
        self.TILE_WIDTH = 128 
        self.TILE_HEIGHT = 128
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_SUBSUBFOLDER,'logs','deltaNLS', "no_downsample")