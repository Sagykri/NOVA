import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingConfig


class SPD_Batch7(SPDPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "batch7")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "batch7_16bit")]
        self.WITH_NUCLEUS_DISTANCE = False

class SPD_Batch7NODS(SPDPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "batch7")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "batch7_16bit_no_downsample")]
        self.TO_DOWNSAMPLE = False
        self.TILE_WIDTH = 128 
        self.TILE_HEIGHT = 128
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_SUBSUBFOLDER,'logs',"no_downsample")
        
class SPD_Batch7New(SPDPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "batch7")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "batch7")]
        self.TO_DOWNSAMPLE = False
        self.TILE_WIDTH = 100
        self.TILE_HEIGHT = 100
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_SUBSUBFOLDER, 'logs', "preprocessing_Dec2023")        