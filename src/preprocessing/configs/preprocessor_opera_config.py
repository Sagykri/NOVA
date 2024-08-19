import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))



from src.common.configs.preprocessing_config import PreprocessingConfig

class OperaPreprocessingConfig(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.RAW_SUBFOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'Opera_sorted')
        self.PROCESSED_SUBFOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera")
        self.OUTPUTS_SUBSUBFOLDER = os.path.join(self.OUTPUTS_SUBFOLDER, "Opera")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_spd", "SPDPreprocessor")
        self.TO_DOWNSAMPLE = False
        self.EXPECTED_SITE_WIDTH = 1024
        self.EXPECTED_SITE_HEIGHT = 1024
        self.TILE_WIDTH = 128 
        self.TILE_HEIGHT = 128
        self.WITH_NUCLEUS_DISTANCE = False
        self.CELL_LINES_TO_INCLUDE = None
        self.BRENNER_BOUNDS_PATH =  None
        self.DELETE_MARKER_FOLDER_IF_EXISTS = False