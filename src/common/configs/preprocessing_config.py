import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.common.configs.base_config import BaseConfig


class PreprocessingConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = None
        self.OUTPUT_FOLDERS = None
        
        # For specifing specific paths to preprocess 
        self.SELECTIVE_INPUT_PATHS = None
        
        self.HOME_SUBFOLDER = os.path.join(self.HOME_FOLDER, "src", "preprocessing")
        self.OUTPUTS_SUBFOLDER = os.path.join(self.OUTPUTS_FOLDER, "preprocessing")
        
        self.PREPROCESSOR_CLASS_PATH = None
        
        self.MARKERS_TO_INCLUDE = None
        self.TO_SHOW = False
        self.NUCLEUS_DIAMETER = 60
        self.TILE_WIDTH = 100
        self.TILE_HEIGHT = 100
        self.TO_DOWNSAMPLE = False
        self.TO_NORMALIZE = True
        self.CELLPROB_THRESHOLD = 0
        self.FLOW_THRESHOLD = 0.4
        self.TO_DENOISE = False    
        self.WITH_NUCLEUS_DISTANCE = False