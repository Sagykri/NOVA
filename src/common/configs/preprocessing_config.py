import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.base_config import BaseConfig


class PreprocessingConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = None
        self.OUTPUT_FOLDERS = None
        
        self.HOME_SUBFOLDER = os.path.join(self.HOME_FOLDER, "src", "preprocessing")
        self.LOGS_FOLDER = os.path.join(self.HOME_SUBFOLDER, 'logs')
        
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
        self.MIN_EDGE_DISTANCE = 2
        self.TO_DENOISE = False    