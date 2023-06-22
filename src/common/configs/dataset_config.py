import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.base_config import BaseConfig


class DatasetConfig(BaseConfig):
    def __init__(self):
        
        super().__init__()
        
        self.DATASETS_HOME_FOLDER = os.path.join(self.HOME_FOLDER, "src", "datasets")
        
        
        
        self.MARKERS = None

        self.MARKERS_TO_EXCLUDE = ['DAPI'] #, 'lysotracker', 'Syto12']
        self.CELL_LINES = None
        self.CONDITIONS = None
        self.SPLIT_DATA = True
        self.DATA_SET_TYPE = None #'train'
        self.MARKERS_FOR_DOWNSAMPLE = None
        self.TRAIN_PCT = 0.7
        self.SHUFFLE = True
        self.ADD_CONDITION_TO_LABEL = True 
        self.ADD_LINE_TO_LABEL = True
        self.ADD_TYPE_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.SPLIT_BY_SET_FOR = None
        self.SPLIT_BY_SET_FOR_BATCH = None
        
        
        ###################################
        # TODO: TO DELETE
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 16 # ~8 tiles site * 32 sites per batch = 256 tiles per batch
        self.MAX_EPOCH = 100
        
        self.HOME_SUBFOLDER = os.path.join(self.HOME_FOLDER, "src", "models", "neuroself")
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.HOME_SUBFOLDER, 'models_outputs')
        
        ########################################
        
        self.AUG_TO_FLIP = True
        self.AUG_TO_ROT = True
