import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.base_config import BaseConfig


class DatasetConfig(BaseConfig):
    def __init__(self):
        
        super().__init__()
        
        self.DATASETS_HOME_FOLDER = os.path.join(self.HOME_FOLDER, "src", "datasets")
        
        
        
        self.MARKERS = None

        self.MARKERS_TO_EXCLUDE = ['TIA1'] #, 'lysotracker', 'Syto12']
        self.CELL_LINES = None
        self.CONDITIONS = None
        self.REPS       = None
        self.SPLIT_DATA = True
        self.DATA_SET_TYPE = None # 'train' | 'test' | 'val'
        self.MARKERS_FOR_DOWNSAMPLE = None
        self.TRAIN_PCT = 0.7
        self.SHUFFLE = True #Must be true whenever using SPLIT_DATA=True otherwise train,val,test set won't be the same as when Shuffle was true
        self.ADD_CONDITION_TO_LABEL = True 
        self.ADD_LINE_TO_LABEL = True
        self.ADD_TYPE_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        self.ADD_REP_TO_LABEL = False
        
        self.SPLIT_BY_SET_FOR = None
        self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.AUG_TO_FLIP = True
        self.AUG_TO_ROT = True
        self.IS_AUG_INPLACE = False
        
        self.SPLIT_LABELS = False

        # How much percentage to sample from the dataset. Set to 1 or None for taking all dataset.
        # Valid values are: 0<SAMPLE_PCT<=1 or SAMPLE_PCT=None (identical to SAMPLE_PCT=1)
        self.SAMPLE_PCT = 1

        self.TRAIN_BATCHES = []
