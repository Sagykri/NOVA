import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.dataset_config import DatasetConfig


class TrainBatch2DLDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch2"]]
        
        self.SPLIT_DATA = True

        self.MARKERS_TO_EXCLUDE = ['DAPI', 'lysotracker', 'Syto12']
        
        ######################
        # TODO: Just for testing on less classes, remove afterwards
        self.CELL_LINES = ['WT']#, 'TDP43', 'TBK1']
        self.CONDITIONS = ['unstressed']
        # self.MARKERS = ["TIA1", "G3BP1", "PURA"] 
        
        # TESTING WITHOUT AUGMENTATIONS
        # self.AUG_TO_FLIP = True
        # self.AUG_TO_ROT = True
        ######################
        
class TrainBatch2DLALLDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch2"]]
        
        self.SPLIT_DATA = True

        self.MARKERS_TO_EXCLUDE = ['DAPI', 'lysotracker', 'Syto12']
        
        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT']#, 'TDP43', 'TBK1']
        # self.CONDITIONS = ['unstressed']
        # self.MARKERS = ["TIA1", "G3BP1", "PURA"] 
        
        # TESTING WITHOUT AUGMENTATIONS
        # self.AUG_TO_FLIP = True
        # self.AUG_TO_ROT = True
        ######################