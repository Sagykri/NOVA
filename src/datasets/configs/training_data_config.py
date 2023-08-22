import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.dataset_config import DatasetConfig


class TrainDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8_16bit"]]

        self.SPLIT_DATA = True

        ######################
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################
        
        
class B78TrainDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7_16bit", "batch8_16bit"]]

        self.SPLIT_DATA = True

class B78NoDSTrainDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7_16bit_no_downsample", "batch8_16bit_no_downsample"]]

        self.SPLIT_DATA = True
        
        self.CONDITIONS = ['Untreated']

class OpenCellTrainDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["OpenCell"]]

        self.SPLIT_DATA = True
        self.IS_AUG_INPLACE = True

        ######################
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################