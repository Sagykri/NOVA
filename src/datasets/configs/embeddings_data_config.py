import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.dataset_config import DatasetConfig

class EmbeddingsB6DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SPLIT_DATA = False        
        #self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        #self.CONDITIONS = ['Untreated']
        #self.MARKERS = ['TOMM20','mitotracker'] #['FUS']
        
class EmbeddingsB9DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9_16bit_no_downsample"]]
        
        self.SPLIT_DATA = False 
        self.REPS = ['rep1', 'rep2']
        self.CELL_LINES = ['WT']#, 'TDP43', 'FUSRevertant'] # 'FUSHomozygous', 'FUSHeterozygous', 
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['TOMM20','mitotracker', 'FUS'] #[]

class EmbeddingsB78DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7_16bit", "batch8_16bit"]]
        
        self.SPLIT_DATA = False        
        self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['TOMM20', 'mitotracker', 'FUS'] 