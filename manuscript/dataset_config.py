import os

import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.datasets.dataset_config import DatasetConfig

############################################################
# Neurons
############################################################        

class B78DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7", "batch8"]]

        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI','FMRP']

############################################################
# deltaNLS
############################################################        

class deltaNLSDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", 'deltaNLS', f) for f in
                        ["batch3", "batch4", "batch5"]]
        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['DAPI']
        
############################################################
# OpenCell
############################################################        

class OpenCellDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["OpenCell"]]

        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = ['DAPI']

        ######################
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################
