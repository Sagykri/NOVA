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
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']

class B78NoRepDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7", "batch8"]]

        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']
        self.ADD_BATCH_TO_LABEL:bool = True
        self.ADD_REP_TO_LABEL:bool = False

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


############################################################
# funova
############################################################        

class FunovaDatasetConfig(DatasetConfig):
    def __init__(self):
        ## Train on batch 1 Funova ##
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1"]]

        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = ['DAPI']
        self.ADD_REP_TO_LABEL = False
        self.COMMON_BASELINE = 'Control'

class FunovaDatasetConfigHalf(DatasetConfig):
    ## Train on half of the patients, batch 1, Funova ##
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1"]]

        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = ['DAPI']
        self.ADD_REP_TO_LABEL = False
        self.COMMON_BASELINE = 'Control'
        self.CELL_LINES = ["Control-1001733","Control-1017118","C9orf72-HRE-1008566","TDP--43-G348V-1057052"]