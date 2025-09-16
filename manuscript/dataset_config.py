import os

import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.datasets.dataset_config import DatasetConfig

############################################################
# Neurons
############################################################        

class B56789DatasetConfig_80pct(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8", f) for f in 
                        ["batch5", "batch6", "batch7", "batch8", "batch9"]]

        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']

############################################################
# OpenCell
############################################################        

class OpenCellDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "OpenCell", f) for f in 
                        ["batch1"]]

        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = ['DAPI']

        ######################
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################
