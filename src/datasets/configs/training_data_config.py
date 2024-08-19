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
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['DAPI']

        ######################
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################

############################################################
# Neurons
############################################################        
                
class B78TrainDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7_16bit", "batch8_16bit"]]

        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['DAPI']


class B78NoDSTrainDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7", "batch8"]]

        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']
        # self.MARKERS = ['G3BP1','PML','FUS','PURA']
# class B78NoDSTrainDatasetConfig(DatasetConfig):
#     def __init__(self):
#         super().__init__()

#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
#                         ["batch7_16bit_no_downsample", "batch8_16bit_no_downsample"]]

#         self.SPLIT_DATA = True
#         self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['DAPI']

############################################################
# NiemannPick
############################################################        

class NiemannPickB14TrainDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", 'NiemannPick', f) for f in
                        ["batch1_16bit_no_downsample", "batch4_16bit_no_downsample"]]
        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['DAPI']

############################################################
# deltaNLS
############################################################        

class deltaNLSB25TrainDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", 'deltaNLS', f) for f in
                        ["batch2_16bit_no_downsample", "batch5_16bit_no_downsample"]]
        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['DAPI']
        
############################################################
# OpenCell
############################################################        

class OpenCellTrainDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["OpenCell"]]

        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['DAPI']
        # self.IS_AUG_INPLACE = True

        ######################
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################

class deltaNLSB25TrainDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", 'deltaNLS', f) for f in 
                        ["batch2_16bit_no_downsample", "batch5_16bit_no_downsample"]]
        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['DAPI']
