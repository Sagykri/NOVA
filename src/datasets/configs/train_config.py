import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.dataset_config import DatasetConfig


class TrainDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################
        
        
        
class TrainB7DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        #self.CELL_LINES = ['WT', 'TDP43', 'TBK1', 'FUSHomozygous', 'FUSRevertant', 'SCNA']
        
        #self.CELL_LINES = ['WT']
        #self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['G3BP1', 'FMRP', 'PURA', 'DCP1A']
        ######################
        
        
        
class TrainB8PiecewiseDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8_linear_piecewise"]]
        # self.INPUT_FOLDERS = ["/home/labs/hornsteinlab/Collaboration/MOmaps_Ilan2/MOmaps/input/images/processed/spd2/SpinningDisk/batch8_linear_piecewise"]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['G3BP1', 'FMRP', 'PURA', 'DCP1A']
        ######################
        
class TrainB8PiecewiseAlsoStressDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8_linear_piecewise"]]
        # self.INPUT_FOLDERS = ["/home/labs/hornsteinlab/Collaboration/MOmaps_Ilan2/MOmaps/input/images/processed/spd2/SpinningDisk/batch8_linear_piecewise"]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        # self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['G3BP1', 'FMRP', 'PURA', 'DCP1A']
        ######################
          
class TrainB8STAlsoStressDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8_testCV2"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        # self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['G3BP1', 'FMRP', 'PURA', 'DCP1A']
        ######################
        
        
        
class TrainWT6789DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6", "batch7", "batch8", "batch9"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################
        
        
        
class TrainALL6789DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6", "batch7", "batch8", "batch9"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        # self.CELL_LINES = ['WT']
        # self.CONDITIONS = ['Untreated']
        ######################
        
        
        
class TrainBatch8NoAugDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################
        
        self.AUG_TO_FLIP = False
        self.AUG_TO_ROT = False
        
        

class TrainStressDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        #self.CONDITIONS = ['Untreated']
        ######################

class TrainB8SmallDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['G3BP1', 'FMRP', 'PURA']
        ######################        
        
class TrainB6789SmallDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6", "batch7", "batch8", "batch9"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['G3BP1', 'FMRP', 'PURA']
        ######################       

class TrainB8AllDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        #self.CONDITIONS = ['Untreated']
        ######################
        
class TrainB8_WT_TDP_DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        self.CELL_LINES = ['WT', 'TDP43']
        self.CONDITIONS = ['Untreated']
        ######################

class TrainB8_TBK1_TDP_DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        self.CELL_LINES = ['TBK1', 'TDP43']
        self.CONDITIONS = ['Untreated']
        ######################
        
class TrainB8_WT_DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################       
        
class TrainALLBatchesDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8", "batch9"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        # self.CELL_LINES = ['WT']
        #self.CONDITIONS = ['Untreated']
        ######################
        
        
        
class TrainBatches678DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6", "batch7", "batch8"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################
        
       
        
        
        
class TrainOpenCellDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["OpenCell"]]

        self.SPLIT_DATA = True
        self.IS_AUG_INPLACE = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################
        
class TrainDatasetConfigBatch8_norm_16b_even(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8_norm_16b_even"]]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################