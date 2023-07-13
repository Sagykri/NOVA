import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.dataset_config import DatasetConfig


class Test8DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8"]]

        self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'test'
        
        # Augmentation has to be False for test
        self.AUG_TO_FLIP = False
        self.AUG_TO_ROT = False
        
        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['G3BP1', "FMRP", "PURA"]
        ######################
        
        ##############################
        
class Test6DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]

        self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'test'
        
        # Augmentation has to be False for test
        self.AUG_TO_FLIP = False
        self.AUG_TO_ROT = False
        
        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        #self.CONDITIONS = ['Untreated']
        self.MARKERS = ['FMRP']
        ######################
        
        ##############################
        
        
class Test9DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]

        self.SPLIT_DATA = False
        # self.DATA_SET_TYPE = 'test'
        
        # Augmentation has to be False for test
        self.AUG_TO_FLIP = False
        self.AUG_TO_ROT = False
        
        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        # self.CELL_LINES = ['FUSRevertant', 'TDP43']
        self.CELL_LINES = ['WT']
        # self.CONDITIONS = ['Untreated']
        self.MARKERS = ['FUS']
        ######################
        
        ##############################
        
    
class Test4DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch4"]]

        self.SPLIT_DATA = False
        # self.DATA_SET_TYPE = 'test'
        
        # Augmentation has to be False for test
        self.AUG_TO_FLIP = False
        self.AUG_TO_ROT = False
        
        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        # self.CELL_LINES = ['WT', 'TDP43', 'FUSHomo']
        self.CELL_LINES = ['WT']
        # self.CONDITIONS = ['Untreated']
        self.MARKERS = ['FMRP']
        # self.MARKERS = ['FUS', 'TDP43']
        ######################
        
        ##############################
        
        
class TrainB8PiecewiseAlsoStressSmallDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8_testCV2"]]
        self.INPUT_FOLDERS = ["/home/labs/hornsteinlab/Collaboration/MOmaps_Ilan2/MOmaps/input/images/processed/spd2/SpinningDisk/batch8_linear_piecewise"]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        # self.CONDITIONS = ['Untreated']
        self.MARKERS = ['G3BP1']#, 'FMRP', 'PURA', 'DCP1A']
        ######################
        
class TrainB8PiecewiseSmallDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8_testCV2"]]
        self.INPUT_FOLDERS = ["/home/labs/hornsteinlab/Collaboration/MOmaps_Ilan2/MOmaps/input/images/processed/spd2/SpinningDisk/batch8_linear_piecewise"]

        self.SPLIT_DATA = True

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1']
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['G3BP1']#, 'FMRP', 'PURA', 'DCP1A']
        ######################
    