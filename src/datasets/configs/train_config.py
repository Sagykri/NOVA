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
        
        
        ##############################
        # TODO: TO DELETE for data manager
        self.DATA_SET_TYPE = 'train'
        self.HOME_SUBFOLDER = os.path.join(os.path.join(self.HOME_FOLDER, "src", "models"), "neuroself_nancy_test")
        self.LOGS_FOLDER = os.path.join(self.HOME_SUBFOLDER, 'logs')
        #############################
        
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
        
        
        ##############################
        # TODO: TO DELETE
        self.DATA_SET_TYPE = 'train'
        self.HOME_SUBFOLDER = os.path.join(os.path.join(self.HOME_FOLDER, "src", "models"), "neuroself_nancy_test")
        self.LOGS_FOLDER = os.path.join(self.HOME_SUBFOLDER, 'logs')
        #############################        

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
        
        
        ##############################
        # TODO: TO DELETE
        self.DATA_SET_TYPE = 'train'
        self.HOME_SUBFOLDER = os.path.join(os.path.join(self.HOME_FOLDER, "src", "models"), "neuroself_nancy_test")
        self.LOGS_FOLDER = os.path.join(self.HOME_SUBFOLDER, 'logs')
        #############################
        
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
        
        # self.AUG_TO_FLIP = False
        # self.AUG_TO_ROT = False
        
        ##############################
        # TODO: TO DELETE
        self.DATA_SET_TYPE = 'train'
        self.HOME_SUBFOLDER = os.path.join(os.path.join(self.HOME_FOLDER, "src", "models"), "neuroself_nancy_test")
        self.LOGS_FOLDER = os.path.join(self.HOME_SUBFOLDER, 'logs')
        #############################