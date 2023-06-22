import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.dataset_config import DatasetConfig


class TrainBatch8DMDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        __folders = [os.path.join('spd2','SpinningDisk','batch8')]
        
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in __folders]

        self.SPLIT_DATA = True
        self.DATA_SET_TYPE = 'train'

        ##############################
        # TODO: USED FOR DATA MANAGER ONLY
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_dm')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs_batch8_dm')
        #############################

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT']
        self.CELL_LINES = ['TDP43', 'TBK1']
        self.CONDITIONS = ['Untreated']
        ######################     

class ValBatch8DMDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        __folders = [os.path.join('spd2','SpinningDisk','batch8')]
        
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in __folders]

        self.SPLIT_DATA = True
        self.DATA_SET_TYPE = 'val'

        ##############################
        # TODO: USED FOR DATA MANAGER ONLY
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_dm')
        #############################

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT']
        self.CELL_LINES = ['TDP43', 'TBK1']
        self.CONDITIONS = ['Untreated']
        ######################
        
class TestBatch8DMDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        __folders = [os.path.join('spd2','SpinningDisk','batch8')]
        
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in __folders]

        self.SPLIT_DATA = True
        self.DATA_SET_TYPE = 'test'
        
        ##############################
        # TODO: USED FOR DATA MANAGER ONLY
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_dm')
        #############################

        ######################
        # TODO: Just for testing on less classes, remove afterwards
        # self.CELL_LINES = ['WT']
        self.CELL_LINES = ['TDP43', 'TBK1']
        self.CONDITIONS = ['Untreated']
        ######################