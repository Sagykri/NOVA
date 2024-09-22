import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.preprocessing.configs.preprocessor_opera18daysReimaged_config import Opera18DaysReimagedPreprocessingConfig


class Opera18DaysReimaged_Batch1(Opera18DaysReimagedPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "batch1")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "batch1")]
        self.WITH_NUCLEUS_DISTANCE = False
        self.TO_DOWNSAMPLE = False
        self.TILE_WIDTH = 128
        self.TILE_HEIGHT = 128
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_SUBSUBFOLDER, 'logs', 'batch1')
        
        self.DELETE_MARKER_FOLDER_IF_EXISTS = False
        
class Opera18DaysReimaged_Batch2(Opera18DaysReimagedPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "batch2")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "batch2")]
        self.WITH_NUCLEUS_DISTANCE = False
        self.TO_DOWNSAMPLE = False
        self.TILE_WIDTH = 128
        self.TILE_HEIGHT = 128
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_SUBSUBFOLDER, 'logs', 'batch2')
        
        self.DELETE_MARKER_FOLDER_IF_EXISTS = False