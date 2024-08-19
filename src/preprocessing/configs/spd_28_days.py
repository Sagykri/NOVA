import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingConfig


class SPD_28Days(SPDPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "240323_day29_neurons_sorted", "batch1")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "240323_day29_neurons", "batch1")]
        self.WITH_NUCLEUS_DISTANCE = False
        self.TO_DOWNSAMPLE = False
        self.TILE_WIDTH = 128
        self.TILE_HEIGHT = 128
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_SUBSUBFOLDER, 'logs', "preprocessing_28days", 'batch1')
        
        self.BRENNER_BOUNDS_PATH =  os.path.join(os.getenv("MOMAPS_HOME"), 'src', 'preprocessing', 'sites_validity_bounds_28.csv')
        self.DELETE_MARKER_FOLDER_IF_EXISTS = False