import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingConfig


class SPD_Batch1_FUS(SPDPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "FUS_lines_stress_2024_sorted", "batch1")]
        # self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "FUS_lines_stress_2024_sorted", "batch1")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "FUS_lines_stress_2024_sorted", "batch1_Untreated")]
        self.WITH_NUCLEUS_DISTANCE = False
        self.TO_DOWNSAMPLE = False
        self.TILE_WIDTH = 128
        self.TILE_HEIGHT = 128
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_SUBSUBFOLDER, 'logs', "preprocessing_FUS", 'batch1_Untreated')
        
        self.BRENNER_BOUNDS_PATH =  os.path.join(os.getenv("MOMAPS_HOME"), 'src', 'preprocessing', 'sites_validity_bounds_FUS.csv')
        # self.DELETE_MARKER_FOLDER_IF_EXISTS = True
        # self.MARKERS_TO_INCLUDE = ['NEMO', 'DCP1A', 'SNCA', 'PSD95', 'FMRP', 'SQSTM1', 'TDP43', 'TOMM20']
        
        self.CELL_LINES_TO_INCLUDE = ['KOLF']
        self.CONDITIONS_TO_INCLUDE = ['Untreated']