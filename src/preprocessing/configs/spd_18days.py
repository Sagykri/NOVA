import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingConfig


class SPD_18Days_Batch1(SPDPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "NOVA_d18_neurons_sorted", "batch1")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "spd18days", "batch1")]
        self.WITH_NUCLEUS_DISTANCE = False
        self.TO_DOWNSAMPLE = False
        self.TILE_WIDTH = 128
        self.TILE_HEIGHT = 128
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_SUBSUBFOLDER, 'logs', "spd18days", 'batch1')
        
        self.BRENNER_BOUNDS_PATH =  os.path.join(os.getenv("MOMAPS_HOME"), 'src', 'preprocessing', 'sites_validity_bounds_spd18days.csv')
        self.DELETE_MARKER_FOLDER_IF_EXISTS = False
        
        self.MARKERS_TO_INCLUDE = ['CLTC', 'DAPI', 'PSD95', 'FMRP', 'Phalloidin', 'SQSTM1', 'AGO2',
                                    'CD41', 'TDP43', 'HNRNPA1', 'PSPC1', 'Tubulin', 'Calreticulin',
                                    'LAMP1', 'ANXA11', 'SNCA', 'G3BP1', 'PURA', 'TOMM20', 'FUS', 'NCL',
                                    'GM130', 'KIF5A', 'DCP1A', 'NEMO', 'PEX14', 'mitotracker',
                                    'NONO', 'VDAC1'] # PML is excluded since all its images were out-of-focus!


class SPD_18Days_Batch2(SPDPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "NOVA_d18_neurons_sorted", "batch2")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "spd18days", "batch2")]
        self.WITH_NUCLEUS_DISTANCE = False
        self.TO_DOWNSAMPLE = False
        self.TILE_WIDTH = 128
        self.TILE_HEIGHT = 128
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_SUBSUBFOLDER, 'logs', "spd18days", 'batch2')
        
        self.BRENNER_BOUNDS_PATH =  os.path.join(os.getenv("MOMAPS_HOME"), 'src', 'preprocessing', 'sites_validity_bounds_spd18days.csv')
        self.DELETE_MARKER_FOLDER_IF_EXISTS = False
        
        self.MARKERS_TO_INCLUDE = ['CLTC', 'DAPI', 'PSD95', 'FMRP', 'Phalloidin', 'SQSTM1', 'AGO2',
                                    'CD41', 'TDP43', 'HNRNPA1', 'PSPC1', 'Tubulin', 'Calreticulin',
                                    'LAMP1', 'ANXA11', 'SNCA', 'G3BP1', 'PURA', 'TOMM20', 'FUS', 'NCL',
                                    'GM130', 'KIF5A', 'DCP1A', 'NEMO', 'PEX14', 'mitotracker',
                                    'NONO', 'VDAC1'] # PML is excluded since all its images were out-of-focus!
        