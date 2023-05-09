import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
sys.path.insert(1,'/home/labs/hornsteinlab/Collaboration/MOmaps/') # Nancy

from src.common.configs.model_config import ModelConfig


class ImgselfConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        
        self.HOME_SUBFOLDER = os.path.join(self.MODELS_HOME_FOLDER, "imgself")
        
        self.INPUT_FOLDERS = os.path.join(self.PROCESSED_FOLDER_ROOT, "microglia") 
        
        self.MODEL_PATH = os.path.join(self.HOME_SUBFOLDER, "NOAM PLEASE FIX ME!")
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.HOME_SUBFOLDER, 'models_outputs')

        # Logs
        self.LOGS_FOLDER = os.path.join(self.HOME_SUBFOLDER, 'logs')

        self.MARKERS = ["G3BP1", "KIF5A", "TIA1", "NONO", "SQSTM1",
                            "FMRP", "CD41", "PSD95", "CLTC", "Phalloidin", "NEMO", "DCP1A",
                            "GM130", "TOMM20", "syto12", "Nucleolin", "SNCA", "ANXA11", "LAMP1",
                            "Calreticulin", "PML", "PEX14", "pNFKB", "IL18RAP", "FUS", "DAPI"] 

        self.MARKERS_TO_EXCLUDE=['TDP43', 'PURA']
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['unstressed']
        self.MARKERS_FOR_DOWNSAMPLE = {"DAPI": 0.12}
        self.TRAIN_PCT = 0.6
        self.SHUFFLE = True
        self.SPLIT_DATA = True
        self.DATA_SET_TYPE = 'test'
        
        
        self.SPLIT_BY_SET_FOR = None
        self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.ADD_CONDITION_TO_LABEL = True 
        self.ADD_LINE_TO_LABEL = True
        self.ADD_TYPE_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN]
        
