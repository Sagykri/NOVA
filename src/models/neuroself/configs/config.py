import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.model_config import ModelConfig

class NeuroselfConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        
        self.HOME_SUBFOLDER = os.path.join(self.MODELS_HOME_FOLDER, "neuroself")
        
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["220814_neurons",
                        "220818_neurons",
                        "220831_neurons",
                        "220908", "220914"]]

        # Logs
        self.LOGS_FOLDER = os.path.join(self.HOME_SUBFOLDER, 'logs')
        
        # Models
        self.MODEL_PATH = os.path.join(self.HOME_SUBFOLDER, "MODEL18_model_weights.0040.h5")
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.HOME_SUBFOLDER, 'models_outputs')

        self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
            "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
            "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
            "mitotracker", "Nucleolin", "SNCA", \
            "GM130", "PEX14", "PML", "DAPI"]

        self.MARKERS_TO_EXCLUDE = ['DAPI', 'lysotracker', 'Syto12']
        self.CELL_LINES = None
        self.CONDITIONS = None
        self.SPLIT_DATA = True
        self.DATA_SET_TYPE = 'test'
        self.MARKERS_FOR_DOWNSAMPLE = None
        self.TRAIN_PCT = 0.7
        self.SHUFFLE = True
        self.ADD_CONDITION_TO_LABEL = True 
        self.ADD_LINE_TO_LABEL = True
        self.ADD_TYPE_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.SPLIT_BY_SET_FOR = None
        self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 32
        self.MAX_EPOCH = 100


        self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]

        