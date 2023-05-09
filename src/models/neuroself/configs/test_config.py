import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.models.neuroself.configs.config import NeuroselfConfig

class NeuroselfTestConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch6", "batch7", "batch8"]]
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6", "batch7"]]

        
        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        self.MARKERS = None

        self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
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
        
        self.SPLIT_BY_SET_FOR = None #("WT", "Untreated")
        self.SPLIT_BY_SET_FOR_BATCH = None #'batch8'
        


        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]
        

        