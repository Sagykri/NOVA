from configs.base_config import BaseConfig


class ModelConfig(BaseConfig):
    def __init__(self):
        super(BaseConfig, self).__init__()
        
        # Preprocessing
        self.TILE_W = 300
        self.TILE_H = 300
        self.NUCLEUS_CHANNEL = 3
        self.FLOW_THRESHOLD = 0.4
        self.CHANNEL_AXIS = -1
        self.NUCLEUS_DIAMETER = 60
        self.MIN_EDGE_DISTANCE = 2

        # Metrics
        # self.METRICS_FOLDER = os.path.join(self.HOME_FOLDER, "metrics")
        # self.METRICS_RANDOM_PATH = os.path.join(self.METRICS_FOLDER, "random.npy")
        # self.METRICS_MATCH_PATH = os.path.join(self.METRICS_FOLDER, "match.npy")

        
        
    
        self.MARKERS_TO_EXCLUDE = None
        self.CELL_LINES = None
        self.CONDITIONS = None
        self.TRAIN_CELL_LINES = None
        self.TEST_CELL_LINES = None
        self.TRAIN_CONDITIONS = None
        self.TEST_CONDITIONS = None
        self.MARKERS_FOR_DOWNSAMPLE = None
        self.TRAIN_PCT = 0.7
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
        
        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #         "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #         "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #         "mitotracker", "Nucleolin", "SNCA", \
        #         "GM130", "PEX14", "PML", "DAPI"]
        # self.MICROGLIA_MARKERS = ["G3BP1", "KIF5A", "TIA1", "NONO", "SQSTM1",
        #                     "FMRP", "CD41", "PSD95", "CLTC", "Phalloidin", "NEMO", "DCP1A",
        #                     "GM130", "TOMM20", "syto12", "Nucleolin", "SNCA", "ANXA11", "LAMP1",
        #                     "Calreticulin", "PML", "PEX14", "pNFKB", "IL18RAP", "FUS", "DAPI"]  # ,"TDP43", "PURA"]

        # self.COMBINED_MARKERS = ["G3BP1", "KIF5A", "TIA1", "NONO", "SQSTM1",
        #                     "FMRP", "CD41", "PSD95", "CLTC", "Phalloidin", "NEMO", "DCP1A",
        #                     "GM130", "TOMM20", "Nucleolin", "SNCA", "ANXA11", "LAMP1",
        #                     "Calreticulin", "PML", "PEX14", "FUS", "DAPI"]
        # UMAP

        


