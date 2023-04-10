import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.base_config import BaseConfig


class ModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        self.MODELS_HOME_FOLDER = os.path.join(self.HOME_FOLDER, "src", "models")
        
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