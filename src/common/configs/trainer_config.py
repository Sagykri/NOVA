import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.base_config import BaseConfig


class TrainerConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        self.OUTPUTS_FOLDER = None

        # Training parameters
        self.LR = None
        self.MIN_LR = None
        self.MAX_EPOCHS = None
        self.WARMUP_EPOCHS = None
        self.WEIGHT_DECAY = None
        self.WEIGHT_DECAY_END = None
        self.BATCH_SIZE = None
        self.NUM_WORKERS = 6
        self.EARLY_STOPPING_PATIENCE = 10
        self.DESCRIPTION = None