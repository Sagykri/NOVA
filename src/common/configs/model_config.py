import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.base_config import BaseConfig


class ModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        # Architecture parameters
        self.VIT_VERSION = 'tiny'
        self.IMAGE_SIZE = 100
        self.PATCH_SIZE = 14
        self.NUM_CHANNELS = 2
        self.NUM_CLASSES = None