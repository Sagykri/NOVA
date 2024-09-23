import os

import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.common.base_config import BaseConfig


class ModelConfig(BaseConfig):
    """Base configuration for the model
    """
    def __init__(self):
        super().__init__()
        
        # Architecture parameters
        
        # The version of the vit (tiny|small|base)
        self.VIT_VERSION:str = 'tiny'
        # The image size (weight==height) the model would expect
        self.IMAGE_SIZE:int = 100
        # The patch size for the model
        self.PATCH_SIZE:int = 14
        # Num of channels the model would expect in the input sampels
        self.NUM_CHANNELS:int = 2
        # The size of the model's output 
        self.OUTPUT_DIM:int = None
        
