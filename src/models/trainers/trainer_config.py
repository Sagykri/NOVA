import os

import sys
from typing import List
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.base_config import BaseConfig


class TrainerConfig(BaseConfig):
    """Base trainer configuration
    """
    def __init__(self):
        super().__init__()

        # Training parameters
        
        # The starting learning rate
        self.LR:float = None
        # The final learning rate at the end of the schedule
        self.MIN_LR:float = None
        # Number of epochs
        self.MAX_EPOCHS:int = None
        # Number of epochs to warmup the learning rate
        self.WARMUP_EPOCHS:int = None
        # The starting weight decay value
        self.WEIGHT_DECAY:float = None
        # The final weight decay value at the end of the schedule
        self.WEIGHT_DECAY_END:float = None
        # The batchsize (how many files to load per batch)
        self.BATCH_SIZE:int = None
        # Number of works to run during the data loading
        self.NUM_WORKERS:int = 6
        # Number of straight epochs without improvement to wait before activating eary stopping 
        self.EARLY_STOPPING_PATIENCE:int = 10
        # The path to the trainer class (the path to the py file, then / and then the name of the class)
        # ex: os.path.join("src", "common", "lib", "models", "trainers", "trainer_classification", "TrainerClassification")
        self.TRAINER_CLASS_PATH:str = None
        # A textual description for the model (optional, default to the trainer class name)
        self.DESCRIPTION:str = None
        # Whether to drop the last batch if its partial of the expected batch size
        self.DROP_LAST_BATCH:bool = False 
        # The path to the aumentation to apply on each sample in the data (the path to the py file, then / and then the name of the class)
        # ex: os.path.join("src", "models", "utils", "augmentations", "RotationsAndFlipsAugmentation")
        self.DATA_AUGMENTATION_CLASS_PATH:str = os.path.join("src", "models", "utils", "augmentations", "RotationsAndFlipsAugmentation")
                           

        


