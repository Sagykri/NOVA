import os
import sys
from typing import List
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.datasets.dataset_config import DatasetConfig

class EmbeddingsConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # The path to the data folders
        self.INPUT_FOLDERS:List[str] = None
        
        # The name for the experiment
        self.EXPERIMENT_TYPE:str = None
        
        # Which dataset type to load: 'trainset', 'valset', 'testset' 
        self.SETS:List[str] = ['testset']