import os
import sys
from typing import List
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.datasets.dataset_config import DatasetConfig

class DistanceConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # The path to the data folders
        self.INPUT_FOLDERS:List[str] = None
        
        # Which cell line + condition to use as baseline
        self.BASELINE_CELL_LINE_CONDITION:str = "WT_Untreated"