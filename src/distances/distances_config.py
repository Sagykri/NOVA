import os
import sys
from typing import List
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.embeddings.embeddings_config import EmbeddingsConfig

class DistanceConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        # The path to the data folders
        self.INPUT_FOLDERS:List[str] = None
        
        # Which cell line + condition to use as baseline
        self.BASELINE_CELL_LINE_CONDITION:str = "WT_Untreated"

        # Wether to use random split on the baseline samples
        self.RANDOM_SPLIT:bool = True