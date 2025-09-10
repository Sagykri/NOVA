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

        # Whether to use random split on the baseline samples; If False, the baseline samples are not randomly split,
        # and ARI are calculated between different reps of the baseline (used when the number of reps is relatively high).
        # If True, the baseline reps are randomly split into halves, and then ARI is calculated between these halves.
        self.RANDOM_SPLIT_BASELINE:bool = True