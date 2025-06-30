import os
import sys
from typing import List
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.embeddings.embeddings_config import EmbeddingsConfig

class EffectConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        # The path to the data folders
        self.INPUT_FOLDERS:List[str] = None
        
        # Dictionary mapping each baseline to a list of perturbations.
        self.BASELINE_PERTURB = {'WT_Untreated':['WT_stress']}
