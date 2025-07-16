import os
import sys
from typing import Dict, List
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.embeddings.embeddings_config import EmbeddingsConfig

class EffectConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        # The path to the data folders
        self.INPUT_FOLDERS:List[str] = None
        
        self.BASELINE:str = None # example: WT_Untreated
        self.PERTURBATION:str = None # example: WT_stress
        
        # Dictionary mapping each baseline to a list of perturbations.
        self.BASELINE_PERTURB:Dict[int:List[int]] = None # Used for Alyssa's data. for example: {'WT_Untreated':['WT_stress']}

        self.MIN_REQUIRED:int = 500
        self.N_BOOT:int = 1000
