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

        self.MIN_REQUIRED:int = 500 # min required sites!
        self.N_BOOT:int = 1000
        self.SUBSAMPLE_FRACTION = 0.8 # fraction of samples to use in each bootstrap iteration (the formula is: max(MIN_REQUIRED, int(n_samples ** SUBSAMPLE_FRACTION)) (i.e. to the power))
        self.BOOTSTRAP_TRIMMING_ALPHA:float = 0 # fraction of extreme values to trim from the bootstrap distribution for estimating the variance (e.g. 0.01 means trimming 1% from each tail) (Default: 0, no trimming)