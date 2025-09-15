import os
import sys
from typing import List, Tuple
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.common.base_config import BaseConfig
#from src.embeddings.embeddings_config import EmbeddingsConfig


class PairWiseDistancesConfig(BaseConfig):
    """Config for extracting subset
    """
    
    def __init__(self):
        # initi object with the data from the config given
        super().__init__()  # Initialize base class normally

        #  metric for distance calculation
        self.METRIC:str = "euclidean"

        #number of pairs from each section
        self.NUM_PAIRS:int = 30

        self.SUBSET_METHOD:str = "sectional" # the method to create the subset with :{sectional (min/max/middle), random}

        self.WITHOUT_REPEAT:bool = True # don't allow repeated samples


