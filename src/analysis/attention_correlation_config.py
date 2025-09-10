import sys
import os
from typing import Dict, List, Tuple

sys.path.insert(1, os.getenv("NOVA_HOME")) 
from src.common.base_config import BaseConfig

class AttnCorrScoresConfig(BaseConfig):
    """Config for plotting
    """
    
    def __init__(self):
        
        super().__init__()

        # correaltion score method (from analyzer_corr_utils): ["pearsonr", "mutual_info", "ssim", "attn_overlap", "soft_overlap", "prob_overlap", "binary_score"]
        self.CORR_METHOD:str = None

        # scores names when multiple results are returned
        self.FEATURES_NAMES:List = None

        # threshold method for binary thresholding (! used onlt if needed for CORR_METHOD)
        # ["percentile", any function from skimage.filters]
        self.THRESHOLD_METHOD:str = None

        # arguments for the binary threshold function. 
        # examples - {"percentile": 0.75} 
        #            {"nbins": 125}
        self.THRESHOLD_ARGS:Dict = None
