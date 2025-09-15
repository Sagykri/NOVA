import sys
import os
from typing import Dict, List, Tuple

sys.path.insert(1, os.getenv("NOVA_HOME")) 
from src.analysis.attention_scores_config import AttnScoresBaseConfig

class AttnScoresConfig(AttnScoresBaseConfig):
    """Config for plotting
    """
    
    def __init__(self):
        
        super().__init__()

        self.CORR_METHOD = "soft_overlap" #["pearsonr", "mutual_info", "ssim", "attn_overlap", "soft_overlap", "prob_overlap", "binary_score"]

        self.CORR_METHOD_ARGS = None

        self.FEATURES_NAMES = ["precision", "recall", "f1"] # scores names when multiple results are returned

        self.THRESHOLD_METHOD  = "percentile" # ["percentile", any function from skimage.filters]

        self.THRESHOLD_ARGS = {"percentile": 0.75} 
