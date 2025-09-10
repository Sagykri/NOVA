import sys
import os
from typing import Dict, List, Tuple, Callable

sys.path.insert(1, os.getenv("NOVA_HOME")) 
from src.common.base_config import BaseConfig



class AttnConfig(BaseConfig):
    """Config for Attention Maps plotting
    """
    
    def __init__(self):
        
        super().__init__()

        # attention method to be used: all_layers/rollout
        self.ATTN_METHOD:str = None 

        # integer represents PIL Image resampling method:
        # best methods: BICUBIC/LANCZOS, other (faster): NEAREST, BOX, BILINEAR, HAMMING
        self.RESAMPLE_METHOD:int = None

        # resuction of number of heads - min/max/mean (which is supported by numpy)
        self.REDUCE_HEAD_FUNC:str = None

        # min value to keep in the attentio maps (below is zeroed out)
        self.MIN_ATTN_THRESHOLD:float = None

        # correlation method to be used between attention maps and original image
        # options: ["pearsonr", "mutual_info", "ssim", "attn_overlap"]
        self.CORR_METHOD:str = None

        # num of workers for processing the raw attn maps parallely (multi-proccesing)
        self.ATTN_NUM_WORKERS:int = None

        #flag to save the raw attention 
        self.SAVE_RAW_ATTN:bool = False


    
        
