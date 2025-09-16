import sys
import os
from typing import Dict, List, Tuple, Callable

sys.path.insert(1, os.getenv("NOVA_HOME")) 
from src.common.base_config import BaseConfig
import cv2
from PIL import Image


class PlotAttnScoreConfig(BaseConfig):
    """Config for Attention Maps plotting
    """
    
    def __init__(self):
        
        super().__init__()

        self.PLOT_SUPTITLE_FONTSIZE:int = None # main title font size 

        self.PLOT_TITLE_FONTSIZE:int = None # each sub-figure font size 

        self.PLOT_SAVEFIG_DPI:int = None # controls the resolution of saved figures. 

        self.SAVE_PLOT:bool = None # whether to save plot in output_folder_path in the plot_correlation function in attn_scores_plotting

        self.SHOW_PLOT:bool = None # whether to display generated plot (plot_correlation function in attn_scores_plotting)

        self.PLOT_CORR_SUMMARY:bool = None # whether to call plotting script at all (in generate_attn_score_and_plot)
        
        self.PLOT_CORR_SEPERATE_MARKERS:bool = None # in attn_scores_plotting - plot seperate graph for each marker

        self.PLOT_CORR_ALL_MARKERS:bool = None  # in attn_scores_plotting - plot combined graph for all marker

        self.PLOT_CORR_PER_LAYER:bool = None # in attn_scores_plotting - plot score for each one of the layer seperatly ( only works if attn method is "all layers")

