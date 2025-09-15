import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.figures.plot_attn_score_config import PlotAttnScoreConfig
import cv2
import numpy as np
from PIL import Image

class BasePlotAttnScorerConfig(PlotAttnScoreConfig):
    def __init__(self):
        super().__init__()


        self.PLOT_SUPTITLE_FONTSIZE:int = 14 # main title font size

        self.PLOT_TITLE_FONTSIZE:int = 12 # each sub-figure font size

        self.PLOT_SAVEFIG_DPI:int = 300 # controls the resolution of saved figures.

        self.SAVE_PLOT:bool = True # whether to save plot in output_folder_path in the plot_correlation function in attn_scores_plotting

        self.SHOW_PLOT:bool = False # whether to display generated plot (plot_correlation function in attn_scores_plotting)


        self.PLOT_CORR_SUMMARY:bool = True # whether to call plotting script at all (in generate_attn_scores_and_plot)
        
        self.PLOT_CORR_SEPERATE_MARKERS:bool = True # in attn_scores_plotting - plot seperate graph for each marker

        self.PLOT_CORR_ALL_MARKERS:bool = True # in attn_scores_plotting - plot combined graph for all marker


        self.PLOT_CORR_PER_LAYER:bool = False # in attn_scores_plotting - plot score for each one of the layer seperatly ( only works if attn method is "all layers")

        
