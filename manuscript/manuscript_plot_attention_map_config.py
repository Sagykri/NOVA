import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))

from NOVA_rotation.Configs.plot_attn_map_config import PlotAttnMapConfig
import cv2
import numpy as np
from PIL import Image

class BaseAttnMapPlotConfig(PlotAttnMapConfig):
    def __init__(self):
        super().__init__()

        self.PLOT_ATTN_NUM_WORKERS:int = 8

        # Controls transparency of the attention overlay (higher alpha = more visible red)
        self.ALPHA:float = 0.2

        self.NUM_CONTOURS:int = 8

        self.ATTN_OVERLAY_THRESHOLD:float = 0.3 # the percentage of the attn values to be visualized on top of the input image 
        # Controls layout size of the output figure.
        self.FIG_SIZE:tuple = (8,8)

        self.SAVE_SEPERATE_LAYERS:bool = False

        self.ALL_LAYERS_FIG_SIZE:tuple = (13, 11)

        self.PLOT_SUPTITLE_FONTSIZE:int = 20 # main title font size

        self.PLOT_TITLE_FONTSIZE:int = 16 # each sub-figure font size

        self.PLOT_LAYER_FONTSIZE:int = 16 # each layer, for SAVE_SEPERATE_LAYERS = True

        self.PLOT_SAVEFIG_DPI:int = 300 # controls the resolution of saved figures.

        self.PLOT_HEATMAP_COLORMAP:int = cv2.COLORMAP_JET # cv2 int constanat that control colors of the heatmap

        self.SAVE_PLOT:bool = True

        self.SHOW_PLOT:bool = False

        self.FILTER_SAMPLES_BY_FOLDER_PATHS = True # boolean or list of folders which had paths files in them , which will be ised to filter the samples

        self.SHOW_CORR_SCORES:bool = False


