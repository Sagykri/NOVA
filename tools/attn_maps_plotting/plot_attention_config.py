import sys
import os
from typing import Dict, List, Tuple, Callable
sys.path.insert(1, os.getenv("NOVA_HOME")) 
from src.common.base_config import BaseConfig
import cv2
from PIL import Image


class PlotAttnMapConfig(BaseConfig):
    """Config for Attention Maps plotting
    """
    
    def __init__(self):
        
        super().__init__()

        # num of workers for plotting the processed attn maps parallely (multi-threading)
        self.PLOT_ATTN_NUM_WORKERS:int = None

        # Controls transparency of the attention overlay (higher alpha = more visible red)
        self.ALPHA:float = None

        self.NUM_CONTOURS:int = None # numbed of contours lines for the attention map overlay

        self.ATTN_OVERLAY_THRESHOLD:float = None
        # Controls layout size of the output figure.
        self.FIG_SIZE:tuple = None

        self.SAVE_SEPERATE_LAYERS:bool = None # for "all_layers" attention methods. if True saves each layer in distinct figure (in addition to all-layers-one-fig)

        self.ALL_LAYERS_FIG_SIZE:tuple = None

        self.PLOT_SUPTITLE_FONTSIZE:int = None # main title font size 

        self.PLOT_TITLE_FONTSIZE:int = None # each sub-figure font size 

        self.PLOT_LAYER_FONTSIZE:int = None # each layer, for SAVE_SEPERATE_LAYERS = True

        self.PLOT_SAVEFIG_DPI:int = None # controls the resolution of saved figures. 

        self.PLOT_HEATMAP_COLORMAP:int = None # cv2 int constanat that control colors of the heatmap

        self.SAVE_PLOT:bool = None  # save figure in the output path

        self.SHOW_PLOT:bool = None  # display plot 

        self.FILTER_SAMPLES_BY_FOLDER_PATHS = None # whether to filter the samples by the paths given in the pair-wise distances analysis

        self.DISPLAY_SUPTITLE:bool = None # whether to display the suptitle of the figure

        self.DISPLAY_COMPONENTS_TITLE:bool = None  # whether to display each components sub-title

        self.FIG_COMPONENTS:list = None # which sub-components to display in the figure. options - ["Overlay", "Marker", "Nucleus", "Heatmap"]
        
        # list of colors to be used in LinearSegmentedColormap for filling the contours
        self.FILL_CMAP_LIST:list  = None
        
        # list of colors to be used in LinearSegmentedColormap for the contours lines
        self.LINE_CMAP_LIST:list  = None

