import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))

from tools.attn_maps_plotting.plot_attention_config import PlotAttnMapConfig
import cv2
import numpy as np
from PIL import Image


class BaseAttnMapPlotConfig(PlotAttnMapConfig):
    def __init__(self):
        super().__init__()

        # num of workers for plotting the processed attn maps parallely (multi-threading)
        self.PLOT_ATTN_NUM_WORKERS:int = 8

        # Controls transparency of the attention overlay (higher alpha = more visible red)
        self.ALPHA:float = 0.2

        self.NUM_CONTOURS:int = 8 # numbed of contours lines for the attention map overlay

        self.ATTN_OVERLAY_THRESHOLD:float = 0.3 # the percentage of the attn values to be visualized on top of the input image 
        # Controls layout size of the output figure.
        self.FIG_SIZE:tuple = (6,6)

        self.SAVE_SEPERATE_LAYERS:bool = False

        self.ALL_LAYERS_FIG_SIZE:tuple = (13, 11)

        self.PLOT_SUPTITLE_FONTSIZE:int = 20 # main title font size

        self.PLOT_TITLE_FONTSIZE:int = 16 # each sub-figure font size

        self.PLOT_LAYER_FONTSIZE:int = 16 # each layer, for SAVE_SEPERATE_LAYERS = True

        self.PLOT_SAVEFIG_DPI:int = 300 # controls the resolution of saved figures.

        self.PLOT_HEATMAP_COLORMAP:int = cv2.COLORMAP_JET # cv2 int constanat that control colors of the heatmap

        self.SAVE_PLOT:bool = True # save figure in the output path

        self.SHOW_PLOT:bool = False # display plot 

        self.DISPLAY_SUPTITLE:bool = False # whether to display the suptitle of the figure

        self.DISPLAY_COMPONENTS_TITLE:bool = True  # whether to display each components sub-title

        self.FILTER_SAMPLES_BY_FOLDER_PATHS = True # whether to filter the samples by the paths given in the pair-wise distances analysis

        self.FIG_COMPONENTS:list = ["Overlay", "Heatmap"] # which sub-components to display in the figure. options - ["Overlay", "Marker", "Nucleus", "Heatmap"]
        
        # list of colors to be used in LinearSegmentedColormap for filling the contours
        self.FILL_CMAP_LIST:list  = [ (0, 0, 0, 0),
                                (1, 1, 0, 0.4),
                                (1, 0.6, 0, 0.6),
                                (1, 0, 0, 0.8)] 
        
        # list of colors to be used in LinearSegmentedColormap for the contours lines
        self.LINE_CMAP_LIST:list  = [ (1, 1, 1, 0.2),
                                (1, 1, 0, 0.4),
                                (1, 0.6, 0, 0.6),
                                (1, 0, 0, 0.8)] 

        self.LINE_WIDTH:float = 3.0 # width of the contour lines
    


