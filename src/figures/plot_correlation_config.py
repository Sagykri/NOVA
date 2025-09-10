import sys
import os
from typing import Dict, List, Tuple, Callable

sys.path.insert(1, os.getenv("NOVA_HOME")) 
from src.common.base_config import BaseConfig
import cv2
from PIL import Image


class PlotCorrConfig(BaseConfig):
    """Config for Attention Maps plotting
    """
    
    def __init__(self):
        
        super().__init__()

        self.PLOT_SUPTITLE_FONTSIZE:int = None # main title font size v

        self.PLOT_TITLE_FONTSIZE:int = None # each sub-figure font size v

        self.PLOT_SAVEFIG_DPI:int = None # controls the resolution of saved figures. V

        self.SAVE_PLOT:bool = None #V

        self.SHOW_PLOT:bool = None #V

        self.PLOT_CORR_SUMMARY:bool = None#V
        
        self.PLOT_CORR_SEPERATE_MARKERS:bool = None#V

        self.PLOT_CORR_ALL_MARKERS:bool = None#V

        self.PLOT_CORR_PER_LAYER:bool = None#V

