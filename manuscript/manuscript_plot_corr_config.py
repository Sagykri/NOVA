import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.figures.plot_correlation_config import PlotCorrConfig
import cv2
import numpy as np
from PIL import Image

class BasePlotCorrConfig(PlotCorrConfig):
    def __init__(self):
        super().__init__()


        self.PLOT_SUPTITLE_FONTSIZE:int = 14 # main title font size

        self.PLOT_TITLE_FONTSIZE:int = 12 # each sub-figure font size

        self.PLOT_SAVEFIG_DPI:int = 300 # controls the resolution of saved figures.

        self.SAVE_PLOT:bool = True

        self.SHOW_PLOT:bool = False

        self.PLOT_CORR_SUMMARY:bool = True
        
        self.PLOT_CORR_SEPERATE_MARKERS:bool = True

        self.PLOT_CORR_ALL_MARKERS:bool = True

        self.PLOT_CORR_PER_LAYER:bool = False
