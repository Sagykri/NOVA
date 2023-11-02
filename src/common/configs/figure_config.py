import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.base_config import BaseConfig

class FigureConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        self.HOME_FIGURES_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "figures") # moved by nancy 29/10/23
        self.OUTPUT_DIR = None
        self.FIGURES = []
        self.HANDLER_CLASS_PATH = None