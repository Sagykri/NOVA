import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import string
from src.common.configs.figure_config import FigureConfig


class FiguresHandler():
    """Base class for figures handlers"""
    def __init__(self, config:FigureConfig):
        self.__config = config
        
    @property
    def config(self):
        return self.__config
    
    @config.setter
    def config(self, config:FigureConfig):
        self.__config = config
        
    def __call_function(self, func_name:string, arg):
        """Call function by name with argument

        Args:
            func_name (string): Function's name
            arg (_type_): Argument to pass to function
        """
        getattr(self, func_name)(arg)
        
    def get_figures(self):
        """Get figures (specified by the config)"""
        
        figs = self.config.figures
        
        for f in figs:
            panels = figs[f]
            self.__call_function(f, panels)