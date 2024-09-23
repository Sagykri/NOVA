import sys
import os
from typing import Dict

sys.path.insert(1, os.getenv("NOVA_HOME")) 
from src.common.base_config import BaseConfig

class PlotConfig(BaseConfig):
    """Config for plotting
    """
    UMAP_MAPPINGS_ALIAS_KEY:str = 'alias'
    UMAP_MAPPINGS_COLOR_KEY:str = 'color'
    
    def __init__(self):
        
        super().__init__()

        # Set the size of the dots
        self.SIZE:int = None
        
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA:float = None
        
        # Set the color mapping dictionary (name: {alias:alias, color:color})
        self.COLOR_MAPPINGS:Dict[str, Dict[str,str]] = None
        
