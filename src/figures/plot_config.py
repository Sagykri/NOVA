import sys
import os
from typing import Dict, List, Tuple

sys.path.insert(1, os.getenv("NOVA_HOME")) 
from src.common.base_config import BaseConfig

class PlotConfig(BaseConfig):
    """Config for plotting
    """
    
    def __init__(self):
        
        super().__init__()

        # Set the size of the dots
        self.SIZE:int = None
        
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA:float = None
        
        # Whether to mix-up different groups' plotting order in UMAP; used when groups are plotted on top of each other.
        self.MIX_GROUPS:bool = False

        # Set the color mapping dictionary (name: {alias:alias, color:color})
        self.COLOR_MAPPINGS:Dict[str, Dict[str,str]] = None
        
        # Set the alias mapping key
        self.MAPPINGS_ALIAS_KEY:str = 'alias'
        # Set the color mapping key
        self.MAPPINGS_COLOR_KEY:str = 'color'

        # Whether to show the baseline's ARI boxplot; used for marker ranking plots
        self.SHOW_BASELINE:bool = True

        # Define marker order for bubble plot
        self.ORDERED_MARKERS:List[str] = None
        
        # Define cell line order for bubble plot
        self.ORDERED_CELL_LINES:List[str] = None

        # Define a range for the y-axis break (used for marker ranking graph, if y-axis cut is needed)
        self.YAXIS_CUT_RANGES: dict[str, Tuple[float, float]] = {
            'UPPER_GRAPH': None,
            'LOWER_GRAPH': None
        }

