import os
import sys
from typing import List

sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.datasets.dataset_config import DatasetConfig

class FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        # The path to the data folders
        self.INPUT_FOLDERS:List[str] = None
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI:bool = False

        # Function to edit labels; only used when SHOW_ARI==True and if the ARI labels needs to be different than the shown labels
        self.ARI_LABELS_FUNC:str = None

        # Which cell line + condition is used as baseline; used for distances figures
        self.BASELINE_CELL_LINE_CONDITION:str = None

        # Which other cell lines + conditions are being compared to the baseline; used for distances figures
        self.CELL_LINES_CONDITIONS:List[str] = None
