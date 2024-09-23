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