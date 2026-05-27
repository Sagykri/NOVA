import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.embeddings.embeddings_config import EmbeddingsConfig
from typing import List



class EmbeddingsRANBP17ExpDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT,'RANBP17_exp')

        self.INPUT_FOLDERS = None
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'RANBP17_exp'    
        self.MARKERS_TO_EXCLUDE = None
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.SHUFFLE:bool = False
        self.SETS:List[str] = ['testset']

        self.CELL_LINES:List[str]  = ["iW11"]
        self.CONDITIONS:List[str]  = ["tardp-kd", "ranbp17-kd", "control-179", "control-180", "both-kd", "untreated"]
        
        self.MARKERS:List[str]  = ['DAPI', 'TDP-43', 'RANBP17']



class EmbeddingsRANBP17ExpBatch1DatasetConfig(EmbeddingsRANBP17ExpDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["batch1"]]

        
