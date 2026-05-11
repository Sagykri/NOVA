import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.embeddings.embeddings_config import EmbeddingsConfig
from typing import List

from manuscript.FuNOVA_Screen_Conditions_Lists import plate1_conditions, plate2_conditions, plate3_conditions, plate4_conditions


class EmbeddingsFuNOVAScreenDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "FuNOVA_Screen")

        self.INPUT_FOLDERS = None
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'FuNOVA_Screen'    
        self.MARKERS_TO_EXCLUDE = ['HDGFL2', 'FK-2']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.SHUFFLE:bool = False
        self.SETS:List[str] = ['testset']

        self.CELL_LINES:List[str]  = ["C9"]
        self.CONDITIONS:List[str]  = plate1_conditions + plate2_conditions + plate3_conditions + plate4_conditions
        self.MARKERS:List[str]  = ['DAPI', 'TDP-43', 'p62', 'pTDP-43', 'ATF6', 'pAMPK', 'G3BP1', 'Calreticulin', 'Aggreagtes', 'Cas3', 'pS6']



class EmbeddingsFuNOVAScreenb1DatasetConfig(EmbeddingsFuNOVAScreenDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["batch1"]]

        
