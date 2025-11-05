import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.embeddings.embeddings_config import EmbeddingsConfig
from typing import List


class EmbeddingsAATNOVADatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "AAT_NOVA", "processed")

        self.INPUT_FOLDERS = None
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'AAT_NOVA'    
        self.MARKERS_TO_EXCLUDE = None
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.SHUFFLE:bool = False
        self.SETS:List[str] = ['testset']

        self.CELL_LINES:List[str]  = ["CTL", "C9"]
        self.CONDITIONS:List[str]  = ["PPP2R1A","HMGCS1","PIK3C3","NDUFAB1","MAPKAP1","NDUFS2","RALA","TLK1","NRIP1","TARDBP","RANBP17","CYLD","NT-1873","NT-6301-3085","Intergenic","Untreated"]
        self.MARKERS:List[str]  = ["DAPI", "Cas3", "FK-2", "SMI32", "pDRP1", "TOMM20", "pCaMKIIa", "pTDP-43", "TDP-43", "ATF6", "pAMPK", "HDGFL2", "pS6", "PAR", "UNC13A", "Calreticulin", "LC3-II", "p62", "CathepsinD"]

class EmbeddingsAATNOVAAllBatchesDatasetConfig(EmbeddingsAATNOVADatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["batch1", "batch2"]]  

class EmbeddingsAATNOVAb1DatasetConfig(EmbeddingsAATNOVADatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["batch1"]]
        
        
class EmbeddingsAATNOVAb2DatasetConfig(EmbeddingsAATNOVADatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["batch2"]]