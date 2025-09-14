import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.embeddings.embeddings_config import EmbeddingsConfig
from typing import List


class EmbeddingsFUNOVADatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "FUNOVA", "processed_new_pipeline_w_thrs_0.8")

        self.INPUT_FOLDERS = None
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'funova'    
        self.MARKERS_TO_EXCLUDE = None
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.SHUFFLE:bool = False
        self.SETS:List[str] = ['testset']

        self.CONDITIONS:List[str]         = ["stress", "Untreated"]

        self.CELL_LINES:List[str]         = ["C9orf72-HRE-1008566",
                        "C9orf72-HRE-981344",
                        "Control-1001733",
                        "Control-1017118",
                        "Control-1025045",
                        "Control-1048087",
                        "TDP--43-G348V-1057052",
                        "TDP--43-N390D-1005373"]
        
        self.MARKERS:List[str]  = ["Aberrant-splicing",
                   "Apoptosis",
                   "Autophagy",
                   "Cytoskeleton",
                   "DAPI",
                   "DNA-damage-P53BP1",
                   "DNA-damage-pH2Ax",
                   "impaired-Autophagosome",
                   "mature-Autophagosome",
                   "Necroptosis-HMGB1",
                   "Necroptosis-pMLKL",
                   "Necrosis",
                   "Neuronal-activity",
                   "Nuclear-speckles-SC35",
                   "Nuclear-speckles-SON",
                   "Parthanatos-early",
                   "Parthanatos-late",
                   "Protein-degradation",
                   "Senescence-signaling",
                   "Splicing-factories",
                   "Stress-initiation",
                   "TDP-43",
                   "Ubiquitin-levels",
                   "UPR-ATF4",
                   "UPR-ATF6",
                   "UPR-IRE1a"]



class EmbeddingsFUNOVADatasetConfigExp3(EmbeddingsFUNOVADatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1", "Batch2"]]
        
        
class EmbeddingsFUNOVADatasetConfigExp4(EmbeddingsFUNOVADatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch3", "Batch4"]]