import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.preprocessing.preprocessing_config import PreprocessingConfig

class PreprocessingBaseConfigAATNOVA(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        self.RAW_FOLDER_ROOT = os.getenv("AAT_NOVA_DATA_DIR")
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'zstack_collapse_2nd_imaging_sorted')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "AAT-NOVA_pilot", "processed")
        self.OUTPUTS_FOLDER =  os.path.join(os.getenv("NOVA_LOCAL"), "outputs", "preprocessing", "AAT-NOVA")
        self.LOGS_FOLDER = self.OUTPUTS_FOLDER 
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_opera", "OperaPreprocessor")        

        self.CELL_LINES = ["CTL", "C9"]
        self.CONDITIONS = ["PPP2R1A","HMGCS1","PIK3C3","NDUFAB1","MAPKAP1","NDUFS2","RALA","TLK1","NRIP1","TARDBP","RANBP17","CYLD","NT-1873","NT-6301-3085","Intergenic","Untreated"]
        self.REPS = ["rep1", "rep2"]
        self.PANELS = ["panelA", "panelB", "panelC", "panelD", "panelE", "panelF"]
        self.MARKERS = ["DAPI", "Cas3", "FK-2", "SMI32", "pDRP1", "TOMM20", "pCaMKIIa", "pTDP-43", "TDP-43", "ATF6", "pAMPK", "HDGFL2", "pS6", "PAR", "UNC13A", "Calreticulin", "LC3-II", "p62", "CathepsinD"]


        # # NEED TO ADJUST
        # self.CELLPOSE = {
        #     'NUCLEUS_DIAMETER': 70,
        #     'CELLPROB_THRESHOLD': 0,
        #     'FLOW_THRESHOLD': 0.22
        # } ## Adjusted cellpose params to get dead dead cells

        # NEED TO ADJUST
        self.MARKERS_FOCUS_BOUNDRIES_PATH = os.path.join(
            os.getenv("NOVA_LOCAL"), 
            'manuscript', 
            'markers_focus_boundries', 
            'markers_focus_boundries_AAT_NOVA.csv'
        )


class PreprocessingBaseConfigAATNOVABatch1(PreprocessingBaseConfigAATNOVA):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch1")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch1")]
        self.OUTPUTS_FOLDER  = os.path.join(self.OUTPUTS_FOLDER, "batch1")

class PreprocessingBaseConfigAATNOVABatch2(PreprocessingBaseConfigAATNOVA):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch2")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch2")]
        self.OUTPUTS_FOLDER  = os.path.join(self.OUTPUTS_FOLDER, "batch2")