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
        self.OUTPUTS_FOLDER =  os.path.join(os.getenv("NOVA_HOME"), "outputs", "preprocessing", "AAT-NOVA")
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'logs')
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_opera", "OperaPreprocessor")        

        self.CELL_LINES = ["CTL", "C9"]
        self.CONDITIONS = ["PPP2R1A","HMGCS1","PIK3C3","NDUFAB1","MAPKAP1","NDUFS2","RALA","TLK1","NRIP1","TARDBP","RANBP17","CYLD","NT-1873","NT-6301-3085","Intergenic","Untreated"]
        self.REPS = ["rep1", "rep2"]
        self.PANELS = ["panelA", "panelB", "panelC", "panelD", "panelE", "panelF"]
        self.MARKERS = ["DAPI", "Cas3", "FK-2", "SMI32", "pDRP1", "TOMM20", "pCaMKIIa", "pTDP-43", "TDP-43", "ATF6", "pAMPK", "HDGFL2", "pS6", "PAR", "UNC13A", "Calreticulin", "LC3-II", "p62", "CathepsinD"]

        self.MARKERS_FOCUS_BOUNDRIES_PATH = os.path.join(
            os.getenv("NOVA_HOME"), 
            'manuscript', 
            'markers_focus_boundries', 
            'markers_focus_boundries_AAT_NOVA.csv'
        )

class PreprocessingBaseConfigAATNOVA_pilot2(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        self.RAW_FOLDER_ROOT = f"{os.getenv('AAT_NOVA_DATA_DIR')}2"
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'sorted')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "AAT_NOVA_pilot2", "processed")
        self.OUTPUTS_FOLDER =  os.path.join(os.getenv("NOVA_HOME"), "outputs", "preprocessing", "AAT_NOVA_pilot2")
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'logs')
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_opera", "OperaPreprocessor")        

        self.CELL_LINES = ["CTL", "C9"]
        self.CONDITIONS = ["PPP2R1A","HMGCS1","PIK3C3","NDUFAB1","MAPKAP1","NDUFS2","RALA","TLK1","NRIP1","TARDBP","RANBP17","CYLD","NT-1873","NT-6301-3085","Intergenic","Untreated"]
        self.REPS = ["rep1", "rep2"]
        self.PANELS = ["panelA", "panelB", "panelC", "panelD", "panelE", "panelF"]
        self.MARKERS = ["DAPI", "ATF4", "FK-2", "SMI32", "pDRP1", "TOMM20", "pCaMKIIa", "pTDP-43", "TDP-43", "ATF6", "pAMPK", "G3BP1", "pS6", "PAR", "UNC13A", "Calreticulin", "POM121", "p62", "CathepsinD", "Brightfield"]

        self.MARKERS_FOCUS_BOUNDRIES_PATH = os.path.join(
            os.getenv("NOVA_HOME"), 
            'manuscript', 
            'markers_focus_boundries', 
            'markers_focus_boundries_AAT_NOVA_pilot2.csv'
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

#-----------------------Pilot 2 Batches -----------------------#
class PreprocessingBaseConfigAATNOVAPilot2Batch1(PreprocessingBaseConfigAATNOVA_pilot2):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch1")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch1")]
        self.OUTPUTS_FOLDER  = os.path.join(self.OUTPUTS_FOLDER, "batch1")

class PreprocessingBaseConfigAATNOVAPilot2Batch2(PreprocessingBaseConfigAATNOVA_pilot2):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch2")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch2")]
        self.OUTPUTS_FOLDER  = os.path.join(self.OUTPUTS_FOLDER, "batch2")

class PreprocessingBaseConfigAATNOVAPilot2Batch3(PreprocessingBaseConfigAATNOVA_pilot2):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch3")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch3")]
        self.OUTPUTS_FOLDER  = os.path.join(self.OUTPUTS_FOLDER, "batch3")

class PreprocessingBaseConfigAATNOVAPilot2Batch3PanelF2Rep2(PreprocessingBaseConfigAATNOVAPilot2Batch3):
    def __init__(self):
        super().__init__()
        
        self.CELL_LINES = ["CTL", "C9"]
        self.CONDITIONS = ["NRIP1","TARDBP","RANBP17","CYLD","NT-1873","NT-6301-3085","Intergenic","Untreated"]
        self.REPS = ["rep2"]
        self.PANELS = ["panelF"]
        self.MARKERS =  ["DAPI", "p62", "POM121",  "CathepsinD", "Brightfield"]