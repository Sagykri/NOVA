import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingBaseConfig


class SPD_Batch6_Test(SPDPreprocessingBaseConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = ["/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/batch6"]#[os.path.join(self.RAW_SUBFOLDER_ROOT, "batch6")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch6_test190924")]
    
        
        self.MARKERS = ['G3BP1', 'DAPI'] # 'DCP1A', 'PURA', 'PML'
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.REPS = ["rep1"]
        
        