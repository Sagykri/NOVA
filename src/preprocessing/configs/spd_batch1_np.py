import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.preprocessing.configs.preprocessor_spd_np_config import SPDPreprocessingConfigdNP


class SPD_Batch1NP(SPDPreprocessingConfigdNP):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_SUBFOLDER_ROOT, "NiemannPick_sort", "batch1")]
        self.OUTPUT_FOLDERS = [os.path.join(self.PROCESSED_SUBFOLDER_ROOT, "NiemannPick","batch1")]      