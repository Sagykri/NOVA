import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))



from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingBaseConfig

class SPDPreprocessingConfigdNLS(SPDPreprocessingBaseConfig):
    def __init__(self):
        super().__init__()
    
        self.MARKERS_FOCUS_BOUNDRIES_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/src/preprocessing/markers_focus_boundries_dNLS.csv"

