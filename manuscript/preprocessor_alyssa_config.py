import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.preprocessing.preprocessing_config import PreprocessingConfig

class AlyssaPreprocessingConfig(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = os.path.join(self.RAW_FOLDER_ROOT, 'AlyssaCoyne', 'batch1')
        self.PROCESSED_FOLDERS = os.path.join(self.PROCESSED_FOLDER_ROOT, "AlyssaCoyne",'batch1')
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "preprocessing", "AlyssaCoyne","batch1")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_spd", "SPDPreprocessor")
        self.RESCALE_INTENSITY = {
          'LOWER_BOUND': 0,
          'UPPER_BOUND': 100,
        }
        self.MARKERS_FOCUS_BOUNDRIES_PATH =  None
        self.TILE_INTERMEDIATE_SHAPE = (146,146)
        self.EXPECTED_IMAGE_SHAPE = (1022,1022)
        
