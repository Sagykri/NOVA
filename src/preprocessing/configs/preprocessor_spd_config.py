import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))



from src.common.configs.preprocessing_config import PreprocessingConfig

class SPDPreprocessingConfig(PreprocessingConfig):
    # Break the 1024x1024 input image (after it has been normalized) 
    # to 16 tiles of 256x256, and then downsample them to 100x100 
    # (down-sample by 2 so tile is 128x128 and then resize tile to 100x100)
    def __init__(self):
        super().__init__()
        
        self.RAW_SUBFOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'SpinningDisk')
        self.PROCESSED_SUBFOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk")
        self.OUTPUTS_SUBSUBFOLDER = os.path.join(self.OUTPUTS_SUBFOLDER, "spd")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_spd", "SPDPreprocessor")
        self.TO_DOWNSAMPLE = True
        self.TILE_WIDTH = 256 
        self.TILE_HEIGHT = 256
        self.WITH_NUCLEUS_DISTANCE = False
