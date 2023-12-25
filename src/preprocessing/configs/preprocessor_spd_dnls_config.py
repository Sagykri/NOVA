import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))



from src.preprocessing.configs.preprocessor_spd_config import SPDPreprocessingConfig

class SPDPreprocessingConfigdNLS(SPDPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.TO_DOWNSAMPLE = False
        self.TILE_WIDTH = 128
        self.TILE_HEIGHT = 128
        self.WITH_NUCLEUS_DISTANCE = False
        
        self.BRENNER_BOUNDS_PATH =  os.path.join(os.getenv("MOMAPS_HOME"), 'src', 'preprocessing', 'sites_validity_bounds_dNLS.csv')

