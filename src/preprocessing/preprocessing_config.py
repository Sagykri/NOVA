import os

import sys
from typing import List, Tuple, Union
sys.path.insert(1, os.getenv("NOVA_HOME"))


from src.common.base_config import BaseConfig


class PreprocessingConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        # The path to the raw data folder
        self.RAW_FOLDER_ROOT:str = os.path.join(self.HOME_DATA_FOLDER, "images", "raw")
        # The path to the output (processed) folder
        self.PROCESSED_FOLDER_ROOT:str = os.path.join(self.HOME_DATA_FOLDER, "images", "processed")

        # Precaution - raw and processed folders can't be the same one!
        assert self.RAW_FOLDER_ROOT != self.PROCESSED_FOLDER_ROOT, f"RAW_FOLDER_ROOT == PROCESSED_FOLDER_ROOT, {self.RAW_FOLDER_ROOT}"
        
        # An array of all the folders to process
        self.INPUT_FOLDERS:List[str] = None
        # An array to where to save the processed files
        self.PROCESSED_FOLDERS:List[str] = None
        
        # The expected image shape
        self.EXPECTED_IMAGE_SHAPE:Tuple[int, int] = (1024,1024)
        # The tile shape when cropping the image into tiles
        self.TILE_INTERMEDIATE_SHAPE:Tuple[int,int] = (128,128)#(100,100)
        # The final tile shape after resizing from TILE_INTERMEDIATE_SHAPE
        self.TILE_SHAPE:Tuple[int, int] = (100,100)
        # Maximum allowed nuclei in a tile
        self.MAX_NUM_NUCLEI:int = 5
        # Threshold for minimal partial area of nuclei contained in tile.
        # If the ratio exceeds this value, the tile will be added.
        # float value between 0 and 1
        self.INCLUDED_AREA_RATIO:float = 0.8
        # The width of main image frame - to recognize nuceli intersecting with main frame
        self.FRAME_WIDTH_BUFFER:float = 1
        # Num of workers to use when running the preprocessing in parallel
        self.NUM_WORKERS:int = 6
        
        # Settings for cellpose
        # For more details please see: https://cellpose.readthedocs.io/en/latest/settings.html
        self.CELLPOSE = {
            'NUCLEUS_DIAMETER': 60,
            'CELLPROB_THRESHOLD': 0,
            'FLOW_THRESHOLD': 0.4
        }
        
        # The lower and upper bounds *percentiles* to shrink the image intenstiy into
        # Requirement: 0<=lower_bound<=upper_bound<=100
        # For more details see: https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity 
        self.RESCALE_INTENSITY = {
            'LOWER_BOUND': 0.5,
            'UPPER_BOUND': 99.9
        }
        
        # The path to the file holding the focus boundries for each marker
        self.MARKERS_FOCUS_BOUNDRIES_PATH:Union[None,str] = None

        # Threshold for filtering out empty tiles or tiles with dead cells  
        self.MAX_INTENSITY_THRESHOLD_TARGET:float = 0.2 # Before rescale intenisty
        self.VARIANCE_THRESHOLD_TARGET:float = 0.0001 # After rescale intenisty
        self.MAX_INTENSITY_THRESHOLD_NUCLEI:float = 0.2 # Before rescale intenisty
        self.VARIANCE_THRESHOLD_NUCLEI:float = 0.03 # After rescale intenisty
        self.MIN_ALIVE_NUCLEI_AREA: int = -1 # Minimum area of a nuclei to be considered alive (in pixels)
        self.MIN_MEDIAN_INTENSITY_NUCLEI_BLOB_THRESHOLD = 0.95 # Minimum median intensity of a nuclei blob to be considered dead (between 0 and 1)

            
        # Which markers to include
        self.MARKERS:Union[None, List[List]]            = None
        # Which markers to exclude
        self.MARKERS_TO_EXCLUDE:Union[None, List[List]] = ['CD41']
        # Cell lines to include
        self.CELL_LINES:Union[None, List[List]]         = None
        # Conditions to include
        self.CONDITIONS:Union[None, List[List]]         = None
        # Reps to include
        self.REPS:Union[None, List[List]]               = None
        # Panels to include
        self.PANELS:Union[None, List[List]]             = None

        
        # The path to the Preprocessor class (the path to the py file, then / and then the name of the class)
        # ex: os.path.join("src", "preprocessing", "preprocessor_spd", "SPDPreprocessor")
        self.PREPROCESSOR_CLASS_PATH:str = None

        
        #######################
        

        
        
