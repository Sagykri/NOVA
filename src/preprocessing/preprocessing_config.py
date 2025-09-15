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
        self.MAX_NUM_NUCLEI:int = 5 # NOVA - 5
        # Threshold for minimal partial area of nuclei contained in tile.
        # If the ratio exceeds this value, the tile will be added.
        # float value between 0 and 1
        self.INCLUDED_AREA_RATIO:float = 0.8 # NOVA -0.8
        # The width of main image frame - to recognize nuceli intersecting with main frame
        self.FRAME_WIDTH_BUFFER:float = 1
        # Num of workers to use when running the preprocessing in parallel
        self.NUM_WORKERS:int = 6
        
        # Settings for cellpose
        # For more details please see: https://cellpose.readthedocs.io/en/latest/settings.html
        self.CELLPOSE = {
            'NUCLEUS_DIAMETER': 70, # old funova 70 # nova 60,
            'CELLPROB_THRESHOLD': 0,# old funova 0 # nova 0,
            'FLOW_THRESHOLD': 0.22 # old funova 0.22 # nova 0.4,
        }
        
        # The lower and upper bounds *percentiles* to shrink the image intenstiy into
        # Requirement: 0<=lower_bound<=upper_bound<=100
        # For more details see: https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity 
        self.RESCALE_INTENSITY = { # PER CHANNEL
            'LOWER_BOUND': [2.5 ,0.5], # Marker, DAPI
            'UPPER_BOUND': [99.45, 100] # Marker, DAPI
        }
        
        # The path to the file holding the focus boundries for each marker
        self.MARKERS_FOCUS_BOUNDRIES_PATH:Union[None,str] = None

        # Threshold for filtering out empty tiles or tiles with dead cells
        # TARGET (MARKER)
        # # Before rescale intenisty  
        self.MAX_INTENSITY_THRESHOLD_TARGET:float =  0 # old funova - 0.2 # none: 0 # NOVA: 0.2 
        self.MAX_INTENSITY_UPPER_BOUND_THRESHOLD_TARGET:float = 1.1 # NEW THRESHOLD # None:1.1
        # After rescale intenisty - lower bound for variance
        self.VARIANCE_THRESHOLD_TARGET:float = 0.001 # old funova - 0.003 # none: 0 # NOVA: 0.0001 
        # New threshold - upper bound for target's variance
        self.VARIANCE_UPPER_BOUND_THRESHOLD_TARGET:float = 0.14 #0.135 # None: 1.1

        # NUCLEI (DAPI)
        # Before rescale intenisty
        self.MAX_INTENSITY_THRESHOLD_NUCLEI:float = 0.17 # old funova  0.2 # none: 0 # NOVA: 0.2 
        # After rescale intenisty
        self.VARIANCE_THRESHOLD_NUCLEI:float = 0.028 # old funova - 0.02 # none: 0 # NOVA: 0.03 

        # Threshold for fitering ALIVE Nucleus detected in [__is_contains_dead_cells]
        # detecting blobs by thesholding to signal vs. background
         # total number of blobs in tile
        self.MAX_NUM_NUCLEI_BLOB:int = 10 #12?# old funova 12 # None - ~500

        # Minimum area of a nuclei to be considered alive (in pixels)
        self.MIN_ALIVE_NUCLEI_AREA: int = 900 #old funova 700 # none:-1 #  NOVA: 800 
        # NEW THRESHOLDS

        # Thresholds for filtering ALIVE cell
        # maximum area of an alive nuclei (above is probably noise or a smear)
        self.MAX_ALIVE_NUCLEI_AREA: int = 4200  
        # Or
        # either below both minimal thresholds
        self.MIN_VARIANCE_THRESHOLD_ALIVE_NUCLEI: float = 0.006 #0.006 # old funova 0.01 # None 0.0
        self.MIN_MEDIAN_INTENSITY_THRESHOLD_ALIVE_NUCLEI: float = 0.6 #0.7? #0.25 # old funova 0.25 # None 0.25
        # or above both maximal thresholds
        self.MAX_VARIANCE_THRESHOLD_ALIVE_NUCLEI: float = 0.025 #0.28?#0.023 # 0.028 # old funova 0.03 # None 1.0
        self.MAX_MEDIAN_INTENSITY_THRESHOLD_ALIVE_NUCLEI: float = 0.95#0.97 # 0.875 # old funova 0.6 # None 1.0

        # Threshold for fitering DEAD Nucleus detected in [__is_contains_dead_cells]
        # Minimum median intensity of a nuclei blob to be considered dead (between 0 and 1)
        self.MIN_NUCLEI_BLOB_AREA:int = 450 # minimum size for a blob to be considered as dead cell
        # AND
        self.MIN_MEDIAN_INTENSITY_NUCLEI_BLOB_THRESHOLD:float =  0.63 #0.55  # old funova 0.4 #  none: 1.9 # NOVA: 0.95 
        # AND
        # (
        # below "BOTTOM THRESHOLD"
        self.MAX_VARIANCE_NUCLEI_BLOB_THRESHOLD:float =  0.01 #0.0065 # old funova 0.005 # None - 0.0
        # or
        # above "UPPER THRESHOLD"
        self.MIN_VARIANCE_NUCLEI_BLOB_THRESHOLD:float = 0.0355# old funova 0.025  # None - 0.0
        # )
       
            
        # Which markers to include
        self.MARKERS:Union[None, List[List]]            = None
        # Which markers to exclude
        self.MARKERS_TO_EXCLUDE:Union[None, List[List]] = None 
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
        

        
        
