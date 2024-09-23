import os
import sys
from typing import List, Tuple
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.common.base_config import BaseConfig


class DatasetConfig(BaseConfig):
    def __init__(self):
        
        super().__init__()
        
        # The path to the root of the processed folder
        self.PROCESSED_FOLDER_ROOT:str = os.path.join(self.HOME_DATA_FOLDER, "images", "processed")
        
        # The path to the data folders
        self.INPUT_FOLDERS:List[str]      = None
        # Which markers to include
        self.MARKERS:List[str]            = None
        # Which markers to exclude
        self.MARKERS_TO_EXCLUDE:List[str] = ['TIA1'] 
        # Cell lines to include
        self.CELL_LINES:List[str]         = None
        # Conditions to include
        self.CONDITIONS:List[str]         = None
        # Reps to include
        self.REPS:List[str]               = None
        # Should split the data to train,val,test?
        self.SPLIT_DATA:bool              = True
        # The percentage of the data that goes to the training set
        self.TRAIN_PCT:float              = 0.7
        # Should shuffle the data within each batch collected?
        ##Must be true whenever using SPLIT_DATA=True otherwise train,val,test set won't be the same as when shuffle was true
        self.SHUFFLE:bool                 = True     
        
        # Should add the cell line to the label?
        self.ADD_LINE_TO_LABEL:bool = True
        # Should add condition to the label?
        self.ADD_CONDITION_TO_LABEL:bool = True 
        # Should add the batch to the label?
        self.ADD_BATCH_TO_LABEL:bool = True
        # Should add the rep to the label?
        self.ADD_REP_TO_LABEL:bool = True

        # Number of channels per image
        self.NUM_CHANNELS:int = 2
        # The size of each image (width,height)
        self.IMAGE_SIZE:Tuple[int, int] = (100,100)