import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.base_config import BaseConfig


class DatasetConfig(BaseConfig):
    def __init__(self):
        
        super().__init__()
        
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.HOME_DATA_FOLDER, "images", "processed")
        
        # The path to the data folders
        self.INPUT_FOLDERS = None
        # Which markers to include
        self.MARKERS = None
        # Which markers to exclude
        self.MARKERS_TO_EXCLUDE = ['TIA1'] 
        # Cell lines to include
        self.CELL_LINES = None
        # Conditions to include
        self.CONDITIONS = None
        # Reps to include
        self.REPS       = None
        # Should split the data to train,val,test?
        self.SPLIT_DATA = True
        # The percentage of the data that goes to the training set
        self.TRAIN_PCT = 0.7
        # Should shuffle the data within each batch collected?
        ##Must be true whenever using SPLIT_DATA=True otherwise train,val,test set won't be the same as when shuffle was true
        self.SHUFFLE = True     
        
        # Should add the cell line to the label?
        self.ADD_LINE_TO_LABEL = True
        # Should add condition to the label?
        self.ADD_CONDITION_TO_LABEL = True 
        # Should add the batch to the label?
        self.ADD_BATCH_TO_LABEL = False
        # Should add the rep to the label?
        self.ADD_REP_TO_LABEL = False

        # Number of channels per image
        self.NUM_CHANNELS = 2
        # The size of each image (width,height)
        self.IMAGE_SIZE = (100,100)