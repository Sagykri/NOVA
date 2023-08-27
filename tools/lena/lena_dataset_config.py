import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.dataset_config import DatasetConfig

        
class LenaDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()


        ######### *EDITABLE - SAFE TO EDIT* ###########
        
        # OPTIONAL configuration to use:
        
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant', 'SCNA', 'OPTN', ]
        # self.CONDITIONS = ['Untreated', 'stress']
        # self.MARKERS = ['ANXA11', 'Calreticulin', 'CD41', 'CLTC', 'DAPI', 'DCP1A', 'FMRP', 'FUS', 'G3BP1', GM130, KIF5A, LAMP1,
        #                 'mitotracker', 'NCL', 'NEMO', 'NONO', 'PEX14', 'Phalloidin', 'PML', 'PSD95', 'PURA', 'SCNA', 'SQSTM1', 'TDP43',
        #                 'TIA1', 'TOMM20']
                                
        self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant', 'TDP43']
        self.MARKERS = ['FUS', 'FMRP']#, 'G3BP1', 'PURA', 'TDP43']#['G3BP1', 'PURA', 'FMRP']#, 'Calreticulin', 'SQSTM1']#, 'NCL', 'TDP43', 'mitotracker', 'FMRP'] #['FUS']
        # self.CONDITIONS = ['Untreated']
        
        # You can add more folder like this: input_folders_names = ["batch9_16bit", "batch7", "batch6",] 
        input_folders_names = ['batch9_16bit_no_downsample']
        
        # Set this var to True if you 'input_folders_names' contains batches that the model used for training (ex. batch7/batch8), otherwise set to False
        self.SPLIT_DATA = False # True
        
        # Set to True if you don't have pre-calculated embeddings for the specified model & input_folders_names
        # (When set to False the results will arrive much faster)
        self.CALCULATE_EMBEDDINGS = False # True
        #######################################
        
        
        
        
        
        
        
        ########### PLEASE DON'T TOUCH THIS SECTION ##############
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        input_folders_names]
        #####################################