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
        # self.REPS = ['rep1', 'rep2']
        
                  
        self.CELL_LINES = ['WT', 'TDP43']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant', 'TDP43']
        #self.MARKERS = ['Rab5', 'SCNA', 'SQSTM1', 'TIA1', 'TOMM20']#, 'G3BP1', 'PURA', 'TDP43']#['G3BP1', 'PURA', 'FMRP']#, 'Calreticulin', 'SQSTM1']#, 'NCL', 'TDP43', 'mitotracker', 'FMRP'] #['FUS']
        self.CONDITIONS = ['Untreated', 'DMSO100uM', 'Riluzole'] # Perturbations: Untreated, DMSO one of them, and one perturbation
        self.EXPERIMENT_TYPE  = 'perturbations_confocal'
        
        # You can add more folder like this: input_folders_names = ["batch9_16bit", "batch7", "batch6",] 
        input_folders_names = ['Perturbations_spd_format']
        
        ### For multiple batches: ###
        #batch_names = ['batch2_16bit_no_downsample', 'batch3_16bit_no_downsample', 'batch4_16bit_no_downsample', 'batch5_16bit_no_downsample'] # Perturbations_16bit_no_downsample
        #input_folders_names = [os.path.join('deltaNLS', batch) for batch in batch_names]
        
        # Set this var to True if you 'input_folders_names' contains batches that the model used for training (ex. batch7/batch8), otherwise set to False
        self.SPLIT_DATA = False # True
                
        # Your can set self.REPS to a specific rep or leave it None to load the two reps 
        self.REPS = ['rep1', 'rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']
        # You can set this var to True if you want the UMAP to color the reps with different colors
        self.ADD_REP_TO_LABEL = False
        # You can set this var to True if you want the UMAP to color the batches with different colors
        self.ADD_BATCH_TO_LABEL = False
        #######################################
        
        
        
        
        
        
        
        ########### PLEASE DON'T TOUCH THIS SECTION ##############
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 input_folders_names]
        
        ## ONLY FOR CONFOCAL: ##
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Confocal", f) for f in 
                                ["Perturbations_spd_format"]]
        
        self.EMBEDDINGS_LAYER = 'vqvec2'
        
        #self.EXPERIMENT_TYPE = 'neurons' 
        #####################################