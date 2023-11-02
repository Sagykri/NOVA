import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.dataset_config import DatasetConfig

        
class LenaDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()


        ######### *EDITABLE - SAFE TO EDIT* ###########
        
        # OPTIONAL configuration to use:
        
        #self.CELL_LINES = ['WT', 'TDP43', 'TBK1', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant', 'OPTN']# ['WT', 'TDP43', 'TBK1', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant', 'SCNA', 'OPTN', ]
        self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']# 
        ##self.CELL_LINES =  ['WT', 'TDP43', 'TBK1', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant', 'SCNA', 'OPTN', ]
        self.CONDITIONS = ['Untreated'] # ['Untreated', 'stress']
        self.MARKERS = ['ANXA11',
                        'Calreticulin', 
                        'CD41', 
                        'CLTC', 
                        'DAPI', 
                        'DCP1A',
                        'FMRP', 
                        #'FUS', 
                        'G3BP1',
                        'GM130', 
                        'KIF5A', 
                        'LAMP1',  
                        'mitotracker', 
                        'NCL', 
                        'NEMO', 
                        'NONO', 
                        'PEX14', 
                        'Phalloidin', 
                        'PML', 
                        'PSD95', 
                        'PURA', 
                        'SCNA', 
                        'SQSTM1', 
                        'TDP43', 
                        'TIA1', 
                        'TOMM20'
                        ]
        
        # self.CELL_LINES = ['WT', 'KO'] 
        # self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['LAMP1', 'FUS', 'PML']
        
        # running
        #
        # Set this var to True if you 'input_folders_names' contains batches that the model used for training (ex. batch7/batch8), otherwise set to False
        self.SPLIT_DATA = False # True
                
        # Your can set self.REPS to a specific rep or leave it None to load the two reps 
        self.REPS = ['rep1'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']
        # You can set this var to True if you want the UMAP to color the reps with different colors
        self.ADD_REP_TO_LABEL = False
        # You can set this var to True if you want the UMAP to color the batches with different colors
        self.ADD_BATCH_TO_LABEL = False
        # You can set whether to use the global representation (vqvec2) or the local representation (vqvec1)
        self.EMBEDDINGS_LAYER = 'vqvec2' # 'vqvec1' / 'vqvec2' / 'vqvec_both'
        # You can set from what experiment (the name of the folder) to pull the embeddings
        self.EXPERIMENT_TYPE = 'neurons' # 'neurons', 'perturbations', 'opencell', 'deltaNLS' , 'NiemannPick'
        #######################################
        
        
        # You can add more folder like this: input_folders_names = ["batch9_16bit", "batch7", "batch6",] 
        input_folders_names = ['batch3_16bit_no_downsample']
        # 
        # ['batch7_16bit_no_downsample', 'batch8_16bit_no_downsample'], 
        # ['OpenCell'] 
        # 'batch9_16bit_no_downsample', 'batch6_16bit_no_downsample', 'batch5_16bit_no_downsample', 'batch4_16bit_no_downsample', 'batch3_16bit_no_downsample', 
        # 'Perturbations_16bit_no_downsample', 
        # 'Perturbations_old_PP_Feb2023', 
        # 'NiemannPick', 
        # 'deltaNLS'  
        
        
        ########### PLEASE DON'T TOUCH THIS SECTION ##############
        ########### Neurons ##############
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        input_folders_names]
        
        ########### NiemannPick ##############
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "NiemannPick", f) for f in 
        #                 input_folders_names]
        #####################################