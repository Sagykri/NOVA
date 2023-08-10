import datetime
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.models.neuroself.configs.config import NeuroselfConfig

class NeuroselfLenaConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        ################### *EDITABLE - SAFE TO EDIT* ####################
        
        # Please specify the path to the model
        self.MODEL_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/outputs/models_outputs_batch78_16bit_shuffle_tl/checkpoints/checkpoint_ep6.chkp"
        
        ###############################################
        
        
        ################# PLEASE DON'T TOUCH THIS SECTION ######################
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.HOME_FOLDER, "tools", "lena", 'models_outputs_Lena')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 30#4
        self.MAX_EPOCH = 100

        self.DATA_VAR = 0.00939656708666626    
        #################################################################
        
        