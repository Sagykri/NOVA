import datetime
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.models.neuroself.configs.config import NeuroselfConfig

class NeuroselfTestingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch678_bs256')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs', "test")
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", "test", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))
        
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "checkpoints", "model_weights.0007.h5")



        
        # self.EARLY_STOP_PATIENCE = 20
        # self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 64#100000#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        # self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        # self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]



class CytoTestingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_cyto')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs', "test")
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", "test", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))
        
        self.MODEL_PATH = self.PRETRAINED_MODEL_PATH#os.path.join(self.MODEL_OUTPUT_FOLDER, "checkpoints", "model_weights.0059.h5")



        
        # self.EARLY_STOP_PATIENCE = 20
        # self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 64#100000#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        # self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        # self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]


