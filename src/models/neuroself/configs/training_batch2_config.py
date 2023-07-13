import datetime
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.models.neuroself.configs.config import NeuroselfConfig

class NeuroselfBatch2DLTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch2_dl_new')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')

        
        # Models
        self.MODEL_PATH = None

        
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 8 #32 # = 2*~16 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based on all training data of batch2
        self.DATA_VAR = 0.00675623276886494


        
class NeuroselfB2SmallTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch2_small2')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 2#20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        self.DATA_VAR = 0.00675623276886494

class NeuroselfB2WTUnstressedTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch2_WT_untresssed3')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 2#20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        self.DATA_VAR = 0.00675623276886494
        
        self.FC_ARGS = {'num_layers': 3, 'act': 'swish'}
        
class NeuroselfB2WTUnstressedTrainingConfig4(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
       
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch2_WT_untresssed4')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 2#20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        self.DATA_VAR = 0.00675623276886494
        
        self.FC_INPUT_TYPE = 'vqindhist'
        
class TEMPNeuroselfB2WTUnstressedTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch2_WT_untresssed_TEMP')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 2#20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        self.DATA_VAR = 0.00675623276886494
        

class NeuroselfBatch2DLALLTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch2_dl_all_new')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')

        
        # Models
        self.MODEL_PATH = None

        
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 8 #32 # = 2*~16 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based on all training data of batch2
        self.DATA_VAR = 0.00675623276886494


        