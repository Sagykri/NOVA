import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.model_config import ModelConfig

class NeuroselfConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        
        self.HOME_SUBFOLDER = os.path.join(self.MODELS_HOME_FOLDER, "neuroself")
        
        # Models
        self.MODEL_PATH = os.path.join(self.HOME_SUBFOLDER, "MODEL18_model_weights.0040.h5")
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'neuroself_models_outputs')
        
        self.EARLY_STOP_PATIENCE = None #10
        self.LEARN_RATE = None #2e-4
        self.BATCH_SIZE = None #32
        self.MAX_EPOCH = None #100
        
        self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]
