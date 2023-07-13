import os
import sys
import datetime
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.model_config import ModelConfig


class CytoselfTrainingConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_cytoself')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None
        
        # Last checkpoint
        self.LAST_CHECKPOINT_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "checkpoints", "checkpoint_ep1.chkp")

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 4 
        self.MAX_EPOCH = 100

        # Was calculated based 150 images per marker (num_markers=1311) from OpenCell data (Total of 71520 "site" images were sampled). site=16 tiles.
        self.DATA_VAR = 0.007928812876343727
        
class Cytoself16TrainingConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_cytoself16')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 8
        self.MAX_EPOCH = 100

        # Was calculated based 150 images per marker (num_markers=1311) from OpenCell data (Total of 71520 "site" images were sampled). site=16 tiles.
        self.DATA_VAR = 0.007928812876343727
        