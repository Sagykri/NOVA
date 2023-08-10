import os
import sys
import datetime
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.model_config import ModelConfig

        
class CytoselfModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_cytoself_qsplit9')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        self.VQ_ARGS = [{'num_embeddings': 2048, 'embedding_dim': 64},
                        {'num_embeddings': 2048, 'embedding_dim': 64, 'channel_split':9}]

        # Models
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "checkpoints", "checkpoint_ep17.chkp")
        # Last checkpoint
        self.LAST_CHECKPOINT_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "checkpoints")

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 4 
        self.MAX_EPOCH = 100

        # Was calculated based 150 images per marker (num_markers=1311) from OpenCell data (Total of 71520 "site" images were sampled). site=16 tiles.
        self.DATA_VAR = 0.007928812876343727