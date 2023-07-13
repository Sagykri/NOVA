import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.base_config import BaseConfig


class EmbeddingsConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        # These are set according to the model selected to use for the inference 
        # For example, MOMAPS/src/outputs/models_outputs_batch8/embeddings/logs/
        self.EMBEDDINGS_FOLDER = None
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_piecewise', 'embeddings', 'logs')
        