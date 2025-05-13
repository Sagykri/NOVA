import os

import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.models.architectures.model_config import ModelConfig

class ClassificationModelConfig(ModelConfig):
    """Configuration for the classification model
    """
    def __init__(self):
        super().__init__()
        
        self.OUTPUT_DIM = 1311
        
class ContrastiveModelConfig(ModelConfig):
    """Configuration for the contrastive learning model
    """
    def __init__(self):
        super().__init__()
        
        self.OUTPUT_DIM = 128