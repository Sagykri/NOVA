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

class FineTuningClassificationWithBatchModelConfig(ModelConfig):
    """Configuration for the fine tuned classification model
    """
    def __init__(self):
        super().__init__()
        
        self.OUTPUT_DIM = 216*2
        
class ContrastivenModelConfig(ModelConfig):
    """Configuration for the contrastive learning model
    """
    def __init__(self):
        super().__init__()
        
        self.OUTPUT_DIM = 128