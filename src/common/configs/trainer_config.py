import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.base_config import BaseConfig


class TrainerConfig(BaseConfig):
    """Base trainer configuration
    """
    def __init__(self):
        super().__init__()
        
        self.OUTPUTS_FOLDER:str = None

        # Training parameters
        self.LR:float = None
        self.MIN_LR:float = None
        self.MAX_EPOCHS:int = None
        self.WARMUP_EPOCHS:int = None
        self.WEIGHT_DECAY:float = None
        self.WEIGHT_DECAY_END:float = None
        self.BATCH_SIZE:int = None
        self.NUM_WORKERS:int = 6
        self.EARLY_STOPPING_PATIENCE:int = 10
        self.DESCRIPTION:str = None
        self.TRAINER_CLASS_PATH:str = None
        
class ClassificationTrainerConfig(BaseConfig):
    """Trainer configuration for the classification model (pretrained model)
    """
    def __init__(self):
        super().__init__()
        
        self.OUTPUTS_FOLDER:str = None

        # Training parameters
        self.LR:float = 0.0008
        self.MIN_LR:float = 1e-6
        self.MAX_EPOCHS:int = 300
        self.WARMUP_EPOCHS:int = 5
        self.WEIGHT_DECAY:float = 0.04
        self.WEIGHT_DECAY_END:float = 0.4
        self.BATCH_SIZE:int = 300
        self.DESCRIPTION:str = "Pretrained model trained with CE loss on the Opencell dataset"
        self.TRAINER_CLASS_PATH:str = os.path.join("src", "common", "lib", "models", "trainers", "trainer_classification", "TrainerClassification")

class ContrastiveTrainerConfig(BaseConfig):
    """Trainer configuration for the contrastive learning model (finetuned model)
    """
    def __init__(self):
        super().__init__()
        
        self.OUTPUTS_FOLDER:str = None

        # Training parameters
        self.LR:float = 0.0008
        self.MIN_LR:float = 1e-6
        self.MAX_EPOCHS:int = 300
        self.WARMUP_EPOCHS:int = 5
        self.WEIGHT_DECAY:float = 0.04
        self.WEIGHT_DECAY_END:float = 0.4
        self.BATCH_SIZE:int = 300
        self.DESCRIPTION:str = "Pretrained model trained with CE loss on the Opencell dataset"
        self.TRAINER_CLASS_PATH:str = os.path.join("src", "common", "lib", "models", "trainers", "trainer_contrastive", "TrainerContrastive")

        self.PRETRAINED_MODEL_NUM_CLASSES = 1311
        self.PRETRAINED_MODEL_PATH = os.path.join(os.getenv("MOMAPS_HOME"), "outputs", "vit_models", "opencell_new", "checkpoints_010824_171221_851216_26417_training_pretrained_model", "checkpoint_best.pth")