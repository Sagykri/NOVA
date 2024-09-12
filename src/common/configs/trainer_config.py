import os

import sys
from typing import List
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.base_config import BaseConfig


class TrainerConfig(BaseConfig):
    """Base trainer configuration
    """
    def __init__(self):
        super().__init__()

        # Training parameters
        
        # The starting learning rate
        self.LR:float = None
        # The final learning rate at the end of the schedule
        self.MIN_LR:float = None
        # Number of epochs
        self.MAX_EPOCHS:int = None
        # Number of epochs to warmup the learning rate
        self.WARMUP_EPOCHS:int = None
        # The starting weight decay value
        self.WEIGHT_DECAY:float = None
        # The final weight decay value at the end of the schedule
        self.WEIGHT_DECAY_END:float = None
        # The batchsize (how many files to load per batch)
        self.BATCH_SIZE:int = None
        # Number of works to run during the data loading
        self.NUM_WORKERS:int = 6
        # Number of straight epochs without improvement to wait before activating eary stopping 
        self.EARLY_STOPPING_PATIENCE:int = 10
        # The path to the trainer class (the path to the py file, then / and then the name of the class)
        # ex: os.path.join("src", "common", "lib", "models", "trainers", "trainer_classification", "TrainerClassification")
        self.TRAINER_CLASS_PATH:str = None
        # A textual description for the model (optional, default to the trainer class name)
        self.DESCRIPTION:str = None
        # Whether to drop the last batch if its partial of the expected batch size
        self.DROP_LAST_BATCH:bool = False 
        # The path to the aumentation to apply on each sample in the data (the path to the py file, then / and then the name of the class)
        # ex: os.path.join("src", "common", "lib", "models", "trainers", "utils", "augmentations", "RotationsAndFlipsAugmentation")
        self.DATA_AUGMENTATION_CLASS_PATH:str = os.path.join("src", "common", "lib", "models", "trainers", "utils", "augmentations", "RotationsAndFlipsAugmentation")
                           
class ClassificationTrainerConfig(TrainerConfig):
    """Trainer configuration for the classification model (pretrained model)
    """
    def __init__(self):
        super().__init__()
        
        self.OUTPUTS_FOLDER:str = "/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/pretrained_model"

        # Training parameters
        self.LR:float = 0.0008
        self.MIN_LR:float = 1e-6
        self.MAX_EPOCHS:int = 300
        self.WARMUP_EPOCHS:int = 5
        self.WEIGHT_DECAY:float = 0.04
        self.WEIGHT_DECAY_END:float = 0.4
        self.BATCH_SIZE:int = 350 
        self.DESCRIPTION:str = "Pretrained model trained with CE loss on the Opencell dataset"
        self.TRAINER_CLASS_PATH:str = os.path.join("src", "common", "lib", "models", "trainers", "trainer_classification", "TrainerClassification")
 
class ContrastiveTrainerConfig(TrainerConfig):
    """Trainer configuration for the contrastive learning model (finetuned model)
    """
    def __init__(self):
        super().__init__()
       
        self.OUTPUTS_FOLDER:str = "/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/finetuned_model"

        # Training parameters
        self.LR:float = 0.0008
        self.MIN_LR:float = 1e-6
        self.MAX_EPOCHS:int = 300
        self.WARMUP_EPOCHS:int = 5
        self.WEIGHT_DECAY:float = 0.04
        self.WEIGHT_DECAY_END:float = 0.4
        self.BATCH_SIZE:int = 750
        self.DESCRIPTION:str = "Finetuned model using contrastive learning on neuronal dataset"
        self.TRAINER_CLASS_PATH:str = os.path.join("src", "common", "lib", "models", "trainers", "trainer_contrastive", "TrainerContrastive")
        self.DROP_LAST_BATCH:bool = True
        self.NEGATIVE_COUNT:int = 5
        
        # Original pretraiend
        self.PRETRAINED_MODEL_PATH:str = "/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/pretrained_model/checkpoints/checkpoint_best.pth"

        
        self.LAYERS_TO_FREEZE:List[str] = ['blocks.10.norm1.bias', 'blocks.3.attn.proj.bias', 'cls_token',
                                            'blocks.2.mlp.fc1.bias', 'blocks.3.attn.qkv.bias',
                                            'blocks.10.attn.qkv.bias', 'blocks.1.mlp.fc2.bias', 'blocks.4.mlp.fc1.bias',
                                            'blocks.1.mlp.fc1.bias', 'blocks.11.mlp.fc1.bias', 'blocks.2.attn.qkv.bias',
                                            'blocks.3.mlp.fc1.bias', 'blocks.2.attn.proj.bias',
                                            'blocks.0.attn.proj.bias', 'blocks.0.mlp.fc2.bias', 'blocks.9.norm1.bias',
                                            'blocks.5.mlp.fc1.bias', 'blocks.6.attn.qkv.bias', 'blocks.6.mlp.fc1.bias',
                                            'blocks.1.attn.proj.bias', 'blocks.7.mlp.fc1.bias', 'blocks.9.mlp.fc1.bias',
                                            'blocks.1.norm2.bias', 'blocks.10.mlp.fc1.bias', 'blocks.8.mlp.fc1.bias',
                                            'blocks.8.norm1.bias', 'blocks.7.attn.qkv.bias', 'blocks.3.norm1.bias',
                                            'blocks.1.norm1.bias', 'blocks.9.attn.qkv.bias', 'blocks.5.attn.qkv.bias',
                                            'blocks.7.norm1.bias', 'blocks.0.norm2.bias', 'blocks.2.norm2.bias',
                                            'blocks.6.norm1.bias', 'blocks.4.attn.qkv.bias', 'blocks.2.norm1.bias',
                                            'blocks.3.norm2.bias', 'blocks.4.norm1.bias', 'blocks.5.norm2.bias',
                                            'blocks.6.norm2.bias', 'blocks.0.norm1.bias', 'blocks.5.norm1.bias',
                                            'blocks.0.attn.qkv.bias', 'blocks.4.norm2.bias', 'blocks.7.norm2.bias',
                                            'blocks.9.norm2.bias', 'blocks.8.norm2.bias', 'blocks.11.norm2.bias',
                                            'blocks.10.norm2.bias', 'blocks.10.norm1.weight', 'blocks.0.norm1.weight',
                                            'blocks.11.norm1.weight', 'blocks.6.norm1.weight', 'blocks.8.norm1.weight',
                                            'blocks.9.norm1.weight', 'blocks.0.norm2.weight', 'blocks.11.norm2.weight',
                                            'blocks.7.norm1.weight', 'blocks.5.norm1.weight', 'blocks.4.norm2.weight',
                                            'blocks.6.norm2.weight', 'blocks.3.norm2.weight', 'blocks.4.norm1.weight',
                                            'blocks.9.norm2.weight', 'blocks.5.norm2.weight', 'blocks.3.norm1.weight',
                                            'blocks.10.norm2.weight', 'blocks.7.norm2.weight', 'blocks.2.norm1.weight',
                                            'blocks.1.norm2.weight', 'blocks.2.norm2.weight', 'blocks.8.norm2.weight',
                                            'blocks.1.norm1.weight', 'norm.weight']