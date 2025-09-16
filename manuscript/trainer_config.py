import os

import sys
from typing import List
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.models.trainers.trainer_config import TrainerConfig


class ClassificationTrainerConfig(TrainerConfig):
    """Trainer configuration for the classification model (pretrained model)
    """
    def __init__(self):
        super().__init__()
        
        self.OUTPUTS_FOLDER:str = os.path.join(os.getenv("NOVA_HOME"), "outputs/vit_models/pretrained_model")

        # Training parameters
        self.LR:float = 0.0008
        self.MIN_LR:float = 1e-6
        self.MAX_EPOCHS:int = 300
        self.WARMUP_EPOCHS:int = 5
        self.WEIGHT_DECAY:float = 0.04
        self.WEIGHT_DECAY_END:float = 0.4
        self.BATCH_SIZE:int = 350 
        self.DESCRIPTION:str = "Pretrained model trained with CE loss on the Opencell dataset"
        self.TRAINER_CLASS_PATH:str = os.path.join("src", "models", "trainers", "trainer_classification", "TrainerClassification")

class ContrastiveTrainerNoFreezeConfig_MLPHead_acrossBatches_B56789_80pct(TrainerConfig):
    """Trainer configuration for the contrastive learning model (finetuned model)
    """
    def __init__(self):
        super().__init__()
       
        self.OUTPUTS_FOLDER:str = os.path.join(os.getenv("NOVA_HOME"), "outputs/vit_models/finetunedModel_MLPHead_acrossBatches_B56789_80pct_noFreeze")

        # Training parameters
        self.LR:float = 0.0008
        self.MIN_LR:float = 1e-6
        self.MAX_EPOCHS:int = 300
        self.WARMUP_EPOCHS:int = 5
        self.WEIGHT_DECAY:float = 0.04
        self.WEIGHT_DECAY_END:float = 0.4
        self.BATCH_SIZE:int = 750
        self.DESCRIPTION:str = "Finetuned model using contrastive learning on neuronal dataset without freezing any layer (with MLP head) (pos and neg are across batches)"
        self.TRAINER_CLASS_PATH:str = os.path.join("src", "models", "trainers", "trainer_contrastive", "TrainerContrastive")
        self.DROP_LAST_BATCH:bool = True
        self.NEGATIVE_COUNT:int = 5
        self.ACROSS_BATCHES: bool = True
        
        # Original pretraiend
        self.PRETRAINED_MODEL_PATH:str = os.path.join(os.getenv("NOVA_HOME"), "outputs/vit_models/pretrained_model/checkpoints/checkpoint_best.pth")


class ContrastiveTrainerFrozenConfig_MLPHead_acrossBatches_B56789_80pct(TrainerConfig):
    """Trainer configuration for the contrastive learning model (finetuned model)
    """
    def __init__(self):
        super().__init__()
       
        self.OUTPUTS_FOLDER:str = os.path.join(os.getenv("NOVA_HOME"), "outputs/vit_models/finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen")

        # Training parameters
        self.LR:float = 0.0008
        self.MIN_LR:float = 1e-6
        self.MAX_EPOCHS:int = 300
        self.WARMUP_EPOCHS:int = 5
        self.WEIGHT_DECAY:float = 0.04
        self.WEIGHT_DECAY_END:float = 0.4
        self.BATCH_SIZE:int = 750
        self.DESCRIPTION:str = "Finetuned model using contrastive learning on neuronal dataset with frozen layers (with MLP head) (pos and neg are across batches)"
        self.TRAINER_CLASS_PATH:str = os.path.join("src", "models", "trainers", "trainer_contrastive", "TrainerContrastive")
        self.DROP_LAST_BATCH:bool = True
        self.NEGATIVE_COUNT:int = 5
        self.ACROSS_BATCHES: bool = True
        
        # Original pretraiend
        self.PRETRAINED_MODEL_PATH:str = os.path.join(os.getenv("NOVA_HOME"), "outputs/vit_models/pretrained_model/checkpoints/checkpoint_best.pth")

        self.LAYERS_TO_FREEZE:List[str] = ['blocks.5.norm2.bias', 'blocks.11.norm1.bias', 'blocks.3.norm2.bias',
                                            'blocks.2.attn.proj.weight', 'blocks.11.norm2.bias', 'blocks.2.norm1.bias',
                                            'blocks.8.norm2.bias', 'blocks.10.norm1.bias', 'blocks.3.attn.proj.bias',
                                            'blocks.9.norm1.bias', 'blocks.6.norm1.bias', 'blocks.6.norm2.bias',
                                            'blocks.2.attn.qkv.bias', 'blocks.10.norm2.bias', 'blocks.6.attn.qkv.bias',
                                            'blocks.7.norm1.bias', 'blocks.7.norm2.bias', 'cls_token',
                                            'blocks.8.norm1.bias', 'blocks.5.norm1.bias', 'blocks.0.norm2.bias',
                                            'blocks.5.attn.qkv.bias', 'blocks.3.attn.qkv.bias', 'blocks.4.norm1.bias',
                                            'blocks.2.norm2.bias', 'blocks.1.attn.proj.bias', 'blocks.8.mlp.fc1.bias',
                                            'blocks.3.norm1.bias', 'blocks.7.mlp.fc1.bias', 'blocks.6.mlp.fc1.bias',
                                            'blocks.2.attn.proj.bias', 'blocks.4.attn.qkv.bias',
                                            'blocks.0.norm1.weight', 'blocks.11.attn.qkv.bias', 'blocks.2.mlp.fc2.bias',
                                            'blocks.1.mlp.fc2.bias', 'blocks.0.norm1.bias', 'blocks.7.attn.qkv.bias',
                                            'blocks.8.attn.qkv.bias', 'blocks.9.mlp.fc1.bias', 'blocks.0.mlp.fc1.bias',
                                            'blocks.9.attn.qkv.bias', 'blocks.5.mlp.fc1.bias', 'blocks.10.mlp.fc1.bias',
                                            'blocks.4.mlp.fc1.bias', 'blocks.10.attn.qkv.bias', 'blocks.0.mlp.fc2.bias',
                                            'blocks.3.mlp.fc1.bias', 'blocks.11.mlp.fc1.bias', 'blocks.1.mlp.fc1.bias',
                                            'blocks.0.norm2.weight', 'blocks.2.mlp.fc1.bias', 'blocks.1.norm1.weight',
                                            'blocks.1.norm2.weight', 'blocks.2.norm1.weight', 'blocks.3.norm1.weight',
                                            'blocks.5.norm1.weight', 'blocks.9.norm1.weight', 'blocks.7.norm1.weight',
                                            'blocks.6.norm1.weight', 'blocks.10.norm1.weight', 'blocks.4.norm1.weight',
                                            'blocks.11.norm1.weight', 'blocks.2.norm2.weight', 'blocks.8.norm1.weight',
                                            'blocks.7.norm2.weight', 'blocks.8.norm2.weight', 'blocks.4.norm2.weight',
                                            'blocks.3.norm2.weight', 'blocks.6.norm2.weight', 'blocks.9.norm2.weight',
                                            'blocks.5.norm2.weight', 'blocks.11.norm2.weight', 'blocks.10.norm2.weight',
                                            'norm.weight']
