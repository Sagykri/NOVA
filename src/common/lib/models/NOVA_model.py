import os
import sys
import torch
import numpy as np
from typing import Self

sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.lib.utils import get_if_exists
from src.common.lib.models.model_utils import load_checkpoint_from_file
from src.common.lib import embeddings_utils
from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.model_config import ModelConfig
from src.common.lib.models import vision_transformer

class NOVAModel():

    def __init__(self, model_config:ModelConfig):
        self.__set_params(model_config)
        self.model = self.__get_vit()        
    
    def __set_params(self, model_config:ModelConfig):
        """Extracting params from the configuration

        Args:
            model_config (ModelConfig): The configuration
        """
        self.model_config = model_config
        self.vit_version = get_if_exists(self.model_config, 'VIT_VERSION', 'tiny')
        self.image_size = get_if_exists(self.model_config, 'IMAGE_SIZE', 100)
        self.patch_size = get_if_exists(self.model_config, 'PATCH_SIZE', 14)
        self.num_channels = get_if_exists(self.model_config, 'NUM_CHANNELS', 2)
        self.num_classes = self.model_config['NUM_CLASSES']

    def __get_vit(self):
        vit_version = self.vit_version
        
        if vit_version == 'base':
            create_vit = vision_transformer.vit_base
        elif vit_version == 'small':
            create_vit = vision_transformer.vit_small
        elif vit_version == 'tiny':
            create_vit = vision_transformer.vit_tiny
        else:
            raise Exception(f"Invalid 'vit_version' detected: {vit_version}. Must be 'base', 'small' or 'tiny'")
        
        vit = create_vit(
                img_size=[self.image_size],
                patch_size=self.patch_size,
                in_chans=self.num_channels,
                num_classes=self.num_classes
        )
        
        return vit
    
    def generate_embeddings(self, dataset_config: DatasetConfig)->np.ndarray:
        """Generate embeddings for the given data using the model

        Args:
            dataset_config (DatasetConfig): The configuration for indicating what data to load

        Returns:
            np.ndarray: The embeddings
        """
        return embeddings_utils.generate_embeddings(self, dataset_config)
    
    @staticmethod
    def load_from_checkpoint(ckp_path: str)->Self:
        """Get model from checkpoint

        Args:
            ckp_path (str): path to checkpoint

        Returns:
            model (NOVAModel): The NOVA model
        """
        checkpoint = load_checkpoint_from_file(ckp_path)
        state_dict = checkpoint['model']
        model_config = checkpoint['model_config']
        training_config = checkpoint['training_config'] 
        dataset_config = checkpoint['dataset_config'] 
        
        nova_model = NOVAModel(model_config)
        nova_model.model.load_state_dict(state_dict)
        nova_model.training_config = training_config
        nova_model.dataset_config = dataset_config
        
        return nova_model
        