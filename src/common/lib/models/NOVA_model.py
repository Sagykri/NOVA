import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from collections import OrderedDict

sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.lib.utils import get_if_exists
from src.common.lib.models.checkpoint_info import CheckpointInfo
from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.model_config import ModelConfig
from src.common.lib.models import vision_transformer
from src.common.configs.base_config import BaseConfig

class NOVAModel():

    def __init__(self, model_config:ModelConfig):
        """Get an instance

        Args:
            model_config (ModelConfig): The model configuration
        """
        self.__set_params(model_config)
        self.model = self.__get_vit()    
        
    @staticmethod
    def load_from_checkpoint(ckp_path: str):
        """Get model from checkpoint

        Args:
            ckp_path (str): path to checkpoint

        Returns:
            model (NOVAModel): The NOVA model
        """
        
        checkpoint:CheckpointInfo = CheckpointInfo.load_from_checkpoint_filepath(ckp_path)
        
        nova_model = NOVAModel(checkpoint.model_config)
        nova_model.model_config = BaseConfig.create_a_copy(checkpoint.model_config)
        nova_model.trainer_config = BaseConfig.create_a_copy(checkpoint.trainer_config)
        nova_model.dataset_config = BaseConfig.create_a_copy(checkpoint.dataset_config)
        nova_model.model.load_state_dict(checkpoint.model_dict)
        
        return nova_model    
    
    def generate_embeddings(self, dataset_config: DatasetConfig)->np.ndarray:
        """Generate embeddings for the given data using the model

        Args:
            dataset_config (DatasetConfig): The configuration for indicating what data to load

        Returns:
            np.ndarray: The embeddings
        """
        from common.lib import embeddings_utils
        
        return embeddings_utils.generate_embeddings(self, dataset_config)
    
    def infer(self, data_loader: DataLoader)->Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]:
        """Run inference on the data_loader data

        Args:
            data_loader (DataLoader): The dataloader to run inference on

        Returns:
            Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]: (all the embeddings, all the labels)
        """
        all_embeddings:List[torch.Tensor] = []
        all_labels:np.ndarray[str] = np.array([])
        
        # Move model to cuda
        self.model = self.model.cuda()
        
        # Set model to eval mode
        self.model.eval()
        
        with torch.no_grad():
            for it, res in enumerate(data_loader):
                logging.info(f"[Inference] Batch number: {it}/{len(data_loader)}")
                
                # extract the X and y from the batch
                images = res['image'].to(torch.float).cuda()
                labels_ind = res['label'].numpy()
                
                # convert from indexes to the labels
                labels = data_loader.dataset.id2label(labels_ind)
                # run the model to get the embeddings
                embeddings = self.model(images).cpu()
                
                all_embeddings.append(embeddings)
                all_labels = np.append(all_labels, labels)
        
        all_embeddings:np.ndarray[torch.Tensor] = np.vstack(all_embeddings)
        
        return all_embeddings, all_labels
    
    def is_equal_architecture(self, other_state_dict: Dict)->bool:
        """Check if the given state_dict is equal to self state_dict

        Args:
            other_state_dict (Dict): The other state dict

        Returns:
            bool: Is equal?
        """
        self_architecture = OrderedDict({key: value.shape for key, value in self.model.state_dict().items()})
        other_architecture = OrderedDict({key: value.shape for key, value in other_state_dict.items()})
        
        # First, check if both shape_dicts have the same keys
        if self_architecture.keys() != other_architecture.keys():
            return False
        
        # Now compare the shapes for each key
        for key in self_architecture.keys():
            if self_architecture[key] != other_architecture[key]:
                return False
        
        return True
        
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
        self.num_classes = self.model_config.NUM_CLASSES

    def __get_vit(self)->vision_transformer.VisionTransformer:
        """Init a vit model

        Returns:
            vision_transformer.VisionTransformer: An initialized vit model
        """
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