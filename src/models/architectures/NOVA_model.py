import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from collections import OrderedDict
import torch.nn.functional as F

sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.models.utils.checkpoint_info import CheckpointInfo
from src.datasets.dataset_config import DatasetConfig
from src.models.architectures.model_config import ModelConfig
from src.models.architectures import vision_transformer

class NOVAModel():

    def __init__(self, model_config:ModelConfig):
        """Get an instance

        Args:
            model_config (ModelConfig): The model configuration
        """
        self.model_config = model_config
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
        nova_model.model.load_state_dict(checkpoint.model_dict)
        nova_model.trainset_paths   = checkpoint.trainset_paths
        nova_model.trainset_labels  = checkpoint.trainset_labels
        nova_model.valset_paths     = checkpoint.valset_paths
        nova_model.valset_labels    = checkpoint.valset_labels
        nova_model.testset_paths    = checkpoint.testset_paths
        nova_model.testset_labels   = checkpoint.testset_labels
        
        return nova_model    
    
    def generate_embeddings(self, dataset_config: DatasetConfig)->np.ndarray:
        """Generate embeddings for the given data using the model

        Args:
            dataset_config (DatasetConfig): The configuration for indicating what data to load

        Returns:
            np.ndarray: The embeddings
        """
        from src.embeddings import embeddings_utils
        
        return embeddings_utils.generate_embeddings(self, dataset_config)
    
    def infer(self, data_loader: DataLoader, return_hidden_outputs=True, normalize_outputs=True)->Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]:
        """Run inference on the data_loader data

        Args:
            data_loader (DataLoader): The dataloader to run inference on
            return_hidden_outputs (bool, optional): Whether to return the hidden outputs (i.e. before the head). Defaults to True.
            normalize_outputs (bool, optional): Whether to normalize the outputs. Defaults to True.

        Returns:
            Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]: (all the outputs, all the labels)
        """
        all_outputs:List[torch.Tensor] = []
        all_labels:np.ndarray[str] = np.array([])
        all_paths:np.ndarray[str] = np.array([])
        
        # Move model to cuda
        self.model = self.model.cuda()
        
        # Set model to eval mode
        self.model.eval()
        
        with torch.no_grad():
            for it, res in enumerate(data_loader):
                logging.info(f"[Inference] Batch number: {it}/{len(data_loader)}")
                X, y, path = res
                X = X.cuda()
                
                # convert from indexes to the labels
                labels = data_loader.dataset.id2label(y)
                # run the model to get the embeddings
                if return_hidden_outputs:
                    _, outputs = self.model(X, return_hidden=return_hidden_outputs) # head outputs, hidden_outputs (i.e. before head)
                else: 
                    outputs = self.model(X, return_hidden=return_hidden_outputs)

                if normalize_outputs:
                    # Normalize the outputs
                    outputs = F.normalize(outputs, dim=-1)

                outputs = outputs.cpu()
                
                all_outputs.append(outputs)
                all_labels = np.append(all_labels, labels)
                all_paths = np.append(all_paths, path)
        
        all_outputs:np.ndarray[torch.Tensor] = np.vstack(all_outputs)
        
        return all_outputs, all_labels, all_paths
    
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

    def __get_vit(self)->vision_transformer.VisionTransformer:
        """Init a vit model

        Returns:
            vision_transformer.VisionTransformer: An initialized vit model
        """
        vit_version = self.model_config.VIT_VERSION
        
        if vit_version == 'base':
            create_vit = vision_transformer.vit_base
        elif vit_version == 'small':
            create_vit = vision_transformer.vit_small
        elif vit_version == 'tiny':
            create_vit = vision_transformer.vit_tiny
        else:
            raise Exception(f"Invalid 'vit_version' detected: {vit_version}. Must be 'base', 'small' or 'tiny'")
        
        vit = create_vit(
                img_size=[self.model_config.IMAGE_SIZE],
                patch_size=self.model_config.PATCH_SIZE,
                in_chans=self.model_config.NUM_CHANNELS,
                num_classes=self.model_config.OUTPUT_DIM
        )
        
        return vit