import logging
import os
import sys
import torch
import numpy as np

from typing import Dict

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.lib.models.trainers.trainer_base import TrainerBase
from src.common.configs.trainer_config import TrainerConfig
from src.common.lib.models.NOVA_model import NOVAModel

class TrainerClassification(TrainerBase):
    def __init__(self, trainer_config:TrainerConfig, nova_model:NOVAModel):
        """Get an instance

        Args:
            conf (TrainerConfig): The trainer configuration
            nova_model (NOVAModel): The NOVA model to train
        """
        super().__init__(trainer_config, nova_model)
        
        self.loss_CE = torch.nn.CrossEntropyLoss().cuda()
        
    def loss(self, outputs:torch.Tensor, targets:torch.Tensor)->float:
        """Calculating the CrossEntropy loss between the given model outputs and the true labels (targets)

        Returns:
            float: The loss value
        """
        return self.loss_CE(outputs, targets)
    
    def forward(self, X: torch.Tensor, y: torch.Tensor=None, paths: np.ndarray[str]=None) -> Dict:
        """Applying the forward pass (running the model on the given data)

        Args:
            X (torch.Tensor): The data to feed into the model
            y (torch.Tensor, optional): The ids for the labels. Defaults to None.
            paths (np.ndarray[str], optional): The paths to the files. Defaults to None.

        Returns:
            Dict: {outputs: The model outputs, targets: The true labels}
        """
        
        logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
        outputs = self.nova_model.model(X)      
        # CrossEntropy loss must get type long for labels
        y = y.long()    
            
        return {'outputs': outputs, 'targets': y}
    
    
        
    