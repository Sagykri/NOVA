import logging
import os
import sys
import torch

from typing import Dict, Self

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.lib.models.trainers.trainer_base import TrainerBase
from src.common.configs.trainer_config import TrainerConfig

class TrainerClassification(TrainerBase):
    def __init__(self, conf:TrainerConfig)->Self:
        super().__init__(conf)
        
        self.loss_CE = torch.nn.CrossEntropyLoss().cuda()
        
    def loss(self, outputs:torch.Tensor, targets:torch.Tensor)->float:
        """Calculating the CrossEntropy loss between the given model outputs and the true labels (targets)

        Returns:
            float: The loss value
        """
        return self.loss_CE(outputs, targets)
    
    def forward(self, model: torch.nn.Module, X:torch.Tensor) -> Dict:
        """Applying the forward pass (running the model on the given data)

        Args:
            model (torch.nn.Module): The model
            X (torch.Tensor): The data to feed into the model

        Returns:
            Dict: {outputs: The model outputs, targets: The true labels}
        """
        with torch.cuda.amp.autocast():
            images, targets = X['image'].to(torch.float).cuda(), X['label'].cuda()
            
            logging.info(f"images shape: {images.shape}, targets shape: {targets.shape}")
            outputs = model(images)          
            
        return {'outputs': outputs, 'targets': targets}
    
    
        
    