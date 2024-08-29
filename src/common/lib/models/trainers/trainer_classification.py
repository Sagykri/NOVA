import logging
import os
import sys
import torch

from typing import Dict

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.lib.models.trainers.trainer_base import TrainerBase
from src.common.configs.trainer_config import TrainerConfig
from src.common.lib.models.NOVA_model import NOVAModel

class TrainerClassification(TrainerBase):
    def __init__(self, conf:TrainerConfig, nova_model:NOVAModel):
        """Get an instance

        Args:
            conf (TrainerConfig): The trainer configuration
            nova_model (NOVAModel): The NOVA model to train
        """
        super().__init__(conf, nova_model)
        
        self.loss_CE = torch.nn.CrossEntropyLoss().cuda()
        
    def loss(self, outputs:torch.Tensor, targets:torch.Tensor)->float:
        """Calculating the CrossEntropy loss between the given model outputs and the true labels (targets)

        Returns:
            float: The loss value
        """
        return self.loss_CE(outputs, targets)
    
    def forward(self, X:torch.Tensor) -> Dict:
        """Applying the forward pass (running the model on the given data)

        Args:
            X (torch.Tensor): The data to feed into the model

        Returns:
            Dict: {outputs: The model outputs, targets: The true labels}
        """
        images, targets = X['image'].to(torch.float).cuda(), X['label'].cuda()
        
        logging.info(f"images shape: {images.shape}, targets shape: {targets.shape}")
        outputs = self.nova_model.model(images)          
            
        return {'outputs': outputs, 'targets': targets}
    
    
        
    