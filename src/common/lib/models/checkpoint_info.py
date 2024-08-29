
import logging
from typing import Dict
import numpy as np
import torch
import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.model_config import ModelConfig
from src.common.configs.trainer_config import TrainerConfig

class CheckpointInfo():
    """Handle the checkpoint
    """
    def __init__(self,
                 model_dict:Dict=None,
                 optimizer_dict: Dict=None,
                 epoch:int=0,
                 trainer_config:TrainerConfig=None,
                 dataset_config:DatasetConfig=None,
                 model_config:ModelConfig=None,
                 scaler_dict:Dict=None,
                 avg_val_loss:float=np.inf,
                 best_avg_val_loss:float=np.inf,
                 early_stopping_counter:int=0,
                 description:str=''):
        """Get an instance

         Args:
             model_dict (Dict, optional): The model state_dict. Defaults to None.
             optimizer_dict (Dict, optional): The optimizier state_dict. Defaults to None.
             epoch (int, optional): The epoch number. Defaults to 0.
             trainer_config (TrainerConfig, optional): The trainer config object. Defaults to None.
             dataset_config (DatasetConfig, optional): The dataset config object. Defaults to None.
             model_config (ModelConfig, optional): The model config object. Defaults to None.
             scaler_dict (Dict, optional): The scaler state_dict. Defaults to None.
             avg_val_loss (float, optional): The average loss on the validation set. Defaults to np.inf.
             best_avg_val_loss (float, optional): The best average loss on the validation set. Defaults to np.inf.
             early_stopping_counter (int, optional): The counter value for the early stopping mechanism. Defaults to 0.
             description (str, optional): A description for this checkpoint. Defaults to ''.
         """
        
        self.model_dict: Dict = model_dict
        self.optimizer_dict: Dict = optimizer_dict
        self.epoch:int = epoch
        self.trainer_config: TrainerConfig = trainer_config
        self.dataset_config: DatasetConfig = dataset_config
        self.model_config: ModelConfig = model_config
        
        self.trainer_config_dict = self.trainer_config.__dict__ if self.trainer_config is not None else {}
        self.dataset_config_dict = self.dataset_config.__dict__ if self.dataset_config is not None else {}
        self.model_config_dict = self.model_config.__dict__ if self.model_config is not None else {}
        
        self.scaler_dict: Dict = scaler_dict
        self.avg_val_loss:float = avg_val_loss
        self.best_avg_val_loss:float = best_avg_val_loss
        self.early_stopping_counter:int = early_stopping_counter
        self.description:str = description
        
        self.rng_state = torch.get_rng_state().tolist()
        self.cuda_rng_state = [l.tolist() for l in torch.cuda.get_rng_state_all()]
    
    @staticmethod
    def load_from_checkpoint_filepath(checkpoint_path: str):
        """Get a CheckpointInfo instance from a path to the checkpoint file

        Args:
            checkpoint_path (str): The path to the checkpoint file

        Returns:
            CheckpointInfo: An instance of CheckpointInfo
        """
        assert os.path.exists(checkpoint_path), f"{checkpoint_path} doesn't exist"
        assert os.path.isfile(checkpoint_path), f"{checkpoint_path} isn't a file"
        
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else "cpu")
        return CheckpointInfo.load_from_checkpoint_object(checkpoint)
        
    @staticmethod
    def load_from_checkpoint_object(checkpoint):
        """Get a new CheckpointInfo instance from a checkpoint object loaded via torch.load

        Args:
            checkpoint (checkpoint): A checkpoint loaded via torch.load

        Returns:
            CheckpointInfo: An instance of CheckpointInfo
        """
        new_instance = CheckpointInfo()
        new_instance.__dict__.update(checkpoint)
        
        new_instance.rng_state = torch.tensor(new_instance.rng_state).byte()
        new_instance.cuda_rng_state = [torch.tensor(l).byte() for l in new_instance.cuda_rng_state]
        
        return new_instance
    
    def save(self, output_filepath:str)->None:
        """Save checkpoint to file

        Args:
            output_filepath (str): The path to save the file to
        """
        outputdir = os.path.dirname(output_filepath)
        if not os.path.exists(outputdir):
            logging.info(f"{outputdir} doesn't exist. Creating dir")
            os.makedirs(outputdir)
            
        logging.info(f"Saving checkpoint to file {output_filepath}")
        torch.save(
            self.__dict__, output_filepath
        )
         