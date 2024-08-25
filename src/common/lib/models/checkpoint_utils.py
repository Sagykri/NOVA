
import logging
from typing import Dict, Self
import numpy as np
import torch
import os

class CheckpointInfo():
    """Handle the checkpoint
    """
    def __init__(self,
                 model_dict:Dict=None,
                 optimizier_dict: Dict=None,
                 epoch:int=0,
                 training_config:Dict=None,
                 dataset_config:Dict=None,
                 model_config:Dict=None,
                 scaler_dict:Dict=None,
                 loss_val_avg:float=np.inf,
                 best_loss_val_avg:float=np.inf,
                 early_stopping_counter:int=0,
                 description:str=''):
        
        self.model_dict: Dict = model_dict
        self.optimizier_dict: Dict = optimizier_dict
        self.epoch:int = epoch
        self.training_config: Dict = training_config
        self.dataset_config: Dict = dataset_config
        self.model_config: Dict = model_config
        self.scaler_dict: Dict = scaler_dict
        self.loss_val_avg:float = loss_val_avg
        self.best_loss_val_avg:float = best_loss_val_avg
        self.early_stopping_counter:int = early_stopping_counter
        self.description:str = description
    
    @staticmethod
    def load_from_checkpoint_filepath(checkpoint_path: str)->Self:
        """Get a CheckpointInfo instance from a path to the checkpoint file

        Args:
            checkpoint_path (str): The path to the checkpoint file

        Returns:
            CheckpointInfo: An instance of CheckpointInfo
        """
        checkpoint = load_checkpoint_from_file(checkpoint_path)
        return CheckpointInfo.load_from_checkpoint_object(checkpoint)
        
    @staticmethod
    def load_from_checkpoint_object(checkpoint)->Self:
        """Get a new CheckpointInfo instance from a checkpoint object loaded via torch.load

        Args:
            checkpoint (checkpoint): A checkpoint loaded via torch.load

        Returns:
            CheckpointInfo: An instance of CheckpointInfo
        """
        new_instance = CheckpointInfo()
        new_instance.__dict__.update(checkpoint)
        
        return new_instance
    
    def save(self, output_filepath:str)->None:
        """Save checkpoint to file

        Args:
            output_filepath (str): The path to save the file to
        """
        save_checkpoint(self, output_filepath)
        

def load_checkpoint_from_file(ckp_path:str):
    """Load checkpoint from file

    Args:
        ckp_path (str): The path to the checkpoint

    Returns:
        Any: The checkpoint
    """
    assert os.path.exists(ckp_path), f"{ckp_path} doesn't exist"
    assert os.path.isfile(ckp_path), f"{ckp_path} isn't a file"
    
    checkpoint = torch.load(ckp_path, map_location='cuda' if torch.cuda.is_available() else "cpu")
    return checkpoint

def save_checkpoint(checkpoint_info: CheckpointInfo, output_filepath:str)->None:
    """Save checkpoints info to file

    Args:
        checkpoint_info (CheckpointInfo): The info to be saved
        output_filepath (str): The path to save the info into

    """
    outputdir = os.path.dirname(output_filepath)
    if not os.path.exists(outputdir):
        logging.info(f"{outputdir} doesn't exist. Creating dir")
        os.makedirs(outputdir)
        
    logging.info(f"Saving checkpoint to file {output_filepath}")
    torch.save(
        checkpoint_info.__dict__, output_filepath
    )