
import logging
from typing import Dict
import numpy as np
import torch
import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.datasets.dataset_config import DatasetConfig
from src.models.architectures.model_config import ModelConfig
from src.models.trainers.trainer_config import TrainerConfig

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
                 trainset_paths:np.ndarray[str]=[],
                 trainset_labels:np.ndarray[str]=[],
                 valset_paths:np.ndarray[str]=[],
                 valset_labels:np.ndarray[str]=[],
                 testset_paths:np.ndarray[str]=[],
                 testset_labels:np.ndarray[str]=[],
                 description:str=''):
        """Get an instance

         Args:
             model_dict (Dict): The model state_dict. 
             optimizer_dict (Dict): The optimizier state_dict. 
             epoch (int): The epoch number. 
             trainer_config (TrainerConfig): The trainer config object.
             dataset_config (DatasetConfig): The dataset config object. 
             model_config (ModelConfig): The model config object. 
             scaler_dict (Dict): The scaler state_dict. 
             avg_val_loss (float): The average loss on the validation set. 
             best_avg_val_loss (float): The best average loss on the validation set. 
             early_stopping_counter (int): The counter value for the early stopping mechanism. 
             trainset_paths (np.ndarray[str]): Paths to the trainset files. 
             trainset_labels (np.ndarray[str]): Labels of the trainset files. 
             valset_paths (np.ndarray[str]): Paths to the valset files.
             valset_labels (np.ndarray[str]): Labels to the valset files.
             testset_paths (np.ndarray[str]): Paths to the testset files.
             testset_labels (np.ndarray[str]): Labels to the testset files.
             description (str, optional): A description for this checkpoint. Defaults to ''.
         """
        
        self.model_dict: Dict = model_dict
        self.optimizer_dict: Dict = optimizer_dict
        self.epoch:int = epoch
        
        self.trainer_config_dict = trainer_config.__dict__ if trainer_config is not None else {}
        self.dataset_config_dict = dataset_config.__dict__ if dataset_config is not None else {}
        self.model_config_dict = model_config.__dict__ if model_config is not None else {}
        
        self.scaler_dict: Dict = scaler_dict
        self.avg_val_loss:float = avg_val_loss
        self.best_avg_val_loss:float = best_avg_val_loss
        self.early_stopping_counter:int = early_stopping_counter
        self.description:str = description
        
        self.trainset_paths:np.ndarray[str] = trainset_paths
        self.trainset_labels:np.ndarray[str] = trainset_labels
        self.valset_paths:np.ndarray[str] = valset_paths
        self.valset_labels:np.ndarray[str] = valset_labels
        self.testset_paths:np.ndarray[str] = testset_paths
        self.testset_labels:np.ndarray[str] = testset_labels
        
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
    
        new_instance = CheckpointInfo()
        new_instance.__dict__.update(checkpoint)
        
        new_instance.rng_state = torch.tensor(new_instance.rng_state).byte()
        new_instance.cuda_rng_state = [torch.tensor(l).byte() for l in new_instance.cuda_rng_state]
        
        # Convert the configuration dict to instances
        new_instance.trainer_config = TrainerConfig.from_dict(new_instance.trainer_config_dict)
        new_instance.dataset_config = DatasetConfig.from_dict(new_instance.dataset_config_dict)
        new_instance.model_config   = ModelConfig.from_dict(new_instance.model_config_dict)
        
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
         