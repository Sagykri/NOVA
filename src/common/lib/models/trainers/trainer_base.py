from abc import abstractmethod
import datetime
import logging
import math
import os
import sys
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from typing import Any, Dict, Self

from torch.utils.tensorboard import SummaryWriter

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.lib.utils import get_if_exists
from src.common.lib.models import model_utils 
from src.common.lib.models.NOVA_model import NOVAModel

class Trainer():
    # TODO: Create the TrainingConfig class
    def __init__(self, conf:TrainingConfig)->Self:
        self.__set_params(conf)     
    
    def __set_params(self, training_config:TrainingConfig)->None:
        """Extracting params from the configuration

        Args:
            training_config (TrainingConfig): The training configuration
        """
        self.training_config = training_config
        self.epochs:int = self.training_config['EPOCHS'] 
        self.lr:float = self.training_config['LR']
        self.min_lr:float = self.training_config["MIN_LR"]
        self.warmup_epochs:int = self.training_config["WARMUP_EPOCHS"]  
        self.weight_decay:float = self.training_config['WEIGHT_DECAY']
        self.weight_decay_end:float = self.training_config['WEIGHT_DECAY_END']
        self.batch_size_per_gpu:int = self.training_config['BATCH_SIZE_PER_GPU']
        self.num_workers:int = get_if_exists(self.training_config, 'NUM_WORKERS', 6)
        self.early_stopping_patience:int = get_if_exists(self.training_config, 'EARLY_STOPPING_PATIENCE', 10)
        self.output_dir:str = self.training_config['OUTPUT_DIR']
        self.description:str = get_if_exists(self.training_config, 'DESCRIPTION', str(type(self)))
        
        self.__set_output_dirs()
        
    def __set_output_dirs(self)->None:
        """Set the path for the output directories (logs, checkpoints and tensorboard plots)
        """
        
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.checkpoints_root_dir = os.path.join(self.output_dir, "checkpoints")
        self.tensorboard_root_dir = os.path.join(self.output_dir, "tensorboard")
        
        ##
        
        now_formatted = datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")
        jobid = os.getenv('LSB_JOBID')
        jobname = os.getenv('LSB_JOBNAME')
        foldername_postfix = f"_{now_formatted}_{jobid}_{jobname}"
        
        self.checkpoints_dir = os.path.join(self.checkpoints_root_dir, f"checkpoints{foldername_postfix}")
        self.tensorboard_dir = os.path.join(self.tensorboard_root_dir, foldername_postfix)
    
    @abstractmethod
    def loss(**kwargs)->float:
        """Calculating the loss

        Returns:
            float: The loss value
        """
        pass
    
    @abstractmethod
    def forward(x, **kwargs) -> Any:
        """Applying the forward pass
        """
        pass
    
    def train(self, nova_model:NOVAModel, data_loader: DataLoader)->None:
        self.__init_params_for_training()
        self.__training_loop(nova_model, data_loader)
        
    def __init_params_for_training(self, nova_model: NOVAModel, data_loader: DataLoader)->None:
        """Init the needed objects for the training (optimizer, schedulers, etc.)

        Args:
            nova_model (NOVAModel): The NOVA model
            data_loader (torch.utils.data.DataLoader): The dataloader
        """
        params_groups = model_utils.get_params_groups(nova_model.model)
        self.optimizer = torch.optim.AdamW(params_groups)
        
        self.scaler = torch.cuda.amp.GradScaler()

        # learning rate schedule
        self.lr_schedule = model_utils.cosine_scheduler(
            self.lr * (self.batch_size_per_gpu) / 256.0,  # linear scaling rule
            self.min_lr,
            self.epochs,
            len(data_loader),
            warmup_epochs=self.warmup_epochs,
        )
        
        # weight decay
        self.wd_schedule = model_utils.cosine_scheduler(
            self.weight_decay,
            self.weight_decay_end,
            self.epochs,
            len(data_loader),
        )
        
        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_dir)

                
    def __training_loop(self, nova_model:NOVAModel, data_loader_train: DataLoader, data_loader_val: DataLoader)->None:
        # TODO: Add support for recovering from a fall (restart_from_checkpoint)
        
        self.__init_params_for_training(nova_model, data_loader_train)
        
        early_stopping_info = model_utils.EarlyStoppingInfo()
        
        for epoch in tqdm(range(start_epoch, self.epochs)):
            self.optimizer.zero_grad()
            
            loss_val_avg = self.__run_one_epoch(nova_model.model, data_loader_train, data_loader_val, epoch)
            
            # Handle saving the checkpoint
            checkpoint_info = model_utils.CheckpointInfo(model_dict = nova_model.model.state_dict(),
                                                        optimizer_dict=self.optimizer.state_dict(),
                                                        epoch=epoch,
                                                        training_config=self.training_config.__dict__,
                                                        dataset_config=data_loader_train.dataset.conf.__dict__,
                                                        model_config=nova_model.model_config.__dict__,
                                                        scaler_dict=self.scaler.state_dict(),
                                                        loss_val_avg=loss_val_avg,
                                                        early_stopping_counter=early_stopping_info.counter,
                                                        description=self.description)

            self.__handle_save_checkpoint(loss_val_avg, checkpoint_info)
            
            # Handle early stopping
            is_early_stopping_reached = self.__is_early_stopping_reached(loss_val_avg, early_stopping_info)
            if is_early_stopping_reached:
                break
            
    def __handle_save_checkpoint(self, loss_val_avg:float, checkpoint_info: model_utils.CheckpointInfo)->None:
        """Save running checkpoint and best checkpoint if we reached a new best 

        Args:
            loss_val_avg (float): The current average validation loss
            checkpoint_info (CheckpointInfo): The info to saved into the checkpoint file
        """
        
        savepath_chkp_last = os.path.join(self.checkpoints_dir, "checkpoint_last")
        savepath_chkp_best = os.path.join(self.checkpoints_dir, "checkpoint_best")
        
        # Save running checkpoint
        model_utils.save_checkpoint(checkpoint_info, savepath_chkp_last)
        
        # Save best checkpoint
        if not os.path.exists(savepath_chkp_best):
            # First best checkpoint
            model_utils.save_checkpoint(checkpoint_info, savepath_chkp_best)
        else:
            best_checkpoint = torch.load(savepath_chkp_best, map_location="cpu")
            best_checkpoint_val_loss_avg = best_checkpoint['val_loss_avg']
            if loss_val_avg < best_checkpoint_val_loss_avg:
                # We have a new best checkpoint!
                model_utils.save_checkpoint(checkpoint_info, savepath_chkp_best)
                
    def __is_early_stopping_reached(self, current_loss_value:float, early_stopping_info:model_utils.EarlyStoppingInfo)->bool:
        """Checks if we exhaused enough epochs without any improvement in order to early stop

        Args:
            current_loss_value (float): The current loss value to check against
            early_stopping_info (EarlyStoppingInfo): Holds info for handeling the early stopping 
        """
        
        if current_loss_value < early_stopping_info.best_loss:
            # Improved!
            early_stopping_info.best_loss = current_loss_value
            early_stopping_info.counter = self.early_stopping_patience  
            
            return False
            
        # No improvement
        early_stopping_info.counter -= 1
        logging.warn(f"No improvement. Early stopping counter is now {early_stopping_info.counter}")
        if early_stopping_info.counter <= 0:
            logging.warn(f"Stopping due to early stopping")
            return True
        
        return False
        
    def __run_one_epoch(self, data_loader_train: DataLoader, data_loader_val: DataLoader, current_epoch: int)->float:
        logging.info(f"Epoch: [{current_epoch}/{self.epochs}]")
        
        loss_train_avg = self.__run_one_epoch_train(current_epoch, data_loader_train)
        loss_val_avg = self.__run_one_epoch_eval(current_epoch, data_loader_val)
        
        self.tensorboard_writer.add_scalars('Loss', {'Training': loss_train_avg, 'Validation': loss_val_avg}, current_epoch)
        
        return loss_val_avg

    def __run_one_epoch_train(self, model:torch.nn.Module, current_epoch:int, data_loader:DataLoader)->float:
        """Run one epoch on the given dataloader in train mode

        Args:
            model (torch.nn.Module): The model
            current_epoch (int): The current epoch
            data_loader (DataLoader): The dataloader to get the data from

        Returns:
            float: The average loss on the training set
        """
        
        loss_train_avg = 0
        model.train()
        
        logging.info(f"--------------------------------------- TRAINING (epoch={current_epoch}) ---------------------------------------")
        for it, res in enumerate(data_loader):
            logging.info(f"[Training epoch={current_epoch}] batch number: {it}/{len(data_loader)}")
            
            self.__handle_scehdulers(current_epoch, data_loader)

            # calc loss value
            loss = self.forward(res)
            current_loss_value = loss.item()
            loss_train_avg += current_loss_value

            if not math.isfinite(current_loss_value):
                logging.info("Loss is {}, stopping training".format(current_loss_value))
                raise Exception(f"Loss is {current_loss_value}, stopping training")
                
            logging.info(f"***** [epoch {current_epoch}] train loss: {current_loss_value} *****")
            
            logging.info("Calculating grads")
            self.__calculate_gradients(model, loss)
                        
            logging.info("Updating model")
            self.__update_model_weights()
            
            # Write to tensorboard
            self.tensorboard_writer.add_scalar('1. Loss/train', current_loss_value, current_epoch * len(data_loader) + it)
            self.tensorboard_writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], current_epoch * len(data_loader) + it)
            self.tensorboard_writer.add_scalar('Weight Decay', self.optimizer.param_groups[0]['weight_decay'], current_epoch * len(data_loader) + it)
            
            self.optimizer.zero_grad()  # Reset gradients after each update
            
        # Take the average loss on the training set
        loss_train_avg /= len(data_loader)
        
        logging.info(f"***** [epoch {current_epoch}] Averaged train loss: {loss_train_avg} *****")
        self.tensorboard_writer.add_scalar('1. Loss/train_avg', loss_train_avg, current_epoch)
        
        return loss_train_avg
    
    def __handle_scehdulers(self, current_epoch:int, data_loader: DataLoader)->None:
        """Setting the learning rate and weight_decay based on the scheduler

        Args:
            current_epoch (int): The current epoch
            data_loader (DataLoader): The dataloader
        """
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * current_epoch + it  # global training iteration
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[it]
    
    def __calculate_gradients(self, model:torch.nn.Module, loss_value: torch.Tensor):
        """Calculate gradients based on the loss

        Args:
            model (torch.nn.Module): The model
            loss_value (torch.Tensor): The loss
        """
        self.scaler.scale(loss_value).backward()
        self.scaler.unscale_(self.optimizer)  # Unscale gradients    
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)  # Clip gradients
        
    def __update_model_weights(self):
        """Update the weights of the model
        """
        self.scaler.step(self.optimizer)  # Update weights
        self.scaler.update()
                    
    def __run_one_epoch_eval(self, model:torch.nn.Module, current_epoch:int, data_loader:DataLoader)->float:
        """Run one epoch on the given dataloader in eval mode

        Args:
            model (torch.nn.Module): The model
            current_epoch (int): The current epoch
            data_loader (DataLoader): The dataloader to get the data from

        Returns:
            float: The average loss on the validation set
        """
        logging.info(f"--------------------------------------- EVAL (epoch={current_epoch}) ---------------------------------------")
        model.eval()
        
        with torch.no_grad():
            loss_val_avg = 0
            for it, res in enumerate(data_loader):
                logging.info(f"[Validation epoch={current_epoch}] batch number: {it}/{len(data_loader)}")
                loss = self.forward(res)
                    
                logging.info(f"***** [epoch {current_epoch}] val loss: {loss.item()} *****")
                
                self.tensorboard_writer.add_scalar('1. Loss/val', loss.item(), current_epoch * len(data_loader) + it)
                loss_val_avg += loss.item()
                
            # Take the average loss on the validation set
            loss_val_avg /= len(data_loader)
            logging.info(f"***** [epoch {current_epoch}] Averaged val loss: {loss_val_avg} *****")
            self.tensorboard_writer.add_scalar('1. Loss/val_avg', loss_val_avg, current_epoch)
            
        return loss_val_avg
    
    
