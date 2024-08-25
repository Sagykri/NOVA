from abc import abstractmethod
import logging
import math
import os
import sys
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from typing import Dict, Self

from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.lib.utils import fix_random_seeds, get_if_exists
from src.common.lib.models.NOVA_model import NOVAModel
from src.common.lib.models.checkpoint_utils import CheckpointInfo
from src.common.lib.models.trainers.utils import trainer_utils 
from src.common.lib.models.trainers.utils.trainer_utils import EarlyStoppingInfo
from src.common.configs.trainer_config import TrainerConfig

class TrainerBase():
    def __init__(self, conf:TrainerConfig)->Self:
        self.__set_params(conf)     
        
    @abstractmethod
    def loss(self, **kwargs)->float:
        """Calculating the loss

        Returns:
            float: The loss value
        """
        pass
    
    @abstractmethod
    def forward(self, model: torch.nn.Module, X: torch.Tensor) -> Dict:
        """Applying the forward pass (running the model on the given data)

        Args:
            model (torch.nn.Module): The model
            X (torch.Tensor): The data to feed into the model

        Returns:
            Dict: Results that will enter the 'loss' function
        """
        pass
    
    def train(self, nova_model:NOVAModel, data_loader_train: DataLoader, data_loader_val: DataLoader)->None:
        self.__init_params_for_training(nova_model, data_loader_train)
        
        self.__try_restart_from_last_checkpoint(nova_model, data_loader_train)
        
        # Save training config into the model
        nova_model.training_config = self.training_config
                
        self.__training_loop(nova_model, data_loader_train, data_loader_val)
        
    def __set_params(self, training_config:TrainerConfig)->None:
        """Extracting params from the configuration

        Args:
            training_config (TrainingConfig): The training configuration
        """
        self.__extract_params_from_config(training_config)
        
        self.__set_output_dirs()
        
        self.checkpoint_last_filename = 'checkpoint_last'
        self.checkpoint_best_filename = 'checkpoint_best'
        
        self.early_stopping_info = EarlyStoppingInfo(self.early_stopping_patience)
        self.starting_epoch = 0
        self.optimizer:torch.optim.Optimizer = None
        self.scaler:torch.cuda.amp.GradScaler = torch.cuda.amp.GradScaler()
        self.lr_schedule = None
        self.wd_schedule = None
        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_dir)
        
        # Make the training reproducable
        fix_random_seeds(self.training_config.SEED)
        cudnn.benchmark = False
        
    def __extract_params_from_config(self, training_config: TrainerConfig)->None:
        self.training_config = training_config
        self.max_epochs:int = self.training_config['MAX_EPOCHS'] 
        self.lr:float = self.training_config['LR']
        self.min_lr:float = self.training_config["MIN_LR"]
        self.warmup_epochs:int = self.training_config["WARMUP_EPOCHS"]
        self.weight_decay:float = self.training_config['WEIGHT_DECAY']
        self.weight_decay_end:float = self.training_config['WEIGHT_DECAY_END']
        self.batch_size:int = self.training_config['BATCH_SIZE']
        self.num_workers:int = get_if_exists(self.training_config, 'NUM_WORKERS', 6)
        self.early_stopping_patience:int = get_if_exists(self.training_config, 'EARLY_STOPPING_PATIENCE', 10)
        self.outputs_folder:str = self.training_config['OUTPUTS_FOLDER']
        self.description:str = get_if_exists(self.training_config, 'DESCRIPTION', str(type(self)))
        
    def __set_output_dirs(self)->None:
        """Set the path for the output directories (logs, checkpoints and tensorboard plots)
        """
        
        assert self.outputs_folder is not None, "outputs folder can't be None"
        
        self.logs_dir = os.path.join(self.outputs_folder, "logs")
        self.tensorboard_dir = os.path.join(self.outputs_folder, "tensorboard")
        self.checkpoints_dir = os.path.join(self.outputs_folder, "checkpoints")
        
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
    def __try_init_from_checkpoint(self, checkpoint:CheckpointInfo)->None:
        """Init trainer's params from the checkpoint file

        Args:
            checkpoint (CheckpointInfo): The checkpoint to init from
        """
        self.starting_epoch = checkpoint.epoch + 1
        self.optimizer.load_state_dict(checkpoint.optimizier_dict)
        self.description = checkpoint.description
        self.early_stopping_info = EarlyStoppingInfo(checkpoint.early_stopping_counter)
        
        logging.info(f"Checkpoint has been loaded successfully. Starting epoch was set to {self.starting_epoch}")
        
    def __init_params_for_training(self, nova_model: NOVAModel, data_loader: DataLoader)->None:
        """Init the needed objects for the training (optimizer, schedulers, etc.)

        Args:
            nova_model (NOVAModel): The NOVA model
            data_loader (torch.utils.data.DataLoader): The dataloader
        """
        params_groups = trainer_utils.get_params_groups(nova_model.model)
        self.optimizer = torch.optim.AdamW(params_groups)

        # learning rate schedule
        self.lr_schedule = trainer_utils.cosine_scheduler(
            self.lr * (self.batch_size) / 256.0,  # linear scaling rule
            self.min_lr,
            self.max_epochs,
            len(data_loader),
            warmup_epochs=self.warmup_epochs,
        )
        
        # weight decay
        self.wd_schedule = trainer_utils.cosine_scheduler(
            self.weight_decay,
            self.weight_decay_end,
            self.max_epochs,
            len(data_loader),
        )       
        
    def __try_restart_from_last_checkpoint(self, nova_model:NOVAModel, dataloader:DataLoader):
        """Try restrat the training from the last checkpoint

        Args:
            nova_model (NOVAModel, optional): The model being used for the training. 
            dataloader (DataLoader, optional): The dataloader being used for the training. 
        """
        last_checkpoint_file_path = os.path.join(self.checkpoints_dir, f'{self.checkpoint_last_filename}.pth')
        if os.path.exists(last_checkpoint_file_path):  
            logging.info(f"NOTE: couldn't find a checkpoint file to restart from ({last_checkpoint_file_path})")
            return
        
        
        logging.info(f"Loading checkpoint from file {last_checkpoint_file_path}")
        checkpoint:CheckpointInfo = CheckpointInfo.load_from_checkpoint_filepath(last_checkpoint_file_path)
        
        # Test we are restarting from the same configurations
        assert self.training_config.__dict__ == checkpoint.training_config, f"Loaded checkpoint ({last_checkpoint_file_path}) doesn't have the same training configuration"
        assert nova_model.model_config.__dict__ == checkpoint.model_config, f"Loaded checkpoint ({last_checkpoint_file_path}) doesn't have the same model configuration"
        assert dataloader.dataset.conf.__dict__ == checkpoint.dataset_config, f"Loaded checkpoint ({last_checkpoint_file_path}) doesn't have the same dataset configuration"
        assert nova_model.model.state_dict() == checkpoint.model_dict, f"Loaded checkpoint ({last_checkpoint_file_path}) doesn't have the same model"

        # Init trainer from checkpoint
        self.__try_init_from_checkpoint(checkpoint)
        
    def __training_loop(self, nova_model:NOVAModel, data_loader_train: DataLoader, data_loader_val: DataLoader)->None:
        """Run the training loop over across all epochs, while handling checkpoint saving and early stopping

        Args:
            nova_model (NOVAModel): The model to train
            data_loader_train (DataLoader): The dataloader for the training dataset
            data_loader_val (DataLoader): The dataloader for the validation dataset
        """
        
        for epoch in tqdm(range(0, self.max_epochs)):
            # If restarted from a checkpoint
            if epoch < self.starting_epoch:
                continue
            
            self.optimizer.zero_grad()
            
            loss_val_avg = self.__run_one_epoch(nova_model.model, data_loader_train, data_loader_val, epoch)
            
            # Handle saving the checkpoint
            checkpoint_info = CheckpointInfo(model_dict = nova_model.model.state_dict(),
                                                        optimizer_dict=self.optimizer.state_dict(),
                                                        epoch=epoch,
                                                        training_config=self.training_config.__dict__,
                                                        dataset_config=data_loader_train.dataset.conf.__dict__,
                                                        model_config=nova_model.model_config.__dict__,
                                                        scaler_dict=self.scaler.state_dict(),
                                                        loss_val_avg=loss_val_avg,
                                                        early_stopping_counter=self.early_stopping_info.counter,
                                                        description=self.description)

            best_val_loss_avg = self.__get_best_val_loss()
            self.__handle_save_checkpoint(checkpoint_info, loss_val_avg, best_val_loss_avg)
            
            # Handle early stopping
            is_early_stopping_reached = self.__is_early_stopping_reached(loss_val_avg, best_val_loss_avg)
            if is_early_stopping_reached:
                break
    
    def __get_best_val_loss(self)->float:
        savepath_chkp_best = os.path.join(self.checkpoints_dir, self.checkpoint_best_filename)
        
        if not os.path.exists(savepath_chkp_best):
            return np.inf
        
        best_checkpoint = torch.load(savepath_chkp_best, map_location="cpu")
        best_checkpoint_val_loss_avg = best_checkpoint['val_loss_avg']
        
        return best_checkpoint_val_loss_avg
            
    def __handle_save_checkpoint(self, checkpoint_info: CheckpointInfo, loss_val_avg:float, best_loss_val_avg:float)->None:
        """Save running checkpoint and best checkpoint if we reached a new best 

        Args:
            checkpoint_info (CheckpointInfo): The info to saved into the checkpoint file
            loss_val_avg (float): The current average validation loss
            best_loss_val_avg (float): The best average validation loss so far
        """
        
        savepath_chkp_last = os.path.join(self.checkpoints_dir, self.checkpoint_last_filename)
        savepath_chkp_best = os.path.join(self.checkpoints_dir, self.checkpoint_best_filename)

        # Save running/latest checkpoint
        checkpoint_info.save(savepath_chkp_last)
        
        # Save best checkpoint
        
        # First best checkpoint or new best
        if loss_val_avg < best_loss_val_avg:
            # We have a new best checkpoint!
            checkpoint_info.save(checkpoint_info, savepath_chkp_best)
                
    def __is_early_stopping_reached(self, current_loss_value:float, best_loss_value:float)->bool:
        """Checks if we exhaused enough epochs without any improvement in order to early stop

        Args:
            current_loss_value (float): The current loss value 
            best_loss_value (float):  The best loss value to check against
        """
        
        if current_loss_value < best_loss_value:
            # Improved!
            self.early_stopping_info.reset()
            
            return False
            
        # No improvement
        self.early_stopping_info.counter -= 1
        logging.warn(f"No improvement. Early stopping counter is now {self.early_stopping_info.counter}")
        if self.early_stopping_info.counter <= 0:
            logging.warn(f"Stopping due to early stopping")
            return True
        
        return False
        
    def __run_one_epoch(self, model: torch.nn.Module, data_loader_train: DataLoader, data_loader_val: DataLoader, current_epoch: int)->float:
        """Run one epoch on the training set and then one on the validation set

        Args:
            model (torch.nn.Module): The model
            data_loader_train (DataLoader): The dataloader to get the training dataset from
            data_loader_val (DataLoader): The dataloader to get the validation dataset from
            current_epoch (int): The number of the current epoch

        Returns:
            float: The average loss on the validation set
        """
        logging.info(f"Epoch: [{current_epoch}/{self.max_epochs}]")
        
        loss_train_avg = self.__run_one_epoch_train(model, current_epoch, data_loader_train)
        loss_val_avg = self.__run_one_epoch_eval(model, current_epoch, data_loader_val)
        
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
            model_output = self.forward(model, res)
            with torch.cuda.amp.autocast():
                loss = self.loss(**model_output)
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
            self.tensorboard_writer.add_scalar('Loss/train', current_loss_value, current_epoch * len(data_loader) + it)
            self.tensorboard_writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], current_epoch * len(data_loader) + it)
            self.tensorboard_writer.add_scalar('Weight Decay', self.optimizer.param_groups[0]['weight_decay'], current_epoch * len(data_loader) + it)
            
            self.optimizer.zero_grad()  # Reset gradients after each update
            
        # Take the average loss on the training set
        loss_train_avg /= len(data_loader)
        
        logging.info(f"***** [epoch {current_epoch}] Averaged train loss: {loss_train_avg} *****")
        self.tensorboard_writer.add_scalar('Loss/train_avg', loss_train_avg, current_epoch)
        
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
                model_output = self.forward(model, res)
                with torch.cuda.amp.autocast():
                    loss = self.loss(**model_output)
                    
                logging.info(f"***** [epoch {current_epoch}] val loss: {loss.item()} *****")
                
                self.tensorboard_writer.add_scalar('Loss/val', loss.item(), current_epoch * len(data_loader) + it)
                loss_val_avg += loss.item()
                
            # Take the average loss on the validation set
            loss_val_avg /= len(data_loader)
            logging.info(f"***** [epoch {current_epoch}] Averaged val loss: {loss_val_avg} *****")
            self.tensorboard_writer.add_scalar('Loss/val_avg', loss_val_avg, current_epoch)
            
        return loss_val_avg
    
    
