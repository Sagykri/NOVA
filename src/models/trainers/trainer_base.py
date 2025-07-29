from abc import abstractmethod
import datetime
import logging
import math
import os
import sys
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from typing import Callable, Dict, List

from torch.utils.tensorboard import SummaryWriter

sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.common.utils import get_class, get_if_exists
from src.models.architectures.NOVA_model import NOVAModel
from src.models.utils.checkpoint_info import CheckpointInfo
from src.models.trainers.trainer_config import TrainerConfig
from src.models.utils.consts import CHECKPOINT_BEST_FILENAME, CHECKPOINT_LAST_FILENAME,\
                                                        CHECKPOINTS_FOLDERNAME, LOGS_FOLDERNAME,\
                                                            MODEL_FILENAME, TENSORBOARD_FOLDERNAME

class _EarlyStoppingInfo():
    """Holds information for handeling the early stopping
    """
    def __init__(self, init_value:int):
        """Get an instance

        Args:
            init_value (int): The number of allowed straight unimproved epochs
        """
        self.__init_value:int = init_value
        self.counter: int = self.__init_value
        
    def reset(self):
        """Reset the counter to its initial value
        """
        logging.info(f"Improved. Reseting the early stopping counter back to {self.__init_value}")
        self.counter: int = self.__init_value
        
    def try_decrease(self)->bool:
        self.counter -= 1
        logging.warning(f"No improvement. Early stopping counter is now {self.counter}")
        
        if self.is_empty():
            logging.warning(f"Stopping due to early stopping")
            return False
        
        return True
    
    def is_empty(self)->bool:
        return self.counter <= 0

class TrainerBase():
    def __init__(self, trainer_config:TrainerConfig, nova_model:NOVAModel):
        """Get an instance

        Args:
            conf (TrainerConfig): The trainer configuration
            nova_model (NOVAModel): The NOVA model to train
        """
        self.__set_params(trainer_config)     
        self.nova_model:NOVAModel = nova_model

        # For fine-tuning
        self.pretrained_model_path:str = get_if_exists(self.trainer_config, 'PRETRAINED_MODEL_PATH', None, verbose=True)
        if self.pretrained_model_path is not None:
            self.__load_weights_from_pretrained_model()
            
        layers_to_freeze = get_if_exists(self.trainer_config, 'LAYERS_TO_FREEZE', None)
        if layers_to_freeze is not None and len(layers_to_freeze) > 0:
            self.__try_freeze_layers(layers_to_freeze)
        
    @abstractmethod
    def loss(self, **kwargs)->float:
        """Calculating the loss

        Returns:
            float: The loss value
        """
        pass
    
    @abstractmethod
    def forward(self, X: torch.Tensor, y: torch.Tensor=None, paths: np.ndarray[str]=None) -> Dict:
        """Applying the forward pass (running the model on the given data)

        Args:
            X (torch.Tensor): The data to feed into the model
            y (torch.Tensor, optional): The ids for the labels. Defaults to None.
            paths (np.ndarray[str], optional): The paths to the files. Defaults to None.

        Returns:
            Dict: Results that will enter the 'loss' function
        """
        pass
    
    def train(self, data_loader_train: DataLoader, data_loader_val: DataLoader, data_loader_test: DataLoader)->None:
        """Train the model on the given data_loader_train and validate it on the data_loader_val

        Args:
            data_loader_train (DataLoader): The dataloader that would provide the dataset for training
            data_loader_val (DataLoader): The dataloader that would provide the dataset for the validation phase
            data_loader_test (DataLoader): The dataloader that would provide the dataset for the test phase
        """
        
        self.__init_params_for_training(data_loader_train)
        
        # Move model to gpu
        self.nova_model.model = self.nova_model.model.cuda()
        
        # Store dataloaders
        self.data_loader_train: DataLoader = data_loader_train
        self.data_loader_val: DataLoader = data_loader_val
        self.data_loader_test: DataLoader = data_loader_test
        
        # Set data augmentations
        self.data_loader_train.dataset.set_transform(self.data_augmentation)
        self.data_loader_val.dataset.set_transform(self.data_augmentation)
        
        self.__try_restart_from_last_checkpoint()
        
        # Run the training loop
        self.__training_loop()
        
        # Close the tensorboard writer
        self.tensorboard_writer.close()
        
        # Save the final model
        self.__save_model()
        
    #######################
    # Protected functions
    #######################
    
    def _freeze_layers(self, layers_names:List[str])->List[str]:
        """Freeze the given layers in the given model

        Args:
            layers_names (List[str]): The names of the layers to freeze

        Returns:
            List[str]: The names of the layers that were successfully freezed 
        """
        freezed_layers = []
        
        if len(layers_names) == 0:
            logging.warning("len(layers_names) == 0 -> No layer got frozen")
            return
        
        # Freeze the specified layers
        for name, param in self.nova_model.model.named_parameters():
            if any(layer_name in name for layer_name in layers_names):
                param.requires_grad = False
                freezed_layers.append(name)
                
        assert set(layers_names) == set(freezed_layers), f"Mismatch in layers to freeze: {set(layers_names).difference(set(freezed_layers))}, {set(freezed_layers).difference(set(layers_names))}"
        
        return freezed_layers
        
    #######################
    # Private functions
    #######################
        
    def __set_params(self, trainer_config:TrainerConfig)->None:
        """Extracting params from the configuration

        Args:
            trainer_config (TrainingConfig): The training configuration
        """
        self.trainer_config:TrainerConfig = trainer_config
        
        self.description:str = get_if_exists(self.trainer_config, 'DESCRIPTION', str(type(self)))
        
        self.__set_output_dirs()
        
        self.checkpoint_last_path:str = os.path.join(self.checkpoints_dir, CHECKPOINT_LAST_FILENAME)
        self.checkpoint_best_path:str = os.path.join(self.checkpoints_dir, CHECKPOINT_BEST_FILENAME)
        self.final_model_path:str     =  os.path.join(self.trainer_config.OUTPUTS_FOLDER, MODEL_FILENAME)
        
        self.early_stopping_info:_EarlyStoppingInfo = _EarlyStoppingInfo(self.trainer_config.EARLY_STOPPING_PATIENCE)
        self.starting_epoch:int = 0
        self.optimizer:torch.optim.Optimizer = None
        self.scaler:torch.cuda.amp.GradScaler = torch.cuda.amp.GradScaler()
        self.lr_schedule:np.ndarray = None
        self.wd_schedule:np.ndarray = None
        self.tensorboard_writer:SummaryWriter = SummaryWriter(log_dir=self.tensorboard_dir)
        self.best_avg_val_loss: float = np.inf
        
    def __set_output_dirs(self)->None:
        """Set the path for the output directories (logs, checkpoints and tensorboard plots)
        """
        
        assert self.trainer_config.OUTPUTS_FOLDER is not None, "outputs folder can't be None"
        
        self.logs_dir = os.path.join(self.trainer_config.OUTPUTS_FOLDER, LOGS_FOLDERNAME)
        self.tensorboard_dir = os.path.join(self.trainer_config.OUTPUTS_FOLDER, TENSORBOARD_FOLDERNAME, f"{datetime.datetime.now().strftime('%d%m%y_%H%M%S_%f')}_JID{os.getenv('LSB_JOBID')}")
        self.checkpoints_dir = os.path.join(self.trainer_config.OUTPUTS_FOLDER, CHECKPOINTS_FOLDERNAME)
        
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
    def __try_init_from_checkpoint(self, checkpoint:CheckpointInfo)->None:
        """Init trainer's params from the checkpoint file

        Args:
            checkpoint (CheckpointInfo): The checkpoint to init from
        """
        self.starting_epoch = checkpoint.epoch + 1
        self.optimizer.load_state_dict(checkpoint.optimizer_dict)
        self.scaler.load_state_dict(checkpoint.scaler_dict)
        self.description = checkpoint.description
        self.early_stopping_info = _EarlyStoppingInfo(checkpoint.early_stopping_counter)
        self.best_avg_val_loss = checkpoint.best_avg_val_loss
        self.nova_model.model.load_state_dict(checkpoint.model_dict)
        
        # Use the exact same dataset as has been used in the checkpoint
        logging.info("Loading dataset split from checkpoint")
        self.data_loader_train.dataset.set_Xy(checkpoint.trainset_paths, checkpoint.trainset_labels)
        self.data_loader_val.dataset.set_Xy(checkpoint.valset_paths, checkpoint.valset_labels)
        self.data_loader_test.dataset.set_Xy(checkpoint.testset_paths, checkpoint.testset_labels)

        torch.set_rng_state(checkpoint.rng_state)
        torch.cuda.set_rng_state_all(checkpoint.cuda_rng_state)
        
        logging.info(f"Checkpoint has been loaded successfully. Starting epoch was set to {self.starting_epoch}")
        
    def __init_params_for_training(self, data_loader: DataLoader)->None:
        """Init the needed objects for the training (optimizer, schedulers, etc.)

        Args:
            data_loader (torch.utils.data.DataLoader): The dataloader
        """
        params_groups = self.__get_params_groups_for_optimizer()
        self.optimizer = torch.optim.AdamW(params_groups)

        # learning rate schedule
        self.lr_schedule = self.__get_cosine_scheduler(
            self.trainer_config.LR * (self.trainer_config.BATCH_SIZE) / 256.0,  # linear scaling rule
            self.trainer_config.MIN_LR,
            self.trainer_config.MAX_EPOCHS,
            len(data_loader),
            warmup_epochs=self.trainer_config.WARMUP_EPOCHS,
        )
        
        # weight decay
        self.wd_schedule = self.__get_cosine_scheduler(
            self.trainer_config.WEIGHT_DECAY,
            self.trainer_config.WEIGHT_DECAY_END,
            self.trainer_config.MAX_EPOCHS,
            len(data_loader),
        )       
        
        logging.info(f"Creating data augmentation object (from class {self.trainer_config.DATA_AUGMENTATION_CLASS_PATH})")
        data_augmentation_class:Callable = get_class(self.trainer_config.DATA_AUGMENTATION_CLASS_PATH)
        
        logging.info(f"Instantiate data augmentation object from class {data_augmentation_class.__name__}")
        self.data_augmentation = data_augmentation_class()
      
  
    def __load_weights_from_pretrained_model(self):
        """Loads the weights from a given pretrained model path, while changing the output dimension of the head to the new output_dim
        """
        if self.pretrained_model_path is None:
            logging.warning("'pretrained_model_path' was set to None. Can't load pretrained model.")
            return
        
        assert os.path.exists(self.pretrained_model_path), f"The path to the pretrained model isn't exists ({self.pretrained_model_path})"
        
        logging.info(f"Loading pretrained model ({self.pretrained_model_path})")
        pretrained_model = NOVAModel.load_from_checkpoint(self.pretrained_model_path)

        # Load the pretrained state_dict
        pretrained_dict = pretrained_model.model.state_dict()

        # Filter out keys related to the head
        filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('head.')}

        # Load model’s state dict and update with the filtered pretrained weights
        model_dict = self.nova_model.model.state_dict()
        model_dict.update(filtered_dict)

        # 4. Load the updated dict
        self.nova_model.model.load_state_dict(model_dict)

        logging.info(f"The updated head is: {self.nova_model.model.head}")
        
    def __try_freeze_layers(self, layers_to_freeze:List[str]):
        """Trying to freeze layers based on the LAYERS_TO_FREEZE param in the config, if exists
        """
        
        if len(layers_to_freeze) == 0:
            logging.warning("No layers to freeze")
            return
        
        # Freezing the layers
        logging.info(f"Freeze layers: {layers_to_freeze}")
        _freezed_layers = self._freeze_layers(layers_to_freeze)
        logging.info(f"Layers freezed successfully : {_freezed_layers}")
  
    def __try_restart_from_last_checkpoint(self):
        """Try restrat the training from the last checkpoint
        """
        if not os.path.exists(self.checkpoint_last_path):  
            logging.info(f"NOTE: couldn't find a checkpoint file to restart from ({self.checkpoint_last_path})")
            return
        
        
        logging.info(f"Loading checkpoint from file {self.checkpoint_last_path}")
        checkpoint:CheckpointInfo = CheckpointInfo.load_from_checkpoint_filepath(self.checkpoint_last_path)
        
        # Test that we are restarting from the same configurations
        assert self.trainer_config.is_equal(checkpoint.trainer_config), f"Loaded checkpoint ({self.checkpoint_last_path}) doesn't have the same training configuration"
        assert self.nova_model.model_config.is_equal(checkpoint.model_config), f"Loaded checkpoint ({self.checkpoint_last_path}) doesn't have the same model configuration"
        assert self.data_loader_train.dataset.dataset_config.is_equal(checkpoint.dataset_config), f"Loaded checkpoint ({self.checkpoint_last_path}) doesn't have the same dataset configuration"
        assert self.nova_model.is_equal_architecture(checkpoint.model_dict), f"Loaded checkpoint ({self.checkpoint_last_path}) doesn't have the same model"

        # Init trainer from checkpoint
        self.__try_init_from_checkpoint(checkpoint)
        
    def __training_loop(self)->None:
        """Run the training loop over across all epochs, while handling checkpoint saving and early stopping
        """
        
        for epoch in tqdm(range(0, self.trainer_config.MAX_EPOCHS)):
            # If restarted from a checkpoint
            if epoch < self.starting_epoch:
                continue
            
            if self.early_stopping_info.is_empty():
                # Activate early stopping
                logging.warning(f"Stopping due to early stopping")
                break
            
            self.optimizer.zero_grad()
            
            avg_val_loss = self.__run_one_epoch(epoch)
                
            is_improved = self.__is_better_val_loss(avg_val_loss)
            
            if is_improved:
                # set the current loss to be the best one reached
                self.best_avg_val_loss = avg_val_loss
        
            self.__update_eary_stopping(is_improved)
            
            # Handle saving the checkpoint
            checkpoint_info = CheckpointInfo(model_dict = self.nova_model.model.state_dict(),
                                                        optimizer_dict=self.optimizer.state_dict(),
                                                        epoch=epoch,
                                                        trainer_config=self.trainer_config,
                                                        dataset_config=self.data_loader_train.dataset.get_configuration(),
                                                        model_config=self.nova_model.model_config,
                                                        scaler_dict=self.scaler.state_dict(),
                                                        avg_val_loss=avg_val_loss,
                                                        best_avg_val_loss=self.best_avg_val_loss,
                                                        early_stopping_counter=self.early_stopping_info.counter,
                                                        trainset_paths=self.data_loader_train.dataset.get_X_paths(),
                                                        trainset_labels=self.data_loader_train.dataset.get_y(),
                                                        valset_paths=self.data_loader_val.dataset.get_X_paths(),
                                                        valset_labels=self.data_loader_val.dataset.get_y(),
                                                        testset_paths=self.data_loader_test.dataset.get_X_paths(),
                                                        testset_labels=self.data_loader_test.dataset.get_y(),
                                                        description=self.description)    
            
            self.__handle_checkpoint_saving(is_improved, checkpoint_info)
    
    def __update_eary_stopping(self, is_improved:bool):
        """Reset the early stopping counter if is_improved=True, otherwise decrease it

        Args:
            is_improved (bool): Is improved?
        """
        if is_improved:
            # reset early stopping
            self.early_stopping_info.reset()
            return
        
        # no improvement
        self.early_stopping_info.try_decrease()
        
    def __handle_checkpoint_saving(self, is_improved:bool, checkpoint_info:CheckpointInfo):
        """Save checkpoint as last, and also as best if is_improved=True

        Args:
            is_improved (bool): Is improved?
            checkpoint_info (CheckpointInfo): The checkpoint to save
        """
        if is_improved:
            # Save as best checkpoint
            checkpoint_info.save(self.checkpoint_best_path)
            
        # Save as last checkpoint
        checkpoint_info.save(self.checkpoint_last_path)
            
    def __is_better_val_loss(self, avg_val_loss:float)->bool:
        """Is the given loss is better than the current best val loss

        Args:
            avg_val_loss (float): The given val loss

        Returns:
            bool: Is the given loss the new best?
        """
        return avg_val_loss < self.best_avg_val_loss
        
    def __run_one_epoch(self, current_epoch: int)->float:
        """Run one epoch on the training set and then one on the validation set

        Args:
            current_epoch (int): The number of the current epoch

        Returns:
            float: The average loss on the validation set
        """
        logging.info(f"Epoch: [{current_epoch}/{self.trainer_config.MAX_EPOCHS}]")
        
        
        logging.info(f"--------------------------------------- TRAINING (epoch={current_epoch}) ---------------------------------------")
        avg_train_loss = self.__run_one_epoch_train(current_epoch, self.data_loader_train)
        
        logging.info(f"--------------------------------------- EVAL (epoch={current_epoch}) ---------------------------------------")
        avg_val_loss = self.__run_one_epoch_eval(current_epoch, self.data_loader_val)
        
        logging.info(f"--------------------------------------- TEST (epoch={current_epoch}) ---------------------------------------")
        avg_test_loss = self.__run_one_epoch_eval(current_epoch, self.data_loader_test, is_testset=True)

        self.tensorboard_writer.add_scalars('1. Loss/All', {'Train': avg_train_loss, 'Validation': avg_val_loss, 'Test': avg_test_loss}, current_epoch)
        
        return avg_val_loss

    def __run_one_epoch_train(self, current_epoch:int, data_loader:DataLoader)->float:
        """Run one epoch on the given dataloader in train mode

        Args:
            current_epoch (int): The current epoch
            data_loader (DataLoader): The dataloader to get the data from

        Returns:
            float: The average loss on the training set
        """
        
        self.nova_model.model.train()
        
        avg_train_loss = 0
        for it, res in enumerate(data_loader):
            logging.info(f"[Training epoch={current_epoch}] Batch number: {it}/{len(data_loader)-1}")
            
            self.__handle_scehdulers(it, current_epoch, data_loader)

            with torch.cuda.amp.autocast():
                X, y, path = res
                X, y = X.cuda(), y.cuda()
                # forward pass
                model_output = self.forward(X, y, path)
                # calc loss value
                loss = self.loss(**model_output)
            current_loss_value = loss.item()
            avg_train_loss += current_loss_value

            if not math.isfinite(current_loss_value):
                logging.info("Loss is {}, stopping training".format(current_loss_value))
                raise Exception(f"Loss is {current_loss_value}, stopping training")
                
            logging.info(f"***** [epoch {current_epoch}, batch {it}] Train loss: {current_loss_value} *****")
            
            logging.info("Calculating grads")
            self.__calculate_gradients(loss)
                        
            logging.info("Updating model")
            self.__update_model_weights()
            
            # Write to tensorboard
            self.tensorboard_writer.add_scalar('1. Loss/Train Steps', current_loss_value, current_epoch * len(data_loader) + it)
            self.tensorboard_writer.add_scalar('2. Optimizer/Learning Rate', self.optimizer.param_groups[0]['lr'], current_epoch * len(data_loader) + it)
            self.tensorboard_writer.add_scalar('2. Optimizer/Weight Decay', self.optimizer.param_groups[0]['weight_decay'], current_epoch * len(data_loader) + it)
            
            self.optimizer.zero_grad()  # Reset gradients after each update
            
        # Take the average loss on the training set
        avg_train_loss /= len(data_loader)
        
        logging.info(f"***** [epoch {current_epoch}] Averaged train loss: {avg_train_loss} *****")
        self.tensorboard_writer.add_scalar('1. Loss/Train Epochs', avg_train_loss, current_epoch)

        
        return avg_train_loss
    
    def __handle_scehdulers(self, current_iteration:int, current_epoch:int, data_loader: DataLoader)->None:
        """Setting the learning rate and weight_decay based on the scheduler

        Args:
            current_epoch (int): The current epoch
            data_loader (DataLoader): The dataloader
        """
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * current_epoch + current_iteration  # global training iteration
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[it]
    
    def __calculate_gradients(self, loss_value: torch.Tensor):
        """Calculate gradients based on the loss

        Args:
            loss_value (torch.Tensor): The loss
        """
        self.scaler.scale(loss_value).backward()
        self.scaler.unscale_(self.optimizer)  # Unscale gradients    
        torch.nn.utils.clip_grad_norm_(self.nova_model.model.parameters(), max_norm=3.0)  # Clip gradients
        
    def __update_model_weights(self):
        """Update the weights of the model
        """
        self.scaler.step(self.optimizer)  # Update weights
        self.scaler.update()
                    
    def __run_one_epoch_eval(self, current_epoch:int, data_loader:DataLoader, is_testset:bool=False)->float:
        """Run one epoch on the given dataloader in eval mode

        Args:
            current_epoch (int): The current epoch
            data_loader (DataLoader): The dataloader to get the data from
            is_testset (bool, optional): Is the dataset is the testset. Default to False (i.e. valset).
        Returns:
            float: The average loss on the validation set
        """
        
        title:str = 'Test' if is_testset else 'Val'
        
        self.nova_model.model.eval()
        
        with torch.no_grad():
            avg_eval_loss = 0
            for it, res in enumerate(data_loader):
                logging.info(f"[{title} epoch={current_epoch}] Batch number: {it}/{len(data_loader)-1}")
                with torch.cuda.amp.autocast():
                    X, y, path = res
                    X, y = X.cuda(), y.cuda()
                    # forward pass
                    model_output = self.forward(X, y, path)
                    # calc loss value
                    loss = self.loss(**model_output)
                
                logging.info(f"***** [epoch {current_epoch}, batch {it}] {title} loss: {loss.item()} *****")
                
                self.tensorboard_writer.add_scalar(f'1. Loss/{title} Steps', loss.item(), current_epoch * len(data_loader) + it)
                avg_eval_loss += loss.item()
                
            # Take the average loss on the validation set
            avg_eval_loss /= len(data_loader)
            logging.info(f"***** [epoch {current_epoch}] Averaged {title} loss: {avg_eval_loss} *****")
            self.tensorboard_writer.add_scalar(f'1. Loss/{title} Epochs', avg_eval_loss, current_epoch)

            
        return avg_eval_loss
    
    def __get_params_groups_for_optimizer(self)->List[Dict]:
        """Get params for the optimizier.\n
        regularized all params except for the bias ones

        Returns:
            List[Dict]: The params
        """
        regularized:List[torch.Tensor] = []
        not_regularized:List[torch.Tensor] = []
        
        for name, param in self.nova_model.model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]

    def __get_cosine_scheduler(
        self,
        base_value:float, 
        final_value:float, 
        epochs:int, 
        niter_per_ep:int, 
        warmup_epochs:int=0, 
        start_warmup_value:int=0)->np.ndarray:
        """Get cosine scheduler's schedules

        Args:
            base_value (float): The value to start from
            final_value (float): The final value to reach to
            epochs (int): Number of epochs
            niter_per_ep (int): Number of iterations per epoch
            warmup_epochs (int, optional): Number of epochs for warmup. Defaults to 0.
            start_warmup_value (int, optional): The starting value during the warmup. Defaults to 0.

        Returns:
            np.ndarray: The schedule
        """

        # Calculate the number of warmup iterations
        warmup_iters = warmup_epochs * niter_per_ep
        
        # Create the warmup schedule if warmup is required
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        else:
            warmup_schedule = np.array([])

        # Calculate the number of iterations for the cosine schedule
        total_iters = epochs * niter_per_ep - warmup_iters
        iters = np.arange(total_iters)
        
        # Create the cosine schedule
        cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / total_iters)
        )

        # Concatenate warmup and cosine schedules
        schedule = np.concatenate((warmup_schedule, cosine_schedule))
        
        # Ensure the schedule has the correct number of elements
        assert len(schedule) == epochs * niter_per_ep, "Schedule length does not match the expected number of iterations."
        
        return schedule
    
    def __save_model(self):
        """Saves the final model to a file
        """
        logging.info(f"Saving the final model to {self.final_model_path}")
        torch.save(self.nova_model.model.state_dict(), self.final_model_path)
        logging.info(f"The final model was saved successfully")
