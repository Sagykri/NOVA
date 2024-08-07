# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random
import subprocess
import sys
import datetime
import math


from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
# import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import transforms


from torch.utils.tensorboard import SummaryWriter
import logging


sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

from src.common.lib.data_loader import get_dataloader
from src.common.lib.dataset import Dataset
from src.common.lib.utils import init_logging, load_config_file
from src.datasets.dataset_spd import DatasetSPD
										  

from sandbox.eval_new_arch.dino4cells.utils import utils
from sandbox.eval_new_arch.dino4cells.archs import vision_transformer as vits
from sandbox.eval_new_arch.dino4cells.utils.config_utils import load_dict_from_pyfile, DictToObject

def train_vit(config_path, config_data_path):
    jobid = os.getenv('LSB_JOBID')
    jobname = os.getenv('LSB_JOBNAME')

    username = 'UnknownUser'
    if jobid:
        # Run the bjobs command to get job details
        result = subprocess.run(['bjobs', '-o', 'user', jobid], capture_output=True, text=True, check=True)
        # Extract the username from the output
        username = result.stdout.replace('USER', '').strip()
    
    config = DictToObject(load_dict_from_pyfile(config_path))
    __now = datetime.datetime.now()
    __now = __now.strftime("%d%m%y_%H%M%S_%f")
    
    if hasattr(config, 'logs_dir'):
        logs_dir = config.logs_dir
        
        log_file_path = os.path.join(logs_dir, f'{__now}_{jobid}_{username}_{jobname}.log')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        init_logging(log_file_path)
        logging.info(f"config_path = {config_path}, config_data_path = {config_data_path}")
    print(f"config_path = {config_path}, config_data_path = {config_data_path}")

    writer = SummaryWriter(log_dir=os.path.join(config.tensorboard_root_folder, f"{__now}_{jobid}_{username}_{jobname}"))


    utils.fix_random_seeds(config.seed)
    cudnn.benchmark = False
    random.seed(config.seed)



    transform =  DataAugmentationVIT(config=config)

    
    config_data = load_config_file(config_data_path)

    dataset = DatasetSPD(config_data, transform)

    logging.info("Split data...")
    train_indexes, val_indexes, _ = dataset.split()
    
    dataset_train = Dataset.get_subset(dataset, train_indexes)
    dataset_val = Dataset.get_subset(dataset, val_indexes)
    
    data_loader = get_dataloader(dataset_train, config.batch_size_per_gpu, num_workers=config.num_workers)
    data_loader_val = get_dataloader(dataset_val, config.batch_size_per_gpu, num_workers=config.num_workers)

    logging.info(f"Data loaded: there are {len(dataset_train)} images.")
    
    create_vit = vits.vit_base
    if config.vit_version == 'base':
        create_vit = vits.vit_base
    elif config.vit_version == 'small':
        create_vit = vits.vit_small
    elif config.vit_version == 'tiny':
        create_vit = vits.vit_tiny
        
    logging.info(f"Vit version = {config.vit_version} ({create_vit})")
    logging.info(f"learning rate = {config.lr}")
    logging.info(f"len(unique_markers) (i.e. num_classes): {len(dataset_train.unique_markers)}")
    
    logging.info(f"config model: {config.__dict__}")
    logging.info(f"config data: {config_data.__dict__}")
    
    model = create_vit(
            img_size=[config.embedding.image_size],
            patch_size=config.patch_size,
            in_chans=config.num_channels,
            num_classes=len(dataset_train.unique_markers)
    )
    
    model = model.cuda()
    
    vit_loss = nn.CrossEntropyLoss().cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(model)
    optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs

    fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        config.lr
        * (config.batch_size_per_gpu)
        / 256.0,  # linear scaling rule
        config.min_lr,
        config.epochs,
        len(data_loader),
        warmup_epochs=config.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        config.weight_decay,
        config.weight_decay_end,
        config.epochs,
        len(data_loader),
    )

    logging.info("Loss, optimizer and schedulers ready.")

    start_epoch = 0
    
    last_best_val_loss = np.inf
    early_stopping_patience = config.early_stopping_patience

    # start_time = time.time()
    logging.info("Starting VIT training !")
    for epoch in tqdm(range(start_epoch, config.epochs)):
        
        optimizer.zero_grad()
        
        # ============ training one epoch ... ============
        loss_val_avg = train_one_epoch(
            model,
            vit_loss,
            data_loader,
            data_loader_val,
            optimizer,
            lr_schedule,
            wd_schedule,
            epoch,
            fp16_scaler,
            config,
            writer
        )
        
        # Save running checkpoint
        save_checkpoint(model, optimizer, epoch, config, config_data, vit_loss, fp16_scaler, loss_val_avg, "checkpoint_last")
        
        # Save best checkpoint
        checkpoint_best_path = os.path.join(config.output_dir, f"checkpoint_best.pth")
        if not os.path.exists(checkpoint_best_path):
            save_checkpoint(model, optimizer, epoch, config, config_data, vit_loss, fp16_scaler, loss_val_avg, "checkpoint_best")
        else:
            best_checkpoint = torch.load(checkpoint_best_path, map_location="cpu")
            best_checkpoint_val_loss_avg = best_checkpoint['val_loss_avg']
            if loss_val_avg < best_checkpoint_val_loss_avg:
                save_checkpoint(model, optimizer, epoch, config, config_data, vit_loss, fp16_scaler, loss_val_avg, "checkpoint_best")

        if last_best_val_loss < loss_val_avg:
            early_stopping_patience -= 1
            logging.warn(f"No improvement. ES patience is now {early_stopping_patience}")
            if early_stopping_patience <= 0:
                logging.warn(f"Stopping due to early stopping")
                break
        else:
            last_best_val_loss = loss_val_avg
            early_stopping_patience = config.early_stopping_patience
    
    writer.close()


def save_checkpoint(model, optimizer, epoch, config, config_data, vit_loss, fp16_scaler, val_loss_avg, filename):
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": config.__dict__,
        "config_data": config_data.__dict__,
        "vit_loss": vit_loss.state_dict(),
        'fp16_scaler': fp16_scaler.state_dict(),
        'val_loss_avg': val_loss_avg
    }
    if not os.path.exists(config.output_dir):
        logging.info(f"{config.output_dir} doesn't exist. Creating dir")
        os.makedirs(config.output_dir)
        
    savepath = os.path.join(config.output_dir, f"{filename}.pth")
    logging.info(f"Saving checkpoint to file {savepath}")
    torch.save(
        save_dict, savepath
    )
    
    return savepath

def forward_pass(model, res, vit_loss):    
    with torch.cuda.amp.autocast():
        images, targets = res['image'].to(torch.float).cuda(), res['label'].cuda()
        
        logging.info(f"images shape: {images.shape}, targets shape: {targets.shape}")
        preds = model(images)
										 
        loss = vit_loss(preds, targets)				   

    
    return loss

def infer_pass(model, data_loader, return_cls_token=False):
    all_predictions = []
    all_labels = []
    
    if return_cls_token:
        all_outputs = []
    
    model.eval()
    with torch.no_grad():
        for it, res in enumerate(data_loader):
            logging.info(f"batch number: {it}/{len(data_loader)}")
            images, targets = res['image'].to(torch.float).cuda(), res['label'].cuda()
            preds, outputs = model(images, return_hidden=True)
            all_predictions.extend(preds.cpu())
            all_labels.extend(targets.cpu())
            
            if return_cls_token:
                all_outputs.append(outputs.cpu())

    if return_cls_token:
        return np.asarray(all_predictions), np.asarray(all_labels), np.vstack(all_outputs)

    return np.asarray(all_predictions), np.asarray(all_labels), None

                

def train_one_epoch(
    model,
    vit_loss,
    data_loader,
    data_loader_val,
    optimizer,
    lr_schedule,
    wd_schedule,
    epoch,
    fp16_scaler,
    args,
    writer
):
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    logging.info(header)
    logging.info(f"**************************** TRAINING (epoch={epoch}) ***********************************")
    
    loss_train_avg = 0
    
    model.train()
    for it, res in enumerate(data_loader):
        logging.info(f"[Training epoch={epoch}] batch number: {it}/{len(data_loader)}")
        
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # loss = forward_pass(model, res, vit_loss, epoch)
        loss = forward_pass(model, res, vit_loss)
        # loss = loss / args.accumulation_steps  # Normalize the loss
        running_loss = loss.item()

        if not math.isfinite(running_loss):
            logging.info("Loss is {}, stopping training".format(running_loss))
            raise Exception(f"Loss is {running_loss}, stopping training")
            
        
        fp16_scaler.scale(loss).backward()
        
        logging.info(f"***** [epoch {epoch}] train loss: {running_loss} *****")
        
        logging.info("Calculating grads")
        
        fp16_scaler.unscale_(optimizer)  # Unscale gradients    
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)  # Clip gradients
                    
            
        logging.info("Updating model")
        fp16_scaler.step(optimizer)  # Update weights
        fp16_scaler.update()
        
        
        writer.add_scalar('1. Loss/train', running_loss, epoch * len(data_loader) + it)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch * len(data_loader) + it)
        writer.add_scalar('Weight Decay', optimizer.param_groups[0]['weight_decay'], epoch * len(data_loader) + it)

        loss_train_avg += running_loss
        optimizer.zero_grad()  # Reset gradients after each update
    
    loss_train_avg /= len(data_loader)
    logging.info(f"***** [epoch {epoch}] AVG train loss: {loss_train_avg} *****")
    writer.add_scalar('1. Loss/train_avg', loss_train_avg, epoch )
            
                
    # Eval
    logging.info(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EVAL (epoch={epoch}) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    model.eval()
    with torch.no_grad():
        loss_val_avg = 0
        for it, res in enumerate(data_loader_val):
            logging.info(f"[Validation epoch={epoch}] batch number: {it}/{len(data_loader_val)}")
            loss = forward_pass(model, res, vit_loss)
                
            logging.info(f"***** [epoch {epoch}] val loss: {loss.item()} *****")
            
            writer.add_scalar('1. Loss/val', loss.item(), epoch * len(data_loader_val) + it)
            loss_val_avg += loss.item()
            
        loss_val_avg /= len(data_loader_val)
        logging.info(f"***** [epoch {epoch}] AVG val loss: {loss_val_avg} *****")
        writer.add_scalar('1. Loss/val_avg', loss_val_avg, epoch)
        
    writer.add_scalars('Loss', {'Training': loss_train_avg, 'Validation': loss_val_avg}, epoch)
        
    
    return loss_val_avg
        


class DataAugmentationVIT(object):
    def __init__(self, config):
        self.config = config

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    def __call__(self, image):
        img = self.transform(image)    
        return img

									
							   