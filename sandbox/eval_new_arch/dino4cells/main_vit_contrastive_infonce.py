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
import argparse
import os
import random
import subprocess
import sys
import datetime
import time
import math
import json
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt

import yaml

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
# import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

from info_nce import InfoNCE

from torch.utils.tensorboard import SummaryWriter
import logging

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score

os.environ['MOMAPS_HOME'] = '/home/labs/hornsteinlab/Collaboration/MOmaps_Noam/MOmaps'
os.environ['MOMAPS_DATA_HOME'] = '/home/labs/hornsteinlab/Collaboration/MOmaps/input'

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

from src.common.lib.data_loader import get_dataloader
from src.common.lib.dataset import Dataset
from src.common.lib.utils import init_logging, load_config_file, flat_list_of_lists
from src.datasets.dataset_spd import DatasetSPD

from sandbox.eval_new_arch.dino4cells.utils import utils
from sandbox.eval_new_arch.dino4cells.archs import vision_transformer as vits
from sandbox.eval_new_arch.dino4cells.archs.vision_transformer import DINOHead


class RandomIntensityChange:
    def __init__(self, intensity_range=(0.5, 1.5), p=0.5):
        self.intensity_range = intensity_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            factor = random.uniform(*self.intensity_range)
            img = img * factor
            img = torch.clamp(img, 0, 1)
        return img

class RandomChannelShutdown:
    def __init__(self, channel_index=1, p=0.5):
        self.channel_index = channel_index
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img[:,self.channel_index, ...] = 0
        return img

def train_vit_contrastive(config, config_data_path):
    
    __now = datetime.datetime.now()
    writer = SummaryWriter(log_dir=os.path.join(config.tensorboard_root_folder, __now.strftime("%d%m%y_%H%M%S_%f")))
    
    
    if hasattr(config, 'logs_dir'):
        logs_dir = config.logs_dir
        jobid = os.getenv('LSB_JOBID')
        
        username = 'UnknownUser'
        if jobid:
            # Run the bjobs command to get job details
            result = subprocess.run(['bjobs', '-o', 'user', jobid], capture_output=True, text=True, check=True)
            # Extract the username from the output
            username = result.stdout.replace('USER', '').strip()
        log_file_path = os.path.join(logs_dir, __now.strftime("%d%m%y_%H%M%S_%f") + f'_{jobid}_{username}.log')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        init_logging(log_file_path)
        logging.info(f"config_data_path={config_data_path}")
    print(f"config_data_path={config_data_path}")

    utils.fix_random_seeds(config.seed)
    cudnn.benchmark = False
    random.seed(config.seed)

    # ============ preparing data ... ============
    transform =  DataAugmentationVIT(config=config)
    config_data = load_config_file(config_data_path)

    dataset_train = DatasetSPD(config_data, transform)
    train_indexes, val_indexes, test_indexes = None, None, None

    if config_data.SPLIT_DATA:
        logging.info("Split data...")
        train_indexes, val_indexes, test_indexes = dataset_train.split()


    ################# TAKING ONLY PART OF THE DATA ################
    quick_train_indices = train_indexes[:]
    quick_val_indices = val_indexes[:]
    # logging.info(f"[WARNING!] TAKING ONLY PART OF THE DATA!! train: {len(quick_train_indices)} val: {len(quick_val_indices)}")
    ############################################################  
    
    dataset = Dataset.get_subset(dataset_train, quick_train_indices)
    dataset_val = Dataset.get_subset(dataset_train, quick_val_indices)
        
    data_loader = get_dataloader(dataset, config.batch_size_per_gpu, num_workers=config.num_workers)
    data_loader_val = get_dataloader(dataset_val, config.batch_size_per_gpu, num_workers=config.num_workers)

    logging.info(f"Data loaded: there are {len(dataset)} images.")

    create_vit = vits.vit_base
    if config.vit_version == 'base':
        create_vit = vits.vit_base
    elif config.vit_version == 'small':
        create_vit = vits.vit_small
    elif config.vit_version == 'tiny':
        create_vit = vits.vit_tiny

    logging.info(f"Vit version = {config.vit_version} ({create_vit})")
    logging.info(f"learning rate = {config.lr}")
    logging.info(f"num_classes = {config.num_classes}")
    logging.info('positive are from random rep (but not same index, meaning not the same *tile*)')
    # logging.info("positive are from different reps and same batches, negatives are from the same batch")

    model = create_vit(
            img_size=[config.embedding.image_size],
            patch_size=config.patch_size,
            in_chans=config.num_channels,
            num_classes=config.num_classes
    )

    model = model.cuda()
    
    vit_loss = VITContrastiveLoss(config.negative_count).cuda()

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

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}

    start_epoch = to_restore["epoch"]
    
    last_best_val_loss = np.inf
    early_stopping_patience = config.early_stopping_patience

    # start_time = time.time()
    logging.info("Starting VIT training !")
    for epoch in tqdm(range(start_epoch, config.epochs)):
        
        optimizer.zero_grad()
        
        # ============ training one epoch... ============
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
        save_checkpoint(model, optimizer, epoch, config, vit_loss, fp16_scaler, loss_val_avg, "checkpoint_last")
        
        # Save best checkpoint
        checkpoint_best_path = os.path.join(config.output_dir, f"checkpoint_best.pth")
        if not os.path.exists(checkpoint_best_path):
            save_checkpoint(model, optimizer, epoch, config, vit_loss, fp16_scaler, loss_val_avg, "checkpoint_best")
        else:
            best_checkpoint = torch.load(checkpoint_best_path, map_location="cpu")
            best_checkpoint_val_loss_avg = best_checkpoint['val_loss_avg']
            if loss_val_avg < best_checkpoint_val_loss_avg:
                save_checkpoint(model, optimizer, epoch, config, vit_loss, fp16_scaler, loss_val_avg, "checkpoint_best")

        if last_best_val_loss < loss_val_avg:
            early_stopping_patience -= 1
            logging.warn(f"No improvement. ES patience is now {early_stopping_patience}")
            if early_stopping_patience <= 0:
                logging.warn(f"Stopping due to early stopping")
                break
        else:
            last_best_val_loss = loss_val_avg
            early_stopping_patience = config.early_stopping_patience


    #     # ============ writing logs ... ============
        # save_checkpoint(model, optimizer, epoch, config, vit_loss, fp16_scaler, np.inf, "checkpoint")
    #     save_dict = {
    #         "model": model.state_dict(),
    #         "optimizer": optimizer.state_dict(),
    #         "epoch": epoch,
    #         "config": config,
    #         "vit_loss": vit_loss.state_dict(),
    #         'fp16_scaler': fp16_scaler.state_dict()
    #     }
    # #     if fp16_scaler is not None:
    #     # save_dict["fp16_scaler"] = fp16_scaler.state_dict()
    # #     utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
    # #     if args.saveckp_freq and epoch % args.saveckp_freq == 0:
    #     utils.save_on_master(
    #         save_dict, os.path.join(config.output_dir, f"checkpoint.pth")
    #     )
    #     log_stats = {
    #         **{f"train_{k}": v for k, v in train_stats.items()},
    #         "epoch": epoch,
    #     }
    #     if utils.is_main_process():
    #         with (Path(args.output_dir) / "log.txt").open("a") as f:
    #             f.write(json.dumps(log_stats) + "\n")
    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # logging.info("Training time {}".format(total_time_str))
    
    writer.close()


def save_checkpoint(model, optimizer, epoch, config, vit_loss, fp16_scaler, val_loss_avg, filename):
     # ============ writing logs ... ============
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
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

def get_positives(anchor, labels_dicts):
    # given an anchor, we define positive as:
    # the same marker, batch, cell line, cond
    # different rep

    positive_marker = anchor['marker']
    positive_batch = anchor['batch']
    positive_cell_line_cond = anchor['cell_line_cond']
    # positive_rep = anchor['rep']
    # positive_site = anchor['site']
    # positives = [i for i, lbl in enumerate(labels_dicts) if lbl['marker'] == positive_marker and lbl['batch'] == positive_batch and \
    #              lbl['cell_line_cond'] == positive_cell_line_cond and lbl['rep'] == positive_rep and lbl['site'] == positive_site \
    #                 and lbl['index'] != anchor['index']]

    positives = [i for i, lbl in enumerate(labels_dicts) if lbl['marker'] == positive_marker \
                and lbl['batch'] == positive_batch \
                and lbl['cell_line_cond'] == positive_cell_line_cond \
                and lbl['index'] != anchor['index']]

    # positives = [i for i, lbl in enumerate(labels_dicts) if lbl['marker'] == positive_marker \
    #         and lbl['cell_line_cond'] == positive_cell_line_cond \
    #         and lbl['index'] != anchor['index']]

    # positives = [i for i, lbl in enumerate(labels_dicts) if lbl['marker'] == positive_marker and lbl['batch'] == positive_batch and \
    #              lbl['cell_line_cond'] == positive_cell_line_cond and lbl['rep'] != anchor['rep']]

    return positives
    
def get_negatives(anchor, labels_dicts):
    # given an anchor, we define negative as:
    # the same marker, batch
    # different cell line, cond
    
    negative_marker = anchor['marker']
    negative_batch = anchor['batch']
    negatives = [i for i, lbl in enumerate(labels_dicts) if lbl['marker'] == negative_marker and \
                 lbl['batch'] == negative_batch and lbl['cell_line_cond'] != anchor['cell_line_cond']]
    # negatives = [i for i, lbl in enumerate(labels_dicts) if lbl['marker'] == negative_marker and lbl['cell_line_cond'] != anchor['cell_line_cond']]

    return negatives

def pair_labels(paths, negative_count=5):
    """
    this function gets all the paths (=labels) in the batch, 
    and pairs all the anochr-positive-negatives it can.

    output:
        anchor_idx(list) : indices of labels that can be used as anchors
        positive_idx(list) : indices of labels that can be used as positive to the anchors in the same location
        negative_idx(list of lists) : lists of indices of labels that can be used as negatives to the anchors in the same location

        for example:
        anchor_idx = [1]
        positive_idx = [10]
        negative_idx = [[3,6,8,12,20]]

        for the embeddings in index 1, the embeddings in index 10 can be used as positive. 
        And, the embeddings in indices [3,6,8,12,20] can be used as negatives.
    """
    labels_list = [p.split(os.sep)[-5:] for p in paths]
    labels_dicts = [{'batch': l[0], 
                    'cell_line_cond': '_'.join(l[1:3]),
                    'marker': l[3], 
                    'rep': l[4].split('_')[0],
                    'site': l[4].split('_')[3],
                    'index':i} for i,l in enumerate(labels_list)]
    anchor_idx, positive_idx, negative_idx = [], [] , []
    for index, anchor in enumerate(labels_dicts):

        positives = get_positives(anchor, labels_dicts)
        if len(positives)==0:
            continue
        else:
            positive_random = random.sample(list(positives),1)[0]
        
        negatives = get_negatives(anchor, labels_dicts)
        if len(negatives) < negative_count:
            continue
        else:
            negatives_random = random.sample(list(negatives), negative_count)
            anchor_idx.append(index)
            positive_idx.append(positive_random)
            negative_idx.append(negatives_random)

    assert len(anchor_idx) == len(positive_idx) == len(negative_idx)
    assert len(negative_idx[0])==negative_count

    return anchor_idx, positive_idx, negative_idx
    
def forward_pass(model, res, vit_loss, negative_count):
    
    with torch.cuda.amp.autocast():
        images = res['image'].to(torch.float).cuda()
        paths = res['image_path']
        logging.info(f"images shape: {images.shape}, paths shape: {paths.shape}")

        # first we need to try and pair for each image (anchor), a positive and $negative_count negatives
        anchor_idx, positive_idx, negative_idx = pair_labels(paths, negative_count) 
        logging.info(f'[InfoNCE] found {len(anchor_idx)}/{images.shape[0]} anchors]')

        all_idx = np.unique(anchor_idx + list(np.unique(flat_list_of_lists(negative_idx))) + positive_idx)
        
        # now we want to create embeddings only for the images that can be used as anchor/positive/negative
        embeddings = model(images[all_idx])
        
        # because we took only the images that can be used as anchor/positive/negative, now the original indices are not true anymore and we need to convert them
        sorter = np.argsort(all_idx)
        anchor_idx = sorter[np.searchsorted(all_idx, anchor_idx, sorter=sorter)]
        positive_idx = sorter[np.searchsorted(all_idx, positive_idx, sorter=sorter)]
        negative_idx = sorter[np.searchsorted(all_idx, negative_idx, sorter=sorter)]    

        loss = vit_loss(embeddings, anchor_idx, positive_idx, negative_idx)

    
    return loss

def infer_pass(model, data_loader, return_cls_token=False):
    def path_to_labels(path):
        split = path.split(os.sep)
        label = '_'.join(split[-5:-2]+[split[-1].split('_')[0]]+[split[-2]])
        return label

    all_predictions = []
    all_labels = []
    all_full_labels = []

    if return_cls_token:
        all_outputs = []
    
    model.eval()
    with torch.no_grad():
        for it, res in enumerate(data_loader):
            logging.info(f"batch number: {it}/{len(data_loader)}")
            # with torch.cuda.amp.autocast():
            images, targets, paths = res['image'].to(torch.float).cuda(), res['label'].cuda(), res['image_path']
            preds, outputs = model(images, return_hidden=True)
            all_predictions.extend(preds.cpu())
            all_labels.extend(targets.cpu())
            
            if return_cls_token:
                all_outputs.append(outputs.cpu())

            labels = [path_to_labels(path) for path in paths]
            all_full_labels.extend(labels)

    if return_cls_token:
        return np.asarray(all_predictions), np.asarray(all_labels), np.vstack(all_outputs), np.asarray(all_full_labels)

    return np.asarray(all_predictions), np.asarray(all_labels), None, np.asarray(all_full_labels)

                
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
    model.train()
    running_loss = 0.0
    loss_train_avg = 0
    for it, res in enumerate(data_loader):
        logging.info(f"[Training epoch={epoch}] batch number: {it}/{len(data_loader)}")
        
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        loss = forward_pass(model, res, vit_loss, args.negative_count)
        loss = loss / args.accumulation_steps  # Normalize the loss
        running_loss += loss.item()

        if not math.isfinite(running_loss):
            logging.info("Loss is {}, stopping training".format(running_loss))
            raise Exception(f"Loss is {running_loss}, stopping training")
        
        fp16_scaler.scale(loss).backward()
        
        if (it + 1) % args.accumulation_steps == 0:
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
            running_loss = 0.0
    
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
            loss = forward_pass(model, res, vit_loss, args.negative_count)
            logging.info(f"***** [epoch {epoch}] val loss: {loss.item()} *****")
            
            writer.add_scalar('1. Loss/val', loss.item(), epoch * len(data_loader_val) + it)
            loss_val_avg += loss.item()
            
        loss_val_avg /= len(data_loader_val)
        logging.info(f"***** [epoch {epoch}] AVG val loss: {loss_val_avg} *****")
        writer.add_scalar('1. Loss/val_avg', loss_val_avg, epoch )
        writer.add_scalars('Loss', {'Training': loss_train_avg, 'Validation': loss_val_avg}, epoch)

        
    
    return loss_val_avg
        
     
class DataAugmentationVIT(object):
    def __init__(self, config):
        self.config = config

        # Define transformations for global and local views
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

     

    def __call__(self, image):
        img = self.transform(image)
        return img
    
class VITContrastiveLoss(nn.Module):
    def __init__(self, negative_count=5, device='cuda'):
        super().__init__()
        self.device = device
        self.negative_count = negative_count
        self.loss = InfoNCE(negative_mode = 'paired')

    def forward(self, embeddings, anchor_idx, positive_idx, negative_idx):
        device = embeddings.device
        
        embeddings_size = embeddings.shape[1]
        
        query = embeddings[anchor_idx]
        positive = embeddings[positive_idx]
        negative = embeddings[torch.as_tensor(negative_idx)]

        assert query.shape[1] == positive.shape[1] == negative.shape[2] == embeddings_size
        assert query.shape[0] == positive.shape[0] == negative.shape[0]
        assert negative.shape[1] == self.negative_count

        
        return self.loss(query, positive, negative)