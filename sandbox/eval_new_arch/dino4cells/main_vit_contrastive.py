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


from torch.utils.tensorboard import SummaryWriter
import logging

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score

os.environ['MOMAPS_HOME'] = '/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps'
os.environ['MOMAPS_DATA_HOME'] = '/home/labs/hornsteinlab/Collaboration/MOmaps/input'

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

from src.common.lib.data_loader import get_dataloader
from src.common.lib.dataset import Dataset
from src.common.lib.utils import init_logging, load_config_file
from src.datasets.dataset_spd import DatasetSPD
# from src.common.lib import image_metrics

from sandbox.eval_new_arch.dino4cells.utils import utils
# from functools import partial
# from archs import xresnet as cell_models
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
            # img = transforms.functional.adjust_brightness(img, factor)
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
    
    # tensorboard_root_folder = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/eval_new_arch/dino4cells/tensorboard"
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

    #### params
    
    # config = {
    #     'seed': 1,
    #     'embedding': {
    #         'image_size': 100
    #     },
    #     'patch_size': 14,
    #     'num_channels': 2,
    #     'out_dim': 225,
    #     'use_bn_in_head': False,
    #     'norm_last_layer': True,
        
    #     'local_crops_number': 8,
    #     'warmup_teacher_temp': 0.04,
    #     'teacher_temp': 0.04,
    #     'warmup_teacher_temp_epochs': 0,
    #     'epochs': 5,
    #     'student_temp': 0.1,
    #     'center_momentum': 0.9,
    #     'momentum_teacher': 0.996,
        
    #     'lr': 0.0005,
    #     'min_lr': 1e-6,
    #     'warmup_epochs': 10,
        
    #     'weight_decay': 0.04,
    #     'weight_decay_end': 0.4,
        
        
    #     'batch_size_per_gpu': 4,
    #     'num_workers': 1
    # }
    
    
    #######
    
    
    
    # utils.init_distributed_mode(args)
    utils.fix_random_seeds(config.seed)
    # init_signal_handler()
    # logging.info("git:\n  {}\n".format(utils.get_sha()))
    # logging.info(
    #     "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    # )
    # cudnn.benchmark = True
    cudnn.benchmark = False
    random.seed(config.seed)

    # chosen_loader = file_dataset.image_modes[config["model"]["image_mode"]]
    # FileList = file_dataset.data_loaders[config["model"]["datatype"]]

    # ============ preparing data ... ============
    transform =  DataAugmentationVIT(config=config)
    # transform = transforms.Compose([
    #         # RandomResizedCropWithCheck(100, scale=(0.6, 1.0)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomVerticalFlip()])
    
    config_data = load_config_file(config_data_path)

    dataset_train = DatasetSPD(config_data, transform)
    # config.out_dim = len(np.unique(dataset_train.label))
    train_indexes, val_indexes, test_indexes = None, None, None
    # dataset_val, dataset_test = None, None


    # if is_one_config_supplied:
    # dataset_val, dataset_test = deepcopy(dataset_train), deepcopy(dataset_train) # the deepcopy is important. do not change. 
    # dataset_test.flip, dataset_test.rot = False, False
    if config_data.SPLIT_DATA:
        logging.info("Split data...")
        train_indexes, val_indexes, test_indexes = dataset_train.split()
    # else:
    #     dataset_val, dataset_test = DatasetSPD(config_val), DatasetSPD(config_test)


    ################# TAKING ONLY PART OF THE DATA ################
    quick_train_indices = train_indexes[:]
    quick_val_indices = val_indexes[:]
    # logging.info(f"[WARNING!] TAKING ONLY PART OF THE DATA!! train: {len(quick_train_indices)} val: {len(quick_val_indices)}")
    ############################################################
    
    
    
    # quick_val_indices = val_indexes[:100]
    dataset_train_subset = Dataset.get_subset(dataset_train, quick_train_indices)
    dataset_val_subset = Dataset.get_subset(dataset_train, quick_val_indices)
    # dataset = FileList(
    #     args.data_path,
    #     config["model"]["root"],
    #     transform=transform,
    #     loader=chosen_loader,
    #     flist_reader=partial(
    #         file_dataset.pandas_reader_only_file,
    #         sample_single_cells=args.sample_single_cells,
    #     ),
    #     with_labels=False,
    #     balance=False,
    #     sample_single_cells=args.sample_single_cells,
    # )
    # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    
    dataset = dataset_train_subset
    dataset_val = dataset_val_subset
    
    data_loader = get_dataloader(dataset, config.batch_size_per_gpu, num_workers=config.num_workers)
    data_loader_val = get_dataloader(dataset_val, config.batch_size_per_gpu, num_workers=config.num_workers)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     # sampler=sampler,
    #     batch_size=config.batch_size_per_gpu,
    #     num_workers=config.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    logging.info(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    model = vits.vit_base(
            img_size=[config.embedding.image_size, config.embedding.image_size],
            patch_size=config.patch_size,
            # drop_path_rate=0.1,  # stochastic depth
            # drop_rate=0.3, # can't go together with drop_path_rate - cuda out of memory
            in_chans=config.num_channels*len(dataset.unique_markers),
            num_classes=256#len(dataset.unique_markers)
    )
    # teacher = vits.vit_base(
    #     img_size=[config.embedding.image_size],
    #     patch_size=config.patch_size,
    #     in_chans=config.num_channels,
    # )
    # embed_dim = student.embed_dim

    # multi-crop wrapper handles forward with inputs of different resolutions
    # student = utils.MultiCropWrapper(
    #     student,
    #     DINOHead(
    #         embed_dim,
    #         config.out_dim,
    #         use_bn=config.use_bn_in_head,
    #         norm_last_layer=config.norm_last_layer,
            
    #         # nlayers=2, #SAGY
    #         # bottleneck_dim=1000,#SAGY
    #         # hidden_dim=1000 #SAGY
    #     ),
    # )
    # teacher = utils.MultiCropWrapper(
    #     teacher,
    #     DINOHead(embed_dim,
    #              config.out_dim,
    #              config.use_bn_in_head,
                 
    #             #  nlayers=2, #SAGY
    #             #  bottleneck_dim=1000,#SAGY
    #             #  hidden_dim=1000 #SAGY),
    #     )
    # )

    # move networks to gpu
    # student, teacher = student.cuda(), teacher.cuda()
    model = model.cuda()
    # student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    # teacher.load_state_dict(student.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    # for p in teacher.parameters():
    #     p.requires_grad = False

    # ============ preparing loss ... ============
    # dino_loss = DINOLoss(
    #     config.out_dim,
    #     config.local_crops_number
    #     + 2,  # total number of crops = 2 global crops + local_crops_number
    #     config.warmup_teacher_temp,
    #     config.teacher_temp,
    #     config.warmup_teacher_temp_epochs,
    #     config.epochs,
    #     config.student_temp,
    #     config.center_momentum,
    # ).cuda()
    
    vit_loss = VITContrastiveLoss(config.local_crops_number).cuda()#nn.CrossEntropyLoss(label_smoothing=0.2).cuda()
    # vit_loss = nn.CrossEntropyLoss().cuda()


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
    # momentum parameter is increased to 1. during training with a cosine schedule
    # momentum_schedule = utils.cosine_scheduler(
    #     config.momentum_teacher, 1, config.epochs, len(data_loader)
    # )
    logging.info("Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    # utils.restart_from_checkpoint(
    #     os.path.join(args.output_dir, "checkpoint.pth"),
    #     run_variables=to_restore,
    #     student=student,
    #     teacher=teacher,
    #     optimizer=optimizer,
    #     fp16_scaler=fp16_scaler,
    #     dino_loss=dino_loss,
    # )
    start_epoch = to_restore["epoch"]
    
    last_best_val_loss = np.inf
    early_stopping_patience = config.early_stopping_patience

    # start_time = time.time()
    logging.info("Starting VIT training !")
    for epoch in tqdm(range(start_epoch, config.epochs)):
        # data_loader.sampler.set_epoch(epoch)
        
        optimizer.zero_grad()
        
        # ============ training one epoch of DINO ... ============
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

                # last_checkpoint_epoch = last_checkpoint['epoch']
                # checkpoint_new_path = os.path.join(config.output_dir, f"checkpoint{last_checkpoint_epoch:02}_bckup.pth")
                # logging.info(f"Old checkpoint is better ({last_checkpoint_val_loss_avg} vs {loss_val_avg}). Changing file name from {checkpoint_path} to {checkpoint_new_path}")
                # os.rename(checkpoint_path, checkpoint_new_path)

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
#     if fp16_scaler is not None:
    # save_dict["fp16_scaler"] = fp16_scaler.state_dict()
#     utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
#     if args.saveckp_freq and epoch % args.saveckp_freq == 0:
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
    # images_global, images_local = res['image_global'], res['image_local']

    # move images to gpu
    # images = [im.float().cuda(non_blocking=True) for im in images]
    # images_global = [im.float().cuda(non_blocking=True) for im in images_global]
    # images_local = [im.float().cuda(non_blocking=True) for im in images_local]
    # images_global = torch.stack(images_global)
    # images_local = torch.stack(images_local)
    # images = torch.stack(images)
    # logging.info(f"global: {images_global.shape}, local: {images_local.shape}")
    # images_global = torch.from_numpy(np.vstack(images_global))
    # teacher and student forward passes + compute dino loss
    # with torch.cuda.amp.autocast(fp16_scaler is not None):
    
    with torch.cuda.amp.autocast():
        images_global, images_local = res['image_global'].to(torch.float).cuda(), res['image_local'].to(torch.float).cuda()
        logging.info(f"images_global shape: {images_global.shape}, images_local shape: {images_local.shape}")
        # with torch.no_grad():
        #     teacher_output = teacher(
        #         images_global
        #     )  # only the 2 global views pass through the teacher
        # student_output = student(images_global)
        # student_output = torch.vstack([student_output, student(images_local)]) 
        # loss = vit_loss(model, teacher_output, epoch)
        preds_global = model(images_global)
        preds_local = model(images_local)
        
        # loss = vit_loss(preds_global, preds_local)
        embeddings = torch.cat([preds_global, preds_local], dim=0)
        loss = vit_loss(embeddings)

    
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
            # with torch.cuda.amp.autocast():
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
    # metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    logging.info(header)
    # for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
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

        # loss = forward_pass(model, res, vit_loss, epoch)
        loss = forward_pass(model, res, vit_loss)
        loss = loss / args.accumulation_steps  # Normalize the loss
        running_loss += loss.item()

        if not math.isfinite(running_loss):
            logging.info("Loss is {}, stopping training".format(running_loss))
            raise Exception(f"Loss is {running_loss}, stopping training")
            
        # logging.info(f"***** [epoch {epoch}] train loss: {loss.item()} *****")

        
        fp16_scaler.scale(loss).backward()
        
        if (it + 1) % args.accumulation_steps == 0:
            logging.info(f"***** [epoch {epoch}] train loss: {running_loss} *****")
            
            
            logging.info("Calculating grads")
            
            fp16_scaler.unscale_(optimizer)  # Unscale gradients    
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)  # Clip gradients
            
            # utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            
                
            logging.info("Updating model")
            fp16_scaler.step(optimizer)  # Update weights
            fp16_scaler.update()
            
            # logging.info("Updating teacher..")
            # with torch.no_grad():
            #     m = momentum_schedule[it]  # momentum parameter
            #     for param_q, param_k in zip(
            #         student.parameters(), teacher.parameters()
            #     ):
            #         param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            
            writer.add_scalar('1. Loss/train', running_loss, epoch * len(data_loader) + it)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch * len(data_loader) + it)
            writer.add_scalar('Weight Decay', optimizer.param_groups[0]['weight_decay'], epoch * len(data_loader) + it)
            # writer.add_scalar('Teacher Momentum', momentum_schedule[it], epoch * len(data_loader) + it)

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
            loss = forward_pass(model, res, vit_loss)
                
            logging.info(f"***** [epoch {epoch}] val loss: {loss.item()} *****")
            
            writer.add_scalar('1. Loss/val', loss.item(), epoch * len(data_loader_val) + it)
            loss_val_avg += loss.item()
            
        loss_val_avg /= len(data_loader_val)
        logging.info(f"***** [epoch {epoch}] AVG val loss: {loss_val_avg} *****")
        writer.add_scalar('1. Loss/val_avg', loss_val_avg, epoch )

        
    
    return loss_val_avg
        
    


# class DINOLoss(nn.Module):
#     def __init__(
#         self,
#         out_dim,
#         ncrops,
#         warmup_teacher_temp,
#         teacher_temp,
#         warmup_teacher_temp_epochs,
#         nepochs,
#         student_temp=0.1,
#         center_momentum=0.9,
#     ):
#         super().__init__()
#         self.student_temp = student_temp
#         self.center_momentum = center_momentum
#         self.ncrops = ncrops
#         self.register_buffer("center", torch.zeros(1, out_dim))
#         # we apply a warm up for the teacher temperature because
#         # a too high temperature makes the training instable at the beginning
#         self.teacher_temp_schedule = np.concatenate(
#             (
#                 np.linspace(
#                     warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
#                 ),
#                 np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
#             )
#         )

#     def forward(self, student_output, teacher_output, epoch):
#         """
#         Cross-entropy between softmax outputs of the teacher and student networks.
#         """
#         student_out = student_output / self.student_temp
#         # logging.info(f"ncrops: {self.ncrops}")
#         student_out = student_out.chunk(self.ncrops)
#         # logging.info(f"len student_out: {len(student_out)}")

#         # teacher centering and sharpening
#         temp = self.teacher_temp_schedule[epoch]
#         teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
#         teacher_out = teacher_out.detach().chunk(2)

#         total_loss = 0
#         n_loss_terms = 0
#         for iq, q in enumerate(teacher_out):
#             for v in range(len(student_out)):
#                 if v == iq:
#                     # we skip cases where student and teacher operate on the same view
#                     continue
#                 loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
#                 total_loss += loss.mean()
#                 n_loss_terms += 1
#         total_loss /= n_loss_terms
#         self.update_center(teacher_output)
#         return total_loss

#     @torch.no_grad()
#     def update_center(self, teacher_output):
#         """
#         Update center used for teacher output.
#         """
#         batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
#         # dist.all_reduce(batch_center)
#         batch_center = batch_center / (len(teacher_output))# * dist.get_world_size())

#         # ema update
#         self.center = self.center * self.center_momentum + batch_center * (
#             1 - self.center_momentum
#         )


class self_normalize(object):
    def __call__(self, x):
        min_val = 0
        max_val = 1
        
        for ax in range(x.shape[1]):
            data_min = torch.min(x[:,ax,...])
            data_max = torch.max(x[:,ax,...])
            
            # Compute the scale and min_val for the transformation
            scale = (max_val - min_val) / (data_max - data_min)
            x[:,ax,...] = scale * (x[:,ax,...] - data_min) + min_val

        return x
    
        # Z scale, looks bad, probably because we have a lot of black in the image..
        # m = x.mean((-2, -1), keepdim=True)
        # s = x.std((-2, -1), unbiased=False, keepdim=True)
        # x -= m
        # x /= s + 1e-7
        # return x
        
class RandomResizedCropWithCheck(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=Image.BILINEAR, retries=3, threshold=1e-2):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)
        self.retries = retries
        self.threshold = threshold

    def __call__(self, img):
        self.brenners = []
        self.crops = []
        
        for _ in range(self.retries):
            # Apply the random resized crop transformation
            crop = super().__call__(img)
            # Check if the crop is not empty
            # print(f"crop: {crop.shape}, img: {img.shape}")
            if self.is_non_empty(crop, self.threshold):
                return crop
        # If all retries fail, return the best crop
        # print(f"otherwise: {self.crops[np.argmax(self.vars)].shape} img: {img.shape}")
        return self.crops[np.argmax(self.brenners)]

    def calculate_image_sharpness_brenner(self, image):
        """
        Low = blur
        High = sharp
        """
        def _brenners_gradient(image):
            # Calculate the squared difference
            shift = 2  # Typical distance between pixels for calculation
            diff = image[:, :-shift] - image[:, shift:]
            brenner = np.sum(diff ** 2)
            
            return brenner
        
        rows_brenner = _brenners_gradient(image)
        cols_brenner = _brenners_gradient(image.T)
        
        return rows_brenner + cols_brenner

    def is_non_empty(self, crop, threshold):
        # Convert the crop to numpy array
        crop_array = np.array(crop)
        crop_brenner = self.calculate_image_sharpness_brenner(crop_array[:,0,...])   
        # crop_var = crop_array[:,0,...].var()
        
        # Check if the crop has non-zero pixels
        self.crops.append(crop)
        self.brenners.append(crop_brenner)
        return crop_brenner > threshold
    

class DataAugmentationVIT(object):
    def __init__(self, config):
        self.config = config
        # (
        #     self.global_transfo1,
        #     self.global_transfo2,
        #     self.local_transfo,
        #     _,
        # ) = tfms_from_config(self.config)

        # Define transformations for global and local views
        self.transform = transforms.Compose([
            # RandomResizedCropWithCheck(100, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # RandomIntensityChange(intensity_range=(0.8, 1.2), p=0.1),
            # RandomChannelShutdown(channel_index=1, p=0.1),
            # self_normalize()
            #Warp
            #Remove channel (set channel to zeros)
            #rescale_protein (rescale intensity - img*=random_factor)
            # Change brightness
            # Change contrast
            
        ])

        self.local_transform = transforms.Compose([
            RandomResizedCropWithCheck(100, scale=(0.1, 0.25),threshold=5, retries=10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomIntensityChange(intensity_range=(0.8, 1.2), p=0.3),
            # RandomChannelShutdown(channel_index=1, p=0.3),
            self_normalize()
            # Warp
            # remove channel (set channel to zeros)
            #rescale protein (rescale intensity - img*=random_factor)
            
            # Change brightness
            # Change contrast
            
            
        ])
     

    def __call__(self, image):
        # return self.transform(image)
        
        # # global_crops = []
        # local_crops = []
        
        # # global_crops.append(self.global_transform(image))
        # # global_crops.append(self.global_transform(image))
    
        # for _ in range(self.config.local_crops_number):
        #     local_crops.append(self.local_transform(image))
            
            
        # # fig, ax = plt.subplots(1,2)
        # # ax[0].imshow(global_crops[0][0,0,...])
        # # ax[1].imshow(local_crops[0][0,0,...])
        # # plt.show()
            
        # global_crops = np.vstack(global_crops)
        # local_crops = np.vstack(local_crops)
        local_image = self.local_transform(image)
        global_image = self.transform(image)
        
        return global_image, local_image
        
        # global_images = np.tile(global_image, (self.config.local_crops_number, 1, 1, 1))#np.repeat(global_image, self.config.local_crops_number, axis=0)
        
        
        
        # local_crops = self.local_transform(image)
        # local_crops = np.repeat(global_image, self.config.local_crops_number, axis=0)
        
        # return global_images, local_crops

class VITContrastiveLoss(nn.Module):
    def __init__(self, n_views, temperature=0.07, device='cuda'):
        super().__init__()
        self.n_views = n_views
        self.temperature = temperature
        self.device = device
        # self.kl = torch.nn.KLDivLoss()
        # self.ce = nn.CrossEntropyLoss(label_smoothing=0.2).cuda()
        
    # def forward(self, outputs_global, outputs_local):
    #     loss = F.kl_div(F.log_softmax(outputs_global, dim=-1), F.softmax(outputs_local, dim=-1), reduction='batchmean')
    #     # loss = torch.sum(-F.softmax(outputs_global) * F.log_softmax(outputs_local, dim=-1), dim=-1)
    #     # loss = loss.mean()
    #     # loss =self.ce(outputs_global, outputs_local)
    #     return loss

    # def forward(self, out_1, out_2, temperature=0.5):
    #     batch_size = out_1.shape[0]
    #     out = torch.cat([out_1, out_2], dim=0)  # Concatenate along the batch dimension
    #     similarity_matrix = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=2)

    #     # Create a mask to remove the diagonal elements
    #     mask = torch.eye(out.shape[0], device=similarity_matrix.device).bool()
    #     negative_mask = ~mask
    #     similarity_matrix = similarity_matrix[negative_mask].view(similarity_matrix.shape[0], -1)

    #     # Apply the mask to remove diagonal elements
    #     negatives = similarity_matrix.masked_select(negative_mask).view(batch_size * 2, -1)

    #     # Extract the positive pairs (diagonal elements before masking)
    #     positives = torch.cat([similarity_matrix[i, batch_size + i].unsqueeze(0) for i in range(batch_size)] +
    #                             [similarity_matrix[batch_size + i, i].unsqueeze(0) for i in range(batch_size)])

    #     # Combine positives and negatives for logits
    #     logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)

    #     # Labels: 0 for the positive pairs
    #     labels = torch.zeros(batch_size * 2, device=similarity_matrix.device, dtype=torch.long)

    #     return F.cross_entropy(logits / temperature, labels)

    def forward(self, embeddings):
        device = embeddings.device
        
        batch_size = embeddings.shape[0] // 2
        
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)
        
        embeddings = F.normalize(embeddings, dim=1)
        
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
       

        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
      
        
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
       

        logits = torch.cat([positives, negatives], dim=1)

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        
        loss = F.cross_entropy(logits / self.temperature, labels)
        
        return loss
        
        # Compute cosine similarity
        # similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        
        # Create a mask to remove the diagonal elements
        # mask = torch.eye(batch_size * 2, device=similarity_matrix.device).bool()
        
        # # Extract positive pairs (i, batch_size + i)
        # positives = similarity_matrix[mask].view(batch_size, -1)
        
        # # Get negatives
        # # Assign -inf to diagonal elements
        # negatives = similarity_matrix.masked_fill(mask, float('-inf')).view(batch_size * 2, -1)
        # # negatives = similarity_matrix[~mask].view(batch_size * 2, -1)
        
        # print(f"pos shape: {positives.shape}; neg shape: {negatives.shape}")
        
        # # Combine positives and negatives for logits
        # logits = torch.cat([positives, negatives], dim=1)
        
        # # Labels: 0 for the positive pairs
        # labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        
        # return F.cross_entropy(logits / self.temperature, labels)

    # def forward(self, out_1, out_2):
    #     batch_size = out_1.shape[0]
    #     features = torch.cat([out_1, out_2], dim=0)  # Concatenate along the batch dimension
        
    #     # labels = torch.arange(batch_size)
    #     labels = torch.cat([torch.arange(batch_size) for i in range(self.n_views)], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #     labels = labels.to(self.device)

    #     features = F.normalize(features, dim=1)

    #     similarity_matrix = torch.matmul(features, features.T)
    #     # assert similarity_matrix.shape == (
    #     #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    #     # assert similarity_matrix.shape == labels.shape

    #     # discard the main diagonal from both: labels and similarities matrix
    #     mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    #     # assert similarity_matrix.shape == labels.shape

    #     # select and combine multiple positives
    #     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    #     # select only the negatives the negatives
    #     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    #     logits = torch.cat([positives, negatives], dim=1)
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

    #     logits = logits / self.temperature
        
    #     loss = F.cross_entropy(logits, labels)
        
    #     return loss
        # return logits, labels

# if __name__ == "__main__":
    # parser = argparse.ArgumentParser("DINO", parents=[get_args_parser()])
    # args = parser.parse_args()
    # config = yaml.safe_load(open(args.config, "r"))
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # train_dino(args, config)