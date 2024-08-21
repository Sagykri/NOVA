
import logging
from typing import Dict
import numpy as np
import torch
import torch
import os

class EarlyStoppingInfo():
    """Holds information for handeling the early stopping
    """
    def __init__(self):
        self.counter: int = None
        self.best_loss: float = np.inf
        
class CheckpointInfo():
    def __init__(self,
                 model_dict:Dict,
                 optimizier_dict: Dict,
                 epoch:int,
                 training_config:str,
                 dataset_config:str,
                 model_config:str,
                 scaler_dict:Dict,
                 loss_val_avg:float,
                 early_stopping_counter:int,
                 description: str):
        
        self.model_dict: Dict = model_dict
        self.optimizier_dict: Dict = optimizier_dict
        self.epoch = epoch
        self.training_config = training_config
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.scaler_dict = scaler_dict
        self.loss_val_avg = loss_val_avg
        self.early_stopping_counter = early_stopping_counter
        self.description = description


def load_checkpoint_from_file(ckp_path):
    checkpoint = torch.load(ckp_path, map_location='cuda' if torch.cuda.is_available() else "cpu")
    return checkpoint

def get_params_groups(*models):
    regularized = []
    not_regularized = []
    for model in models:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]

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

###############################################################


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
"""
Misc functions.
Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

def fix_random_seeds(seed=1):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
