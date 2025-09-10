import os
import sys
import cv2
from PIL import Image
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import logging
from src.models.architectures.NOVA_model import NOVAModel
from src.embeddings.embeddings_utils import load_embeddings
from src.common.utils import load_config_file
from src.datasets.dataset_config import DatasetConfig
from src.models.utils.consts import CHECKPOINT_BEST_FILENAME, CHECKPOINTS_FOLDERNAME
from typing import Dict, List, Optional, Tuple, Callable
from copy import deepcopy
import numpy as np
import torch
from src.common.utils import get_if_exists
from src.datasets.data_loader import get_dataloader
from src.datasets.dataset_NOVA import DatasetNOVA
from src.datasets.label_utils import get_batches_from_labels, get_unique_parts_from_labels, get_markers_from_labels,\
    edit_labels_by_config, get_batches_from_input_folders, get_reps_from_labels, get_conditions_from_labels, get_cell_lines_from_labels
from torch.utils.data import DataLoader
from NOVA_rotation.Configs.attn_config import AttnConfig #TODO: CHANGE TO NOVA



REDUCE_HEAD_FUNC_MAP = {
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
}


def generate_attn_maps(model:NOVAModel, config_data:DatasetConfig, 
                        batch_size:int=700, num_workers:int=6)->Tuple[List[np.ndarray[torch.Tensor]],
                                                                      List[np.ndarray[str]]]:
    logging.info(f"[generate_attn_maps] Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"[generate_attn_maps] Num GPUs Available: {torch.cuda.device_count()}")

    all_attn_maps, all_labels, all_paths = [], [], []

    train_paths:np.ndarray[str] = model.trainset_paths
    val_paths:np.ndarray[str] = model.valset_paths
    
    full_dataset = DatasetNOVA(config_data)
    full_paths = full_dataset.get_X_paths()
    full_labels = full_dataset.get_y()
    logging.info(f'[generate_attn_maps]: total files in dataset: {full_paths.shape[0]}')
    for set_paths, set_type in zip([train_paths, val_paths, None],
                                   ['trainset','valset','testset']):
        
        if set_type=='testset':
            paths_to_remove = np.concatenate([train_paths, val_paths])
            current_paths = full_dataset.get_X_paths()
            current_labels = full_dataset.get_y()
            indices_to_keep = np.where(~np.isin(current_paths, paths_to_remove))[0]      
            new_set_paths = current_paths[indices_to_keep]
            new_set_labels = current_labels[indices_to_keep]
        
        else:
            indices_to_keep = np.where(np.isin(full_paths, set_paths))[0]
            if indices_to_keep.shape[0]==0:
                continue
            
            new_set_paths = full_paths[indices_to_keep]
            new_set_labels = full_labels[indices_to_keep]


        logging.info(f'[generate_attn_maps]: for set {set_type}, there are {new_set_paths.shape} paths and {new_set_labels.shape} labels')
        new_set_dataset = deepcopy(full_dataset)
        new_set_dataset.set_Xy(new_set_paths, new_set_labels)
        
        attn_maps, labels, paths = __generate_attn_maps_with_dataloader(new_set_dataset, model, batch_size, num_workers)
        
        all_attn_maps.append(attn_maps)
        all_labels.append(labels)
        all_paths.append(paths)

    return all_attn_maps, all_labels, all_paths


def save_attn_maps(attn_maps:List[np.ndarray[torch.Tensor]], 
                    labels:List[np.ndarray[str]], 
                    paths:List[np.ndarray[str]],
                    data_config:DatasetConfig, 
                    output_folder_path:str)->None:

    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[save_attn_maps] unique_batches: {unique_batches}')
    
    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
        
    for i, set_type in enumerate(data_set_types):
        cur_attn_maps, cur_labels, cur_paths = attn_maps[i], labels[i], paths[i]
        batch_of_label = get_batches_from_labels(cur_labels, data_config)
        __dict_temp = {batch: np.where(batch_of_label==batch)[0] for batch in unique_batches}
        for batch, batch_indexes in __dict_temp.items():
            # create folder if needed
            batch_save_path = os.path.join(output_folder_path, data_config.EXPERIMENT_TYPE, batch)
            os.makedirs(batch_save_path, exist_ok=True)
            
            if not data_config.SPLIT_DATA:
                # If we want to save a full batch (without splittint to train/val/test), the name still will be testset.npy.
                # This is why we want to make sure that in this case, we never saved already the train/val/test sets, because this would mean this batch was used as training batch...
                if os.path.exists(os.path.join(batch_save_path,f'trainset_labels.npy')) or os.path.exists(os.path.join(batch_save_path,f'valset_labels.npy')):
                    logging.warning(f"[save_attn_maps] SPLIT_DATA={data_config.SPLIT_DATA} BUT there exists trainset or valset in folder {batch_save_path}!! make sure you don't overwrite the testset!!")
            logging.info(f"[save_attn_maps] Saving {len(batch_indexes)} in {batch_save_path}")


            np.save(os.path.join(batch_save_path,f'{set_type}_labels.npy'), np.array(cur_labels[batch_indexes]))
            np.save(os.path.join(batch_save_path,f'{set_type}.npy'), cur_attn_maps[batch_indexes])
            np.save(os.path.join(batch_save_path,f'{set_type}_paths.npy'), cur_paths[batch_indexes])

            logging.info(f'[save_attn_maps] Finished {set_type} set, saved in {batch_save_path}')



def __generate_attn_maps_with_dataloader(dataset:DatasetNOVA, model:NOVAModel, batch_size:int=700, 
                                          num_workers:int=6)->Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]:
    data_loader = get_dataloader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    logging.info(f"[generate_attn_maps_with_dataloader] Data loaded: there are {len(dataset)} images.")
    
    attn_maps, labels, paths = model.gen_attn_maps(data_loader) # (num_samples, num_layers, num_heads, num_patches, num_patches)
    logging.info(f'[generate_attn_maps_with_dataloader] total attn_maps: {attn_maps.shape}')
    
    return attn_maps, labels, paths




def process_attn_maps(attn_maps: np.ndarray[float], labels: np.ndarray[str], 
                        data_config: DatasetConfig, config_attn: AttnConfig, num_workers:int = 4):
    """
    Process attention maps.

    Parameters
    ----------
    attn_maps :         np.ndarray of shape (num_samples, num_layers, num_heads, num_patches, num_patches)
                        The attention maps for all samples. Each map shows how patches attend to each other across layers and heads.
    labels :            np.ndarray of shape (num_samples,)
                        Class labels for each sample (used for labeling plots).
    data_config:        DatasetConfig 
    config_attn:        AttnConfig

    algo:
        (1) process attention using attn_method and head_reduction_method
        (2) normalize and apply thershold if specified 
    return:
        processed attention maps 

    """

    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[process_attn_maps] unique_batches: {unique_batches}')

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
    
    all_attn_maps = []

    img_shape = data_config.IMAGE_SIZE # suppose to be square (100, 100)
    for i, set_type in enumerate(data_set_types):
        cur_attn_maps, cur_labels = attn_maps[i], labels[i]
        batch_of_label = get_batches_from_labels(cur_labels, data_config)
        __dict_temp = {batch: np.where(batch_of_label==batch)[0] for batch in unique_batches}

        logging.info(f'[process_attn_maps]: for set {set_type}, starting proceesing {len(cur_labels)} samples.')
        
        set_attn_maps = []
        for batch, batch_indexes in __dict_temp.items():
            batch_attn_maps = cur_attn_maps[batch_indexes]

            fn = partial(_process_single_attn_sample, config_attn=config_attn, img_shape=img_shape)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                set_attn_maps.extend(executor.map(fn, batch_attn_maps))
        # end of set type
        set_attn_maps = np.stack(set_attn_maps)
        all_attn_maps.append(set_attn_maps)

    # end of samples
    return all_attn_maps

def _process_single_attn_sample(sample_attn, config_attn, img_shape):
    processed_attn_map = globals()[f"_process_attn_map_{config_attn.ATTN_METHOD}"](sample_attn, config_attn)
    num_patches = processed_attn_map.shape[-1]
    patch_dim = int(np.sqrt(num_patches))
    processed_attn_map = __resize_attn_map(processed_attn_map, patch_dim, img_shape, resample_method=config_attn.RESAMPLE_METHOD)
    return processed_attn_map

def _process_attn_map_rollout(attn:np.ndarray[float], config_attn:AttnConfig):
    # Rollout attention workflow : 
    # head and layer reduction using rollout processing
    # attention map processing
    attn = __attn_map_rollout(attn, attn_layer_dim=0, heads_reduce_fn=REDUCE_HEAD_FUNC_MAP[config_attn.REDUCE_HEAD_FUNC])
    processed_attn_map = __process_attn_map(attn, min_attn_threshold=config_attn.MIN_ATTN_THRESHOLD)
    return processed_attn_map

def _process_attn_map_all_layers(attn:np.ndarray[float], config_attn:AttnConfig):
    # Basic attention workflow
    # head reduction and processing for each layer seperatly 
    attn = __attn_map_all_layers(attn, attn_layer_dim=0, heads_reduce_fn=REDUCE_HEAD_FUNC_MAP[config_attn.REDUCE_HEAD_FUNC])
    num_layers, _, _= attn.shape #(num_layers, num_patches, num_patches)
    attn_maps_all_layers = []

    for layer_idx in range(num_layers):
        layer_attn = attn[layer_idx]
        processed_layer_attn_map = __process_attn_map(layer_attn, min_attn_threshold=config_attn.MIN_ATTN_THRESHOLD)
        attn_maps_all_layers.append(processed_layer_attn_map)
       
    attn_maps_all_layers = np.stack(attn_maps_all_layers)
    return attn_maps_all_layers



def __process_attn_map(attn, min_attn_threshold=None):
        """
        Process the attention from the attention matrix:
            (1) Extract CLS token attention
            (2) Normalize
            (3) Apply threshold
            (4) Scale to [0,1]

        Parameters:
            attn: attention matrix of shape (num_patches, num_patches)
            min_attn_threshold [optional]: minimum value threshold

        Returns:
            processed_attn: float32 attention vector of shape (num_patches - 1,), scaled to [0,1]
        """
        cls_attn = attn[0, 1:]  # shape: (num_patches - 1,)

        # Normalize
        attn_min = cls_attn.min()
        attn_max = cls_attn.max()
        processed_attn = (cls_attn - attn_min) / (attn_max - attn_min + 1e-6)

        # Apply optional threshold
        if min_attn_threshold is not None:
            processed_attn[processed_attn < min_attn_threshold] = 0.0

        
        return processed_attn.astype(np.float32)

def __resize_attn_map(processed_attn, patch_dim, img_shape, resample_method=Image.BICUBIC):
    """
    Resize attention map(s) to the original image shape (H, W).

    Supports:
    - A single attention map: shape (patch_dim * patch_dim,)
    - Multiple attention maps: shape (num_layers, patch_dim * patch_dim)

    Parameters:
        processed_attn: np.ndarray, shape (num_patches,) or (num_layers, num_patches)
        patch_dim: int, e.g., 14 for 14x14 patch grid
        img_shape: tuple (H, W)
        resample_method: PIL.Image resampling method

    Returns:
        np.ndarray of shape (H, W) or (num_layers, H, W), scaled to [0, 1]
    """
    H, W = img_shape

    def resize_single(attn_1d):
        attn_2d = attn_1d.reshape(patch_dim, patch_dim)
        attn_img = Image.fromarray((attn_2d * 255).astype(np.uint8))
        attn_resized = attn_img.resize((W, H), resample=resample_method)
        return np.array(attn_resized).astype(np.float32) / 255.0

    if processed_attn.ndim == 1:
        return resize_single(processed_attn)  # (H, W)
    
    elif processed_attn.ndim == 2:
        return np.stack([resize_single(layer) for layer in processed_attn])  # (num_layers, H, W)

    else:
        raise ValueError(f"Unsupported shape {processed_attn.shape}. Expected (N,) or (L, N).")



def __color_heatmap_attn_map(processed_attn, patch_dim, img_shape, heatmap_color=cv2.COLORMAP_JET, resample_method=Image.BICUBIC):
    """
    Create attention heatmap:
        (1) Reshape to (patch_dim, patch_dim)
        (2) Resize to image shape
        (3) Apply colormap

    Parameters:
        processed_attn: float32 vector of shape (patch_dim * patch_dim,), scaled to [0,1]
        patch_dim: dimension of square patch grid
        img_shape: (H, W) of the original image
        heatmap_color: OpenCV colormap type (default: cv2.COLORMAP_JET)

    Returns:
        heatmap_colored: uint8 colored attention heatmap of shape (H, W, 3)
    """
    attn_square = processed_attn.reshape(patch_dim, patch_dim)
    heatmap_uint8 = (attn_square * 255).astype(np.uint8)
    heatmap_resized = Image.fromarray(heatmap_uint8).resize(img_shape, resample=resample_method)
    heatmap_resized = np.array(heatmap_resized).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, heatmap_color)
    return heatmap_colored


def __attn_map_all_layers(attn, attn_layer_dim=0, heads_reduce_fn:callable=np.mean):
    """ reduce attention across heads according to heads_reduce_fn, for each layer.

    parameteres:
        attn: attention values of shape: ([<num_samples>], num_layers, num_heads, num_patches, num_patches)
        attn_layer_dim: the dimension of the attention layer to iterate the rollout through
                        ** for one sample should be 0 (as it the first dim)
                        ** for multiple samples should be 1 (as num_samples is the 0 dimension)
        heads_reduce_fn: numpy function to reduce the heads layer with (for example: np.mean/np.max/np.min...)

    return:
        reduced_attn: attention map per layer: (num_layers, num_patches, num_patches)
    """
    reduced_attn = heads_reduce_fn(attn, axis=(attn_layer_dim + 1))
    return reduced_attn

def __attn_map_rollout(attn, attn_layer_dim:int=0, heads_reduce_fn:callable=np.mean, start_layer_index:int=0):
    """  aggregates attention maps across multiple layers, using the rollout method:

    parameteres:
        attn: attention values of shape: ([<num_samples>], num_layers, num_heads, num_patches, num_patches)
        attn_layer_dim: the dimension of the attention layer to iterate the rollout through
                        ** for one sample should be 0 (as it the first dim)
                        ** for multiple samples should be 1 (as num_samples is the 0 dimension)
        heads_reduce_fn: numpy function to reduce the heads layer with (for example: np.mean/np.max/np.min...)
        start_layer_index: the index of the layer to start the rollput from.

    returns:
        rollout: attention map for all layers and heads: (num_patches, num_patches)
    """

    # Initialize rollout with identity matrix
    rollout = np.eye(attn.shape[-1]) #(num_patches, num_patches)

    attn = heads_reduce_fn(attn, axis=(attn_layer_dim + 1)) # Average attention across heads (A)

    # Multiply attention maps layer by layer
    for layer_idx in range(start_layer_index,attn.shape[attn_layer_dim]):
        # extract the layer data
        if attn_layer_dim == 0:
            layer_attn = attn[layer_idx]        # layers are in the first dimension
        elif attn_layer_dim == 1:
            layer_attn = attn[:, layer_idx]     # layers are in the second dimension (after batch)
        else:  
            idx = [slice(None)] * attn.ndim
            idx[attn_layer_dim] = layer_idx
            layer_attn = attn[tuple(idx)]

        # rollout mechanism 
        layer_attn += np.eye(layer_attn.shape[-1]) # A + I
        layer_attn /= layer_attn.sum(axis=-1, keepdims=True) # Normalizing A
        rollout = rollout @ layer_attn  # Matrix multiplication
    
    return rollout
