
import os
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))
import logging
from src.datasets.dataset_config import DatasetConfig
from typing import Dict, List, Optional, Tuple, Callable
from copy import deepcopy
import numpy as np
import torch
from src.datasets.label_utils import get_batches_from_labels, get_unique_parts_from_labels, get_markers_from_labels,\
    edit_labels_by_config, get_batches_from_input_folders, get_reps_from_labels, get_conditions_from_labels, get_cell_lines_from_labels
from collections import OrderedDict
from tools.load_data_from_npy import parse_paths, load_tile, load_paths_from_npy, parse_path_item
from skimage import filters
from src.analysis.attention_scores_config import AttnScoresBaseConfig

def threshold_percentile(arr, percentile):
        # top X% of the array
        return np.quantile(arr, percentile)
# Map threshold method names to their functions
THRESHOLD_METHODS = {
        "percentile": threshold_percentile,  
        "otsu": filters.threshold_otsu,
        "isodata": filters.threshold_isodata,
        "yen": filters.threshold_yen,
        "li": filters.threshold_li,
        "triangle": filters.threshold_triangle,
        "mean": filters.threshold_mean,
        "niblack": filters.threshold_niblack,
        "sauvola": filters.threshold_sauvola,
    }

def corr_ssim(m1, m2, config  = None):
    """
    assumes m1 and m2 are normalized.
    Calculates structural similarity index measure between the 2 matrices.
    """
    from skimage.metrics import structural_similarity as ssim
    args = getattr(config, 'CORR_METHOD_ARGS', None)
    full = args.get('full', True) if args else True
    score, ssim_map = ssim(m1, m2, full=full)

    return score

def corr_attn_overlap(m1, m2, config = None):
    """
        m1: attn
        m2: channel
        for attention maps:
            sums the values of attention (m1) only in the masked area of the input (m2).
                --> "segment" to get only the most important pixels of the input images and calculate
                    the average attention value in those areas.  
    """
    args = getattr(config, 'CORR_METHOD_ARGS', None)
    m2_binary_perc = args.get('m2_binary_perc', 0.75) if args else 0.75
    # Use top X% of m2 (img) as binary mask
    threshold = np.quantile(m2, m2_binary_perc)
    m2_mask = m2 >= threshold
    if m2_mask.sum() == 0:
        return 0.0
    score = (m1[m2_mask].sum()) / m2_mask.sum() #normalize by the mask size
    return score


def corr_soft_overlap(attn_map, img_ch, config = None):

    # element-wise multiplication (Hadamard product) - "soft overlap"
    overlap = np.sum(attn_map * img_ch) 

    # Normalizations
    total_attn = np.sum(attn_map)
    total_marker = np.sum(img_ch)

    # Avoid division by zero
    precision_like = overlap / total_attn if total_attn > 0 else 0
    recall_like = overlap / total_marker if total_marker > 0 else 0

    # Harmonic mean (F1-like)
    if precision_like + recall_like > 0:
        f1_like = 2 * precision_like * recall_like / (precision_like + recall_like)
    else:
        f1_like = 0
    
    return precision_like, recall_like, f1_like


def binary_mask(arr, config):
        # Check if the requested method exists in the map
        method = config.THRESHOLD_METHOD
        if method not in THRESHOLD_METHODS:
            raise ValueError(f"No threshold function found for {method}")
            
        func = THRESHOLD_METHODS[method]
        args = config.THRESHOLD_ARGS

        try:
            threshold = func(arr, **args)
        except TypeError as e:
            raise TypeError(f"The threshold function '{method}' received invalid arguments: {e}")

        return arr >= threshold

def corr_binary_score(attn_map , img_ch, config):

    attn_mask = binary_mask(attn_map, config)
    ch_mask = binary_mask(img_ch, config)

    tp = np.logical_and(attn_mask, ch_mask).sum()  # True Positives
    fp = np.logical_and(attn_mask, np.logical_not(ch_mask)).sum()  # False Positives
    tn = np.logical_and(np.logical_not(attn_mask), np.logical_not(ch_mask)).sum()  # True Negatives
    fn = np.logical_and(np.logical_not(attn_mask), ch_mask).sum()  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def normalize(v1):
    denom = v1.max() - v1.min()
    if denom == 0:
        return np.zeros_like(v1)
    else:
        return (v1 - v1.min()) / denom

def compute_correlation(attn, img_ch, corr_config:AttnScoresBaseConfig):
    assert attn.shape == img_ch.shape, f"[compute_correlation] Shape mismatch: attn.shape={attn.shape}, img_ch.shape={img_ch.shape}"

    # make sure both are normalized
    attn = normalize(attn) 
    img_ch = normalize(img_ch)

    return globals()[f"corr_{corr_config.CORR_METHOD}"](attn, img_ch, corr_config)


def get_percentiles(data, prc_list = [25,50,75], axis=0):
    perc_tuple = ()
    for prc in prc_list:
        res = np.percentile(data, prc, axis=axis)
        perc_tuple += (res,)
    return perc_tuple


def compute_corr_data(attn_map, channels, corr_config:AttnScoresBaseConfig):
    """
        input:
            attn_map: attention maps values, already in the img shape (H,W), rescale to [0,1]
            channels: image input channels
            corr_method
        
        returns:
            corrs: list of correlation for each channel
            normalized_ent: normalized [0,1] entropy of the attn map
            corr_config: config with the correlation scores parameters
    """

    corrs = []
    for channel in channels:
        # compute correlation - 
        ch_corr = compute_correlation(attn_map, channel, corr_config)
        if isinstance(ch_corr, tuple):
            corrs.append(ch_corr)
        else:
            corrs.append((ch_corr,))  # Add dummy dimension

    return corrs

def parse_corr_data_rollout(corr_data):
    """
    corr_data: [ [float, float, ...], entropy ]
    Returns: np.array([ch_1, ch_2, ..., ch_M, entropy])
    """
    corr_values = corr_data[0]          # list of floats, length = num_channels
    entropy_value = corr_data[1]        # single float
    corr_data_reshaped = np.array(corr_values + [entropy_value])
    return corr_data_reshaped

def parse_corr_data_all_layers(corr_data):
    """
    corr_data: [ [ [float list], entropy ], ... ] with length = num_layers
    Returns: np.array of shape (num_channels + 1, num_layers)
    """
    corr_data_reshaped = [parse_corr_data_rollout(layer_data) for layer_data in corr_data]
    corr_data_reshaped = np.array(corr_data_reshaped).T  # Transpose to get (num_channels + 1, num_layers)
    return corr_data_reshaped


def parse_corr_data_list(corr_data):

        """
            Converts corr_data from list of corr data items returned from *compute_corr_data* into a an np.array
            
            args:
                corr_data: list of length N, such that each item is a ([list of correlation for each channel], entropy value)

            returns:
                corr_data_reshaped: np.array of shape (N, num_channels+1)
        """
        corr_data_reshaped = []

        for ch_index in range(len(corr_data[0])): # iterate on the number of channels
            corr_list = [item[0][ch_index] for item in corr_data]
            corr_data_reshaped.append(corr_list)
            
        ent_list = [item[1] for item in corr_data]
        corr_data_reshaped.append(ent_list)
        corr_data_reshaped = np.array(corr_data_reshaped) 
        # transpose to have (num_samples x [num_channels + 1 (entropy))])
        corr_data_reshaped = corr_data_reshaped.T

        return corr_data_reshaped


##################### calculate correlation##################### 

def _compute_attn_corr(processed_attn_map, sample_info, corr_config:AttnScoresBaseConfig):
    # Sample Info
    img_path, site, tile, label = sample_info
    marker, nucleus, input_img = load_tile(img_path, tile)

    if processed_attn_map.ndim == 3:
        # All layers: (num_layers, N, N)
        corr_data_all_layers = []
        for layer_attn in processed_attn_map:
            corr_data = compute_corr_data(layer_attn, [nucleus, marker], corr_config)
            corr_data_all_layers.append(corr_data)
        return corr_data_all_layers

    elif processed_attn_map.ndim == 2:
        # Rollout: (N, N)
        corr_data = compute_corr_data(processed_attn_map, [nucleus, marker], corr_config)
        return corr_data

    else:
        raise ValueError(f"Unexpected processed_attn_map shape: {processed_attn_map.shape}")



def compute_attn_correlations(processed_attn_maps: np.ndarray[float], labels: np.ndarray[str], 
                    paths: np.ndarray[str], data_config: DatasetConfig, corr_config:AttnScoresBaseConfig):
    """
    for each sample in processed_attn_maps create and saves a figure of the input image, its attention map and overlay. 
    in the process it calculate ad return each samples correlation score between the attn map and the input image. 

    """

    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[compute_attn_correlations] unique_batches: {unique_batches}')

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
    
    all_corr_data = []
    for i, set_type in enumerate(data_set_types):
        cur_attn_maps, cur_labels, cur_paths = processed_attn_maps[i], labels[i], paths[i]
        batch_of_label = get_batches_from_labels(cur_labels, data_config)
        __dict_temp = {batch: np.where(batch_of_label==batch)[0] for batch in unique_batches}

        img_path_df = parse_paths(cur_paths)
        logging.info(f'[compute_attn_correlations]: for set {set_type}, starting calculating correlation for {len(cur_paths)} samples.')
        
        set_corr_data = []
        for batch, batch_indexes in __dict_temp.items():

            #extract current batch samples
            batch_attn_maps = cur_attn_maps[batch_indexes]
            batch_labels = cur_labels[batch_indexes]
            batch_paths = cur_paths[batch_indexes]

            for index, (sample_attn, label, img_path) in enumerate(zip(batch_attn_maps, batch_labels, batch_paths)):
                # load img details
                path_item = img_path_df.iloc[index]
                img_path, tile, site = parse_path_item(path_item)

                # compute corr
                corr_data = _compute_attn_corr(sample_attn, (img_path, site, tile, label), corr_config)
                set_corr_data.append(corr_data)
        
        # end of set type
        set_corr_data = np.stack(set_corr_data)
        all_corr_data.append(set_corr_data)

    # end of samples
    return all_corr_data