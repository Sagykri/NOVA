import multiprocessing
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np
import pandas as pd
import itertools  
import logging

from src.common.lib.utils import flat_list_of_lists, get_if_exists, load_config_file, init_logging

def load_vit_features(model_output_folder, config_data, training_batches=['batch7','batch8']):
    """Loads the vit vectors 
    """
    
    experiment_type = get_if_exists(config_data, 'EXPERIMENT_TYPE', None)
    assert experiment_type is not None, "EXPERIMENT_TYPE can't be None"
    logging.info(f"[load_vit_features] experiment_type = {experiment_type}")
    
    logging.info(f"[load_vit_features] model_output_folder = {model_output_folder}")

    input_folders = get_if_exists(config_data, 'INPUT_FOLDERS', None)
    assert input_folders is not None, "INPUT_FOLDERS can't be None"
    logging.info(f"[load_vit_features] input_folders = {input_folders}")

    cell_lines_conds = get_if_exists(config_data, 'CELL_LINES_CONDS', None)
    if cell_lines_conds:
        logging.info(f"[load_vit_features] cell_lines_conds = {cell_lines_conds}")
    
    cell_lines = get_if_exists(config_data, 'CELL_LINES', None)
    if cell_lines:
        logging.info(f"[load_vit_features] cell_lines = {cell_lines}")

    conditions = get_if_exists(config_data, 'CONDITIONS', None)
    if conditions:
        logging.info(f"[load_vit_features] conditions = {conditions}")

    markers_to_exclude = get_if_exists(config_data, 'MARKERS_TO_EXCLUDE', None)
    if markers_to_exclude:
        logging.info(f"[load_vit_features] markers_to_exclude = {markers_to_exclude}")
    
    markers = get_if_exists(config_data, 'MARKERS', None)
    if markers:
        logging.info(f"[load_vit_features] markers = {markers}")
    
    reps = get_if_exists(config_data, 'REPS', None)
    if reps:
        logging.info(f"[load_vit_features] reps = {reps}")

    batches = [folder.split(os.sep)[-1].split('_')[0] for folder in input_folders]
    embeddnigs_folder = os.path.join(model_output_folder,"embeddings", experiment_type)
    vit_features, labels = load_multiple_vit_feaures(batches = batches,
                                                    embeddings_folder = embeddnigs_folder,
                                                    config_data=config_data,
                                                    training_batches=training_batches)
    
    vit_features = np.concatenate(vit_features)
    labels = np.concatenate(labels)
    vit_df = pd.DataFrame(vit_features)
    vit_df['label'] = labels
    def rearrange_string(s, config_data):
            parts = s.split('_')
            if config_data.EXPERIMENT_TYPE == 'U2OS':
                return f"{parts[6]}_{parts[3]}_{parts[4]}_{parts[0]}_{parts[5]}"
            else:
                return f"{parts[4]}_{parts[1]}_{parts[2]}_{parts[0]}_{parts[3]}"
    
    vit_df['label'] = vit_df['label'].apply(lambda x: rearrange_string(x, config_data))
    
    if cell_lines_conds:
        vit_df = vit_df[vit_df.label.str.contains('|'.join(cell_lines_conds), regex=True)]        
    if markers_to_exclude:
        vit_df = vit_df[~vit_df.label.str.startswith(tuple(markers_to_exclude))]        
    if markers:
        vit_df = vit_df[vit_df.label.str.startswith(tuple(markers))]        
    if cell_lines:
        vit_df = vit_df[vit_df['label'].str.split('_', expand=True)[1].isin(cell_lines)]        
    if conditions:
        vit_df = vit_df[vit_df['label'].str.contains('|'.join(conditions), regex=True)]        
    if reps:
        vit_df = vit_df[vit_df['label'].str.contains('|'.join(reps), regex=True)]        

    all_embedings_data = np.array(vit_df.drop(columns=['label']))
    logging.info(f'[load_vit_features] all_embedings_data shape: {all_embedings_data.shape}')
    all_labels = np.array(vit_df['label'])
    logging.info(f'[load_vit_features] all_labels shape: {all_labels.shape}')
    logging.info(f'[load_vit_features] example label: {all_labels[0]}')
    return all_embedings_data, all_labels

def load_multiple_vit_feaures(batches, embeddings_folder, config_data, training_batches=['batch7','batch8']):
    
    """Load vqinhist1, labels and paths of tiles in given batches
    Args:        
        batches (list of strings): list of batch folder names (e.g., ['batch6', 'batch7'])
        embeddings_folder (string): full path to stored embeddings
    Returns:
        vit_features: list of np.arrays from shape (# cell lines). each np.array is in shape (# tiles, 2048)
        labels: list of np.arrays from shape (# cell lines). each np.array is in shape (# tiles) and the stored value is full label
    """
    
    vit_features, labels = [] , []
    set_type = 'testset' if config_data.SPLIT_DATA else 'all'
    for batch in batches:
        if set_type=='all' and training_batches is not None and batch in training_batches:
            cur_set_type='testset'
            logging.info(f'[load_multiple_vit_feaures] loading only testset for {batch} although set_type:{set_type} for DISTANCES!')
        else:
            cur_set_type=set_type
        cur_vit_features, cur_labels =  np.load(os.path.join(embeddings_folder, batch, f"{cur_set_type}.npy")),\
                                        np.load(os.path.join(embeddings_folder, batch, f"{cur_set_type}_labels.npy"))
        cur_vit_features = cur_vit_features.reshape(cur_vit_features.shape[0], -1)
        vit_features.append(cur_vit_features)
        labels.append(cur_labels)
    return vit_features, labels

