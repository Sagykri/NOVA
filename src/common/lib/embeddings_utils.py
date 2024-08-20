import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np
import pandas as pd
import logging
import datetime
import torch

from src.common.lib.utils import get_if_exists, load_config_file, init_logging
from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_spd import DatasetSPD
from src.common.lib.dataset import Dataset

from sandbox.eval_new_arch.dino4cells.main_vit_fine_tuning import infer_pass

def load_embeddings(model_output_folder, config_data, training_batches=['batch7','batch8']):
    """Loads the vit embeddings 
    """
    
    experiment_type = get_if_exists(config_data, 'EXPERIMENT_TYPE', None)
    assert experiment_type is not None, "EXPERIMENT_TYPE can't be None"
    logging.info(f"[load_embeddings] experiment_type = {experiment_type}")
    
    logging.info(f"[load_embeddings] model_output_folder = {model_output_folder}")

    input_folders = get_if_exists(config_data, 'INPUT_FOLDERS', None)
    assert input_folders is not None, "INPUT_FOLDERS can't be None"
    logging.info(f"[load_embeddings] input_folders = {input_folders}")

    cell_lines_conds = get_if_exists(config_data, 'CELL_LINES_CONDS', None)
    if cell_lines_conds:
        logging.info(f"[load_embeddings] cell_lines_conds = {cell_lines_conds}")
    
    cell_lines = get_if_exists(config_data, 'CELL_LINES', None)
    if cell_lines:
        logging.info(f"[load_embeddings] cell_lines = {cell_lines}")

    conditions = get_if_exists(config_data, 'CONDITIONS', None)
    if conditions:
        logging.info(f"[load_embeddings] conditions = {conditions}")

    markers_to_exclude = get_if_exists(config_data, 'MARKERS_TO_EXCLUDE', None)
    if markers_to_exclude:
        logging.info(f"[load_embeddings] markers_to_exclude = {markers_to_exclude}")
    
    markers = get_if_exists(config_data, 'MARKERS', None)
    if markers:
        logging.info(f"[load_embeddings] markers = {markers}")
    
    reps = get_if_exists(config_data, 'REPS', None)
    if reps:
        logging.info(f"[load_embeddings] reps = {reps}")

    batches = [folder.split(os.sep)[-1].split('_')[0] for folder in input_folders]
    embeddnigs_folder = os.path.join(model_output_folder,"embeddings", experiment_type)
    vit_features, labels = load_multiple_batches(batches = batches,
                                                    embeddings_folder = embeddnigs_folder,
                                                    config_data=config_data,
                                                    training_batches=training_batches)
    
    vit_features = np.concatenate(vit_features)
    labels = np.concatenate(labels)
    vit_df = pd.DataFrame(vit_features)
    vit_df['label'] = labels
    
    
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
    logging.info(f'[load_embeddings] all_embedings_data shape: {all_embedings_data.shape}')
    all_labels = np.array(vit_df['label'])
    logging.info(f'[load_embeddings] all_labels shape: {all_labels.shape}')
    logging.info(f'[load_embeddings] example label: {all_labels[0]}')
    return all_embedings_data, all_labels

def load_multiple_batches(batches, embeddings_folder, config_data, training_batches=['batch7','batch8']):
    
    """Load embeddings and labels in given batches
    Args:        
        batches (list of strings): list of batch folder names (e.g., ['batch6', 'batch7'])
        embeddings_folder (string): full path to stored embeddings
        config_data: dataset config is used to check if data needs to be split (train/val/test)
        training_batches (list of string or None): is used for the case where we want to load mulitple batches, while some of them needs to be splitted and some of them not.
    Returns:
        vit_features: list of np.arrays from shape (# batches). each np.array is in shape (# tiles, 128)
        labels: list of np.arrays from shape (# batches). each np.array is in shape (# tiles) and the stored value is full label
    """
    
    vit_features, labels = [] , []
    set_type = 'testset' if config_data.SPLIT_DATA else 'all'
    for batch in batches:
        if set_type=='all' and training_batches is not None and batch in training_batches:
            cur_set_type='testset'
            logging.info(f'[load_multiple_batches] loading only testset for {batch} although set_type:{set_type} for DISTANCES!')
        else:
            cur_set_type=set_type
        cur_vit_features, cur_labels =  np.load(os.path.join(embeddings_folder, batch, f"{cur_set_type}.npy")),\
                                        np.load(os.path.join(embeddings_folder, batch, f"{cur_set_type}_labels.npy"))
        cur_vit_features = cur_vit_features.reshape(cur_vit_features.shape[0], -1)
        vit_features.append(cur_vit_features)
        labels.append(cur_labels)
    return vit_features, labels

def rearrange_string(s, config_data):
            parts = s.split('_')
            if config_data.EXPERIMENT_TYPE == 'U2OS':
                return f"{parts[6]}_{parts[3]}_{parts[4]}_{parts[0]}_{parts[5]}"
            else:
                return f"{parts[4]}_{parts[1]}_{parts[2]}_{parts[0]}_{parts[3]}"

def generate_embeddings_with_dataloader(dataset, model):
    data_loader = get_dataloader(dataset, model.trainer_config.batch_size_per_gpu, num_workers=model.trainer_config.num_workers, drop_last=False)
    logging.info(f"Data loaded: there are {len(dataset)} images.")

    embeddings, _, _, labels = infer_pass(model, data_loader)
    logging.info(f'total embeddings: {embeddings.shape}')
    
    return embeddings, labels

def save_embeddings(model, embeddings, labels, data_config):
    output_folder_path = model.model_config.output_folder_path
    os.makedirs(output_folder_path, exist_ok=True)
    
    unique_batches = np.unique([label.split('_')[0] for label in labels])
    logging.info(f'unique_batches: {unique_batches}')
    
    if data_config.SPLIT_DATA:
        for i, set_type in enumerate(['trainset','valset','testset']):
            cur_embeddings, cur_labels = embeddings[i], labels[i]
            __dict_temp = {value: [index for index, item in enumerate(cur_labels) if item.split('_')[0] == value] for value in unique_batches}
            for batch, batch_indexes in __dict_temp.items():
                # create folder if needed
                batch_save_path = os.path.join(output_folder_path, 'embeddings', data_config.EXPERIMENT_TYPE, batch)
                os.makedirs(batch_save_path, exist_ok=True)
                
                logging.info(f"Saving {len(batch_indexes)} in {batch_save_path}")
                
                np.save(os.path.join(batch_save_path,f'{set_type}_labels.npy'), np.array(cur_labels[batch_indexes]))
                np.save(os.path.join(batch_save_path,f'{set_type}.npy'), cur_embeddings[batch_indexes])

                logging.info(f'Finished {set_type} set, saved in {batch_save_path}')
    else:
        set_type = 'all'
        cur_embeddings, cur_labels = embeddings[0], labels[0]
        __dict_temp = {value: [index for index, item in enumerate(cur_labels) if item.split('_')[0] == value] for value in unique_batches}
        for batch, batch_indexes in __dict_temp.items():
            # create folder if needed
            batch_save_path = os.path.join(output_folder_path, 'embeddings', data_config.EXPERIMENT_TYPE, batch)
            os.makedirs(batch_save_path, exist_ok=True)
            
            logging.info(f"Saving {len(batch_indexes)} in {batch_save_path}")
            
            np.save(os.path.join(batch_save_path,f'{set_type}_labels.npy'), np.array(cur_labels[batch_indexes]))
            np.save(os.path.join(batch_save_path,f'{set_type}.npy'), cur_embeddings[batch_indexes])

            logging.info(f'Finished {set_type} set, saved in {batch_save_path}')

def generate_embeddings(model, config_path_data):
    output_folder_path = model.model_config.output_folder_path
    os.makedirs(output_folder_path, exist_ok=True)
    
    logs_folder = os.path.join(output_folder_path, "logs")
    os.makedirs(logs_folder, exist_ok=True)

    __now = datetime.datetime.now()
    jobid = os.getenv('LSB_JOBID')
    init_logging(os.path.join(logs_folder, __now.strftime("%d%m%y_%H%M%S_%f") + f'_{jobid}_embeddings.log'))
    
    logging.info(f"init (jobid: {jobid})")
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")

    config_data = load_config_file(config_path_data) 

    if config_data.SPLIT_DATA: # we need to load all the training markers (remove DAPI), then split, then load only DAPI and split, then concat them, This is because DAPI wasn't in the training
        all_embeddings, all_labels = [], []
        config_data.MARKERS_TO_EXCLUDE = config_data.MARKERS_TO_EXCLUDE + ['DAPI']
        dataset = DatasetSPD(config_data)
        logging.info("Split data...")
        train_indexes, val_indexes, test_indexes = dataset.split()
        
        for idx, set_type in zip([train_indexes, val_indexes, test_indexes],['trainset','valset','testset']):
            dataset_subset = Dataset.get_subset(dataset, idx)
            logging.info(f'running on {set_type}')
            
            if set_type=='testset':
                config_data = load_config_file(config_path_data)
                config_data.MARKERS = ['DAPI']
                dataset_DAPI = DatasetSPD(config_data)
                _, _, test_DAPI_indexes = dataset_DAPI.split()
                dataset_DAPI_subset = Dataset.get_subset(dataset_DAPI, test_DAPI_indexes) 
                dataset_subset.unique_markers = np.concatenate((dataset_subset.unique_markers, dataset_DAPI_subset.unique_markers), axis=1)
                dataset_subset.label = np.concatenate((dataset_subset.label, dataset_DAPI_subset.label), axis=0)
                dataset_subset.X_paths = np.concatenate((dataset_subset.X_paths, dataset_DAPI_subset.X_paths), axis=0)
                dataset_subset.y = np.concatenate((dataset_subset.y, dataset_DAPI_subset.y), axis=0)
            embeddings, labels = generate_embeddings_with_dataloader(dataset_subset, model)
            all_embeddings.append(embeddings)
            all_labels.append(labels)
        
        return all_embeddings, all_labels
    
    else:
        dataset_subset = DatasetSPD(config_data)
        embeddings, labels = generate_embeddings_with_dataloader(dataset_subset, model)
        return [embeddings], [labels]
