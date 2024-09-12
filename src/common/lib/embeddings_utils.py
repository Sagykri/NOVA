import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME")) 

from typing import List, Optional, Tuple, Callable
from copy import deepcopy
import numpy as np
import logging
import torch

from src.common.lib.utils import get_if_exists
from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_NOVA import DatasetNOVA
from src.common.configs.dataset_config import DatasetConfig
from src.common.lib.models.NOVA_model import NOVAModel
from src.datasets.label_utils import get_batches_from_labels, get_unique_parts_from_labels, get_markers_from_labels,\
    edit_labels_by_config, get_batches_from_input_folders, get_reps_from_labels, get_conditions_from_labels, get_cell_lines_from_labels

###############################################################
# Utils for Generate Embeddings (run from MOmaps/src/runables/generate_embeddings.py)
###############################################################

def generate_embeddings(model, config_data:DatasetConfig, batch_size:int=700, num_workers:int=6)->Tuple[List[np.ndarray[torch.Tensor]],List[np.ndarray[str]]]:#TODO:add NOVAMODEL to the model type    
    logging.info(f"[generate_embeddings] Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"[generate_embeddings] Num GPUs Available: {torch.cuda.device_count()}")


    all_embeddings, all_labels = [], []

    train_paths:np.ndarray[str] = model.trainset_paths
    val_paths:np.ndarray[str] = model.valset_paths
    
    full_dataset = DatasetNOVA(config_data)
    all_paths = full_dataset.get_X_paths()
    all_labels = full_dataset.get_y()

    for set_paths, set_type in zip([train_paths, val_paths, None],
                                   ['trainset','valset','testset']):
        if set_type=='testset':
            paths_to_remove = np.concatenate([train_paths, val_paths])
            current_paths = full_dataset.get_X_paths()
            current_labels = full_dataset.get_y()
            indices_to_keep = np.where(~np.isin(current_paths, paths_to_remove))[0]
            assert indices_to_keep.shape[0] == current_paths.shape[0] - paths_to_remove.shape[0]
            
            new_set_paths = current_paths[indices_to_keep]
            new_set_labels = current_labels[indices_to_keep]
        
        else:
            indices_to_keep = np.where(np.isin(all_paths, set_paths))[0]
            if indices_to_keep.shape[0]==0:
                continue
            
            new_set_paths = all_paths[indices_to_keep]
            new_set_labels = all_labels[indices_to_keep]

        new_set_dataset = deepcopy(full_dataset).setXy(new_set_paths, new_set_labels)
        embeddings, labels = __generate_embeddings_with_dataloader(new_set_dataset, model, batch_size, num_workers)
        
        all_embeddings.append(embeddings)
        all_labels.append(labels)

    return all_embeddings, all_labels

def save_embeddings(embeddings:List[np.ndarray[torch.Tensor]], labels:List[np.ndarray[str]], data_config:DatasetConfig, output_folder_path)->None:#TODO:add NOVAMODEL to the model type
    os.makedirs(output_folder_path, exist_ok=True)
    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[save_embeddings] unique_batches: {unique_batches}')
    
    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
        
    for i, set_type in enumerate(data_set_types):
        cur_embeddings, cur_labels = embeddings[i], labels[i]
        batch_of_label = get_batches_from_labels(cur_labels, data_config)
        __dict_temp = {batch: np.where(batch_of_label==batch)[0] for batch in unique_batches}
        for batch, batch_indexes in __dict_temp.items():
            # create folder if needed
            batch_save_path = os.path.join(output_folder_path, 'embeddings', data_config.EXPERIMENT_TYPE, batch)
            os.makedirs(batch_save_path, exist_ok=True)
            
            if not data_config.SPLIT_DATA:
                # If we want to save a full batch (without splittint to train/val/test), the name still will be testset.npy.
                # This is why we want to make sure that in this case, we never saved already the train/val/test sets, because this would mean this batch was used as training batch...
                if os.path.exists(os.path.join(batch_save_path,f'trainset_labels.npy')) or os.path.exists(os.path.join(batch_save_path,f'valset_labels.npy')):
                    logging.warning(f"[save_embeddings] SPLIT_DATA={data_config.SPLIT_DATA} BUT there exists trainset or valset in folder {batch_save_path}!! make sure you don't overwrite the testset!!")
            logging.info(f"[save_embeddings] Saving {len(batch_indexes)} in {batch_save_path}")
            
            np.save(os.path.join(batch_save_path,f'{set_type}_labels.npy'), np.array(cur_labels[batch_indexes]))
            np.save(os.path.join(batch_save_path,f'{set_type}.npy'), cur_embeddings[batch_indexes])

            logging.info(f'[save_embeddings] Finished {set_type} set, saved in {batch_save_path}')

def load_embeddings(model_output_folder:str, config_data:DatasetConfig)-> Tuple[np.ndarray[float], np.ndarray[str]]:
    """Loads the vit embeddings 
    """

    experiment_type = get_if_exists(config_data, 'EXPERIMENT_TYPE', None)
    assert experiment_type is not None, "EXPERIMENT_TYPE can't be None"
    logging.info(f"[load_embeddings] experiment_type = {experiment_type}")
    

    input_folders = get_if_exists(config_data, 'INPUT_FOLDERS', None)
    assert input_folders is not None, "INPUT_FOLDERS can't be None"
    logging.info(f"[load_embeddings] input_folders = {input_folders}")

    logging.info(f"[load_embeddings] model_output_folder = {model_output_folder}")

    batches = get_batches_from_input_folders(input_folders)
    embeddings_folder = os.path.join(model_output_folder,"embeddings", experiment_type)
    embeddings, labels = __load_multiple_batches(batches = batches,embeddings_folder = embeddings_folder,
                                                 config_data=config_data)
    
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    labels = edit_labels_by_config(labels, config_data)
    filtered_labels, filtered_embeddings = __filter(labels, embeddings, config_data)

    logging.info(f'[load_embeddings] embeddings shape: {filtered_embeddings.shape}')
    logging.info(f'[load_embeddings] labels shape: {filtered_labels.shape}')
    logging.info(f'[load_embeddings] example label: {filtered_labels[0]}')
    return filtered_embeddings, filtered_labels

def __generate_embeddings_with_dataloader(dataset:DatasetNOVA, model:NOVAModel, batch_size:int=700, 
                                          num_workers:int=6)->Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]:
    data_loader = get_dataloader(dataset, batch_size, num_workers, drop_last=False)
    logging.info(f"[generate_embeddings_with_dataloader] Data loaded: there are {len(dataset)} images.")

    embeddings, labels = model.infer(data_loader)
    logging.info(f'[generate_embeddings_with_dataloader] total embeddings: {embeddings.shape}')
    
    return embeddings, labels

def __load_multiple_batches(batches:List[str], embeddings_folder:str, config_data:DatasetConfig)-> Tuple[List[np.ndarray[float]],List[np.ndarray[np.str_]]]:
    
    """Load embeddings and labels in given batches
    Args:        
        batches (List[str]): List of batch folder names to load (e.g., ['batch6', 'batch7'])
        embeddings_folder (str): full path to stored embeddings
        config_data (DatasetConfig): dataset config is used to check if data needs to be split (train/val/test)
    Returns:
        embeddings: List of np.arrays of length (# batches). each np.array is in shape (# tiles, 128)
        labels: List of np.arrays of length (# batches). each np.array is in shape (# tiles) and the stored value is full label
    """
    sets_to_load = config_data.SETS #TODO: change to default "testset" after  genereating embeddings with new approach
    embeddings, labels = [] , []
    for batch in batches:
        for set_type in sets_to_load:
            cur_embeddings, cur_labels = np.load(os.path.join(embeddings_folder, batch, f"{set_type}.npy")),\
                                         np.load(os.path.join(embeddings_folder, batch, f"{set_type}_labels.npy"))
            embeddings.append(cur_embeddings)
            labels.append(cur_labels)
    return embeddings, labels

def __filter(labels:np.ndarray[str], embeddings:np.ndarray[float], 
            config_data:DatasetConfig)->Tuple[np.ndarray[str],np.ndarray[float]]:
    # Extract from config_data the filtering required on the labels
    cell_lines = get_if_exists(config_data, 'CELL_LINES', None)
    conditions = get_if_exists(config_data, 'CONDITIONS', None)
    markers_to_exclude = get_if_exists(config_data, 'MARKERS_TO_EXCLUDE', None)
    markers = get_if_exists(config_data, 'MARKERS', None)
    reps = get_if_exists(config_data, 'REPS', None)

    # Perform the filtering
    if markers_to_exclude:
        logging.info(f"[embeddings_utils._filter] markers_to_exclude = {markers_to_exclude}")
        labels, embeddings = __filter_by_label_part(labels, embeddings, markers_to_exclude,
                                  get_markers_from_labels, include=False)
    if markers:
        logging.info(f"[embeddings_utils._filter] markers = {markers}")
        labels, embeddings = __filter_by_label_part(labels, embeddings, markers,
                                  get_markers_from_labels, include=True)
    if cell_lines:
        logging.info(f"[embeddings_utils._filter] cell_lines = {cell_lines}")
        if config_data.ADD_LINE_TO_LABEL:
            labels, embeddings = __filter_by_label_part(labels, embeddings, cell_lines,
                                  get_cell_lines_from_labels, config_data, include=True)
        else:
            logging.warning(f'[embeddings_utils._filter]: Cannot filter by cell lines because of config_data: ADD_LINE_TO_LABEL:{config_data.ADD_LINE_TO_LABEL}')

    if conditions:
        logging.info(f"[embeddings_utils._filter] conditions = {conditions}")
        if config_data.ADD_CONDITION_TO_LABEL:
            labels, embeddings = __filter_by_label_part(labels, embeddings, conditions,
                                  get_conditions_from_labels, config_data, include=True)
        else:
            logging.warning(f'[embeddings_utils._filter]: Cannot filter by condition because of config_data: ADD_CONDITION_TO_LABEL: {config_data.ADD_CONDITION_TO_LABEL}')

    if reps:
        logging.info(f"[embeddings_utils._filter] reps = {reps}") 
        if config_data.ADD_REP_TO_LABEL:
            labels, embeddings = __filter_by_label_part(labels, embeddings, reps,
                                  get_reps_from_labels, config_data, include=True)
        else:
            logging.warning(f'[embeddings_utils._filter]: Cannot filter by reps because of config_data: ADD_REP_TO_LABEL:{config_data.ADD_REP_TO_LABEL}')

    return labels, embeddings

def __filter_by_label_part(labels:np.ndarray[str], embeddings:np.ndarray[float], 
                          filter_on:List[str], get_parts_from_labels:Callable, config_data:Optional[DatasetConfig]=None, 
                          include:bool=True,) -> Tuple[np.ndarray[str],np.ndarray[float]]:
    
    if config_data is not None:
        parts_of_labels = get_parts_from_labels(labels, config_data)
    else:
        parts_of_labels = get_parts_from_labels(labels)
    if include:
        indices_to_keep = np.where(np.isin(parts_of_labels, filter_on))[0]
    if not include:
        indices_to_keep = np.where(~np.isin(parts_of_labels, filter_on))[0]
    labels = labels[indices_to_keep]
    embeddings = embeddings[indices_to_keep]
    return labels, embeddings







