import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME")) 

from typing import List, Optional, Tuple, Callable
from copy import deepcopy
import numpy as np
import logging
import torch

from src.common.utils import get_if_exists
from src.datasets.data_loader import get_dataloader
from src.datasets.dataset_NOVA import DatasetNOVA
from src.datasets.dataset_config import DatasetConfig
from src.models.architectures.NOVA_model import NOVAModel
from src.datasets.label_utils import get_batches_from_labels, get_unique_parts_from_labels, get_markers_from_labels,\
    edit_labels_by_config, get_batches_from_input_folders, get_reps_from_labels, get_conditions_from_labels, get_cell_lines_from_labels

###############################################################
# Utils for Generate Embeddings (run from HOME/src/runables/generate_embeddings.py)
###############################################################

def generate_embeddings(model:NOVAModel, config_data:DatasetConfig, 
                        batch_size:int=700, num_workers:int=6)->Tuple[List[np.ndarray[torch.Tensor]],
                                                                      List[np.ndarray[str]]]:
    logging.info(f"[generate_embeddings] Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"[generate_embeddings] Num GPUs Available: {torch.cuda.device_count()}")

    all_embeddings, all_labels, all_paths, all_paths = [], [], [], []

    train_paths:np.ndarray[str] = model.trainset_paths
    val_paths:np.ndarray[str] = model.valset_paths
    
    full_dataset = DatasetNOVA(config_data)
    full_paths = full_dataset.get_X_paths()
    full_labels = full_dataset.get_y()
    logging.info(f'[generate_embbedings]: total files in dataset: {full_paths.shape[0]}')
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


        logging.info(f'[generate_embbedings]: for set {set_type}, there are {new_set_paths.shape} paths and {new_set_labels.shape} labels')
        new_set_dataset = deepcopy(full_dataset)
        new_set_dataset.set_Xy(new_set_paths, new_set_labels)
        
        embeddings, labels, paths = __generate_embeddings_with_dataloader(new_set_dataset, model, batch_size, num_workers)
        
        all_embeddings.append(embeddings)
        all_labels.append(labels)
        all_paths.append(paths)

    return all_embeddings, all_labels, all_paths

def save_embeddings(embeddings:List[np.ndarray[torch.Tensor]], 
                    labels:List[np.ndarray[str]], paths:List[np.ndarray[str]],
                    data_config:DatasetConfig, output_folder_path)->None:

    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[save_embeddings] unique_batches: {unique_batches}')
    
    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
        
    for i, set_type in enumerate(data_set_types):
        cur_embeddings, cur_labels, cur_paths = embeddings[i], labels[i], paths[i]
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
            np.save(os.path.join(batch_save_path,f'{set_type}_paths.npy'), cur_paths[batch_indexes])

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
    embeddings, labels, paths = __load_multiple_batches(batches = batches,embeddings_folder = embeddings_folder,
                                                 config_data=config_data)
    
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    paths = np.concatenate(paths)
    labels = edit_labels_by_config(labels, config_data)
    filtered_labels, filtered_embeddings, filtered_paths = __filter(labels, embeddings, paths, config_data)

    logging.info(f'[load_embeddings] embeddings shape: {filtered_embeddings.shape}')
    logging.info(f'[load_embeddings] labels shape: {filtered_labels.shape}')
    logging.info(f'[load_embeddings] example label: {filtered_labels[0]}')
    logging.info(f'[load_embeddings] paths shape: {filtered_paths.shape}')
    return filtered_embeddings, filtered_labels, filtered_paths

def __generate_embeddings_with_dataloader(dataset:DatasetNOVA, model:NOVAModel, batch_size:int=700, 
                                          num_workers:int=6)->Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]:
    data_loader = get_dataloader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    logging.info(f"[generate_embeddings_with_dataloader] Data loaded: there are {len(dataset)} images.")
    
    embeddings, labels, paths = model.infer(data_loader)
    logging.info(f'[generate_embeddings_with_dataloader] total embeddings: {embeddings.shape}')
    
    return embeddings, labels, paths

def __load_multiple_batches(batches:List[str], embeddings_folder:str, config_data:DatasetConfig)-> Tuple[List[np.ndarray[float]],List[np.ndarray[np.str_]]]:
    
    """Load embeddings and labels in given batches
    Args:        
        batches (List[str]): List of batch folder names to load (e.g., ['batch6', 'batch7'])
        embeddings_folder (str): full path to stored embeddings
        config_data (DatasetConfig): dataset config is used to check if data needs to be split (train/val/test)
    Returns:
        embeddings: List of np.arrays of length (# batches). each np.array is in shape (# tiles, 128)
        labels: List of np.arrays of length (# batches). each np.array is in shape (# tiles) and the stored value is full label
        paths: List of np.arrays of length (# batches). each np.array is in shape (# tiles) and the stored value is the path to the tile's image
    """
    sets_to_load = get_if_exists(config_data, 'SETS', ['testset']) 
    embeddings, labels, paths = [] , [], []
    for batch in batches:
        for set_type in sets_to_load:
            cur_embeddings, cur_labels = np.load(os.path.join(embeddings_folder, batch, f"{set_type}.npy")),\
                                         np.load(os.path.join(embeddings_folder, batch, f"{set_type}_labels.npy"))
            paths_path  = os.path.join(embeddings_folder, batch, f"{set_type}_paths.npy")
            if os.path.isfile(paths_path):
                cur_paths = np.load(paths_path)
            else:
                cur_paths = np.full(cur_labels.shape, None, dtype=object)            
            embeddings.append(cur_embeddings)
            labels.append(cur_labels)
            paths.append(cur_paths)
    return embeddings, labels, paths

def __filter(labels:np.ndarray[str], embeddings:np.ndarray[float], paths:np.ndarray[str],
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
        labels, embeddings, paths = __filter_by_label_part(labels, embeddings, paths, markers_to_exclude,
                                  get_markers_from_labels, include=False)
    if markers:
        logging.info(f"[embeddings_utils._filter] markers = {markers}")
        labels, embeddings, paths = __filter_by_label_part(labels, embeddings, paths, markers,
                                  get_markers_from_labels, include=True)
    if cell_lines:
        logging.info(f"[embeddings_utils._filter] cell_lines = {cell_lines}")
        if config_data.ADD_LINE_TO_LABEL:
            if get_if_exists(config_data, 'COMMON_BASELINE', None):
                common_cell_line = config_data.COMMON_BASELINE
                cell_lines = np.array(cell_lines, dtype=str)  # Ensure it's an array  
                cell_lines[np.char.find(cell_lines, common_cell_line) >= 0] = common_cell_line  
            labels, embeddings, paths = __filter_by_label_part(labels, embeddings, paths, cell_lines,
                                  get_cell_lines_from_labels, config_data, include=True)
        else:
            logging.warning(f'[embeddings_utils._filter]: Cannot filter by cell lines because of config_data: ADD_LINE_TO_LABEL:{config_data.ADD_LINE_TO_LABEL}')

    if conditions:
        logging.info(f"[embeddings_utils._filter] conditions = {conditions}")
        if config_data.ADD_CONDITION_TO_LABEL:
            labels, embeddings, paths = __filter_by_label_part(labels, embeddings, paths, conditions,
                                  get_conditions_from_labels, config_data, include=True)
        else:
            logging.warning(f'[embeddings_utils._filter]: Cannot filter by condition because of config_data: ADD_CONDITION_TO_LABEL: {config_data.ADD_CONDITION_TO_LABEL}')

    if reps:
        logging.info(f"[embeddings_utils._filter] reps = {reps}") 
        if config_data.ADD_REP_TO_LABEL:
            labels, embeddings, paths = __filter_by_label_part(labels, embeddings, paths, reps,
                                  get_reps_from_labels, config_data, include=True)
        else:
            logging.warning(f'[embeddings_utils._filter]: Cannot filter by reps because of config_data: ADD_REP_TO_LABEL:{config_data.ADD_REP_TO_LABEL}')
    return labels, embeddings, paths

def __filter_by_label_part(labels:np.ndarray[str], embeddings:np.ndarray[float], paths:np.ndarray[str],
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
    paths = paths[indices_to_keep]
    return labels, embeddings, paths







