import os
import sys

from src.datasets.label_utils import get_cell_lines_conditions_from_labels, edit_labels_by_config, get_batches_from_labels, get_cell_lines_from_labels, get_conditions_from_labels, get_markers_from_labels, get_reps_from_labels, get_unique_parts_from_labels, get_batches_from_input_folders
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np
import logging
import torch

from typing import List, Optional, Tuple, Callable
from src.common.lib.utils import get_if_exists
from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_spd import DatasetSPD
from src.common.lib.dataset import Dataset
from src.common.configs.dataset_config import DatasetConfig
# from src.common.lib.models.NOVA_model import NOVAModel
from sandbox.eval_new_arch.dino4cells.main_vit_fine_tuning import infer_pass #TODO: remove


def _filter(labels:np.ndarray[str], embeddings:np.ndarray[float], 
            config_data:DatasetConfig)->Tuple[np.ndarray[str],np.ndarray[float]]:
    # Extract from config_data the filtering required on the labels
    cell_lines_conds = get_if_exists(config_data, 'CELL_LINES_CONDS', None)
    cell_lines = get_if_exists(config_data, 'CELL_LINES', None)
    conditions = get_if_exists(config_data, 'CONDITIONS', None)
    markers_to_exclude = get_if_exists(config_data, 'MARKERS_TO_EXCLUDE', None)
    markers = get_if_exists(config_data, 'MARKERS', None)
    reps = get_if_exists(config_data, 'REPS', None)

    # Perform the filtering
    if cell_lines_conds:
        logging.info(f"[embeddings_utils._filter] cell_lines_conds = {cell_lines_conds}")
        if config_data.ADD_LINE_TO_LABEL and config_data.ADD_CONDITION_TO_LABEL:
            labels, embeddings = _filter_by_label_part(labels, embeddings, cell_lines_conds,
                                  get_cell_lines_conditions_from_labels, config_data, include=True)
        else:
            logging.warning(f'[embeddings_utils._filter]: Cannot filter by cell line condition because of config_data: ADD_LINE_TO_LABEL:{config_data.ADD_LINE_TO_LABEL}, ADD_CONDITION_TO_LABEL: {config_data.ADD_CONDITION_TO_LABEL}')

    if markers_to_exclude:
        logging.info(f"[embeddings_utils._filter] markers_to_exclude = {markers_to_exclude}")
        labels, embeddings = _filter_by_label_part(labels, embeddings, markers_to_exclude,
                                  get_markers_from_labels, include=False)
    if markers:
        logging.info(f"[embeddings_utils._filter] markers = {markers}")
        labels, embeddings = _filter_by_label_part(labels, embeddings, markers,
                                  get_markers_from_labels, include=True)
    if cell_lines:
        logging.info(f"[embeddings_utils._filter] cell_lines = {cell_lines}")
        if config_data.ADD_LINE_TO_LABEL:
            labels, embeddings = _filter_by_label_part(labels, embeddings, cell_lines,
                                  get_cell_lines_from_labels, config_data, include=True)
        else:
            logging.warning(f'[embeddings_utils._filter]: Cannot filter by cell lines because of config_data: ADD_LINE_TO_LABEL:{config_data.ADD_LINE_TO_LABEL}')

    if conditions:
        logging.info(f"[embeddings_utils._filter] conditions = {conditions}")
        if config_data.ADD_CONDITION_TO_LABEL:
            labels, embeddings = _filter_by_label_part(labels, embeddings, conditions,
                                  get_conditions_from_labels, config_data, include=True)
        else:
            logging.warning(f'[embeddings_utils._filter]: Cannot filter by condition because of config_data: ADD_CONDITION_TO_LABEL: {config_data.ADD_CONDITION_TO_LABEL}')

    if reps:
        logging.info(f"[embeddings_utils._filter] reps = {reps}") 
        if config_data.ADD_REP_TO_LABEL:
            labels, embeddings = _filter_by_label_part(labels, embeddings, reps,
                                  get_reps_from_labels, config_data, include=True)
        else:
            logging.warning(f'[embeddings_utils._filter]: Cannot filter by reps because of config_data: ADD_REP_TO_LABEL:{config_data.ADD_REP_TO_LABEL}')

    return labels, embeddings


def _filter_by_label_part(labels:np.ndarray[str], embeddings:np.ndarray[float], 
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

def load_embeddings(model_output_folder:str, config_data:DatasetConfig)-> Tuple[np.ndarray[float], np.ndarray[str]]:
    """Loads the vit embeddings 
    """
    training_batches = get_if_exists(config_data, 'TRAIN_BATCHES', None)
    if training_batches:
        logging.info(f'config_data.TRAIN_BATCHES: {training_batches}')

    experiment_type = get_if_exists(config_data, 'EXPERIMENT_TYPE', None)
    assert experiment_type is not None, "EXPERIMENT_TYPE can't be None"
    logging.info(f"[load_embeddings] experiment_type = {experiment_type}")
    

    input_folders = get_if_exists(config_data, 'INPUT_FOLDERS', None)
    assert input_folders is not None, "INPUT_FOLDERS can't be None"
    logging.info(f"[load_embeddings] input_folders = {input_folders}")

    logging.info(f"[load_embeddings] model_output_folder = {model_output_folder}")

    batches = get_batches_from_input_folders(input_folders)
    embeddnigs_folder = os.path.join(model_output_folder,"embeddings", experiment_type)
    embeddings, labels = load_multiple_batches(batches = batches,embeddings_folder = embeddnigs_folder,
                                                 config_data=config_data,training_batches=training_batches)
    
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    labels = edit_labels_by_config(labels, config_data)
    filtered_labels, filtered_embeddings = _filter(labels, embeddings, config_data)

    logging.info(f'[load_embeddings] embeddings shape: {filtered_embeddings.shape}')
    logging.info(f'[load_embeddings] labels shape: {filtered_labels.shape}')
    logging.info(f'[load_embeddings] example label: {filtered_labels[0]}')
    return filtered_embeddings, filtered_labels

def load_multiple_batches(batches:List[str], embeddings_folder:str, config_data:DatasetConfig, 
                          training_batches:Optional[List[str]] = ['batch7', 'batch8'])-> Tuple[List[np.ndarray[float]],List[np.ndarray[np.str_]]]:
    
    """Load embeddings and labels in given batches
    Args:        
        batches (List[str]): List of batch folder names to load (e.g., ['batch6', 'batch7'])
        embeddings_folder (str): full path to stored embeddings
        config_data (DatasetConfig): dataset config is used to check if data needs to be split (train/val/test)
        training_batches (Optional[List[str]] = ['batch7', 'batch8']): is used for the case where we want to load mulitple batches, while some of them needs to be splitted (training batches) and some of them not.
    Returns:
        embeddings: List of np.arrays of length (# batches). each np.array is in shape (# tiles, 128)
        labels: List of np.arrays of length (# batches). each np.array is in shape (# tiles) and the stored value is full label
    """
    
    embeddings, labels = [] , []
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
        embeddings.append(cur_vit_features)
        labels.append(cur_labels)
    return embeddings, labels

def generate_embeddings_with_dataloader(dataset:DatasetSPD, model)->Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]: #TODO:add NOVAMODEL to the model type
    # data_loader = get_dataloader(dataset, model.trainer_config.BATCH_SIZE, num_workers=model.trainer_config.NUM_WORKERS, drop_last=False)
    data_loader = get_dataloader(dataset, 700, num_workers=6, drop_last=True) #TODO: remove
    logging.info(f"[generate_embeddings_with_dataloader] Data loaded: there are {len(dataset)} images.")

    # embeddings, labels = model.infer(data_loader)
    embeddings, labels, _ , _ = infer_pass(model, data_loader) #TODO:remove
    logging.info(f'[generate_embeddings_with_dataloader] total embeddings: {embeddings.shape}')
    
    return embeddings, labels

def save_embeddings(embeddings:List[np.ndarray[torch.Tensor]], labels:List[np.ndarray[str]], data_config:DatasetConfig, output_folder_path)->None:#TODO:add NOVAMODEL to the model type
    os.makedirs(output_folder_path, exist_ok=True)
    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[save_embeddings] unique_batches: {unique_batches}')
    
    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['all']
    for i, set_type in enumerate(data_set_types):
        cur_embeddings, cur_labels = embeddings[i], labels[i]
        batch_of_label = get_batches_from_labels(cur_labels, data_config)
        __dict_temp = {batch: np.where(batch_of_label==batch)[0] for batch in unique_batches}
        for batch, batch_indexes in __dict_temp.items():
            # create folder if needed
            batch_save_path = os.path.join(output_folder_path, 'embeddings', data_config.EXPERIMENT_TYPE, batch)
            os.makedirs(batch_save_path, exist_ok=True)
            
            logging.info(f"[save_embeddings] Saving {len(batch_indexes)} in {batch_save_path}")
            
            np.save(os.path.join(batch_save_path,f'{set_type}_labels.npy'), np.array(cur_labels[batch_indexes]))
            np.save(os.path.join(batch_save_path,f'{set_type}.npy'), cur_embeddings[batch_indexes])

            logging.info(f'[save_embeddings] Finished {set_type} set, saved in {batch_save_path}')

def generate_embeddings(model, config_data:DatasetConfig)->Tuple[List[np.ndarray[torch.Tensor]],List[np.ndarray[str]]]:#TODO:add NOVAMODEL to the model type    
    logging.info(f"[generate_embeddings] Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"[generate_embeddings] Num GPUs Available: {torch.cuda.device_count()}")

    if config_data.SPLIT_DATA: # we need to load all the training markers (remove DAPI), then split, then load only DAPI and split, then concat them, This is because DAPI wasn't in the training
        all_embeddings, all_labels = [], []
        config_data.MARKERS_TO_EXCLUDE = config_data.MARKERS_TO_EXCLUDE + ['DAPI']
        dataset = DatasetSPD(config_data)
        logging.info("[generate_embeddings] Split data...")
        train_indexes, val_indexes, test_indexes = dataset.split()
        
        for idx, set_type in zip([train_indexes, val_indexes, test_indexes],['trainset','valset','testset']):
            dataset_subset = Dataset.get_subset(dataset, idx)
            logging.info(f'[generate_embeddings] running on {set_type}')
            
            if set_type=='testset':
                config_data.MARKERS = ['DAPI']
                config_data.MARKERS_TO_EXCLUDE.remove('DAPI')
                dataset_DAPI = DatasetSPD(config_data)
                if len(dataset_DAPI) > 0:
                    _, _, test_DAPI_indexes = dataset_DAPI.split()
                    dataset_DAPI_subset = Dataset.get_subset(dataset_DAPI, test_DAPI_indexes) 
                    dataset_subset.unique_markers = np.concatenate((dataset_subset.unique_markers, dataset_DAPI_subset.unique_markers), axis=1) #TODO: test axis (mine was 1, Sagy was 0)
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
