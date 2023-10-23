import multiprocessing
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np
import pandas as pd
import itertools  
import logging
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from src.common.lib.image_sampling_utils import find_marker_folders
from src.common.lib.utils import flat_list_of_lists, get_if_exists, load_config_file, init_logging
from src.common.lib.model import Model
from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_spd import DatasetSPD
import re

###############################################################
# Utils for Generate Embeddings (run from MOmaps/src/runables/generate_embeddings.py)
###############################################################
def init_model_for_embeddings(config_path_model):
    """initiates the trained model

    Args:
        config_path_model (string): The path to config file of the model (src.common.lib.model.Model) to be used for infering the embeddings 

    Returns:
        model: trained model (src.common.lib.model.Model object)
        config_model: src.models.neuroself.configs.model_config
    """

    # Get configs of model (trained model) 
    config_model = load_config_file(config_path_model, 'model')
    model = Model(config_model)
    logging.info(f"Init model {config_model}")
    return model, config_model

def load_dataset_for_embeddings(config_data, batch_size, config_model):
    """Returns torch.utils.data.DataLoader objects 

    Use the dataset config (src.datasets.configs.train_config) to load the dataset that we want to calc embbedings for
    Init a DatasetSPD object (src.datasets.dataset_spd.DatasetSPD)
    If needed, returns the DataLoader with the original train/val/test split

    Args:
        config_data (Dataset): Dataset config object (src.datasets.configs.train_config)
        batch_size (int): 

    Returns:
        torch.utils.data.DataLoader object/s 
    """
    
    # Init dataset
    dataset = DatasetSPD(config_data)
    logging.info(f"Data shape: {dataset.X_paths.shape}, {dataset.y.shape}")
    
        
    __unique_labels_path = os.path.join(config_model.MODEL_OUTPUT_FOLDER, "unique_labels.npy")
    if os.path.exists(__unique_labels_path):
        logging.info(f"[load_dataset_for_embeddings] unique_labels.npy files has been detected - using it. ({__unique_labels_path})")
        dataset.unique_markers = np.load(__unique_labels_path)
    else:
        logging.warn(f"[load_dataset_for_embeddings] Couldn't find unique_labels file: {__unique_labels_path}")
    
    # important! we don't want to get the augmented images
    dataset.flip, dataset.rot = False, False
    
    logging.info(f"Init dataloaders (batch_size={batch_size})")
    if config_data.SPLIT_DATA:
        logging.info(f"Get the data split that was used during training...")
        # Get numeric indexes of train, val and test sets
        train_indexes, val_indexes, test_indexes = dataset.split()
        # Get loaders
        dataloader_train, dataloader_val, dataloader_test = get_dataloader(dataset, batch_size, indexes=train_indexes, num_workers=2),\
                                                            get_dataloader(dataset, batch_size, indexes=val_indexes, num_workers=2),\
                                                            get_dataloader(dataset, batch_size, indexes=test_indexes, num_workers=2)
        
        return [dataloader_train, dataloader_val, dataloader_test]
    
    else:
        # Load the data
        # Include all the data by using "indexes=None"
        dataloader = get_dataloader(dataset, batch_size, indexes=None, num_workers=2)    
    
        return [dataloader]
    
def load_model_with_dataloader(model, datasets_list):
    """Actual loading of the trained model 
    
    Args:
        model: trained model (src.common.lib.model.Model object)
        datasets_list (_type_): list of torch.utils.data.DataLoader object/s 

    Returns:
        model: trained model with datasets
    """
    
    if (model.conf.MODEL_PATH is None) and (model.conf.LAST_CHECKPOINT_PATH is not None): 
        model.conf.MODEL_PATH = model.conf.LAST_CHECKPOINT_PATH
    else:
        logging.info(f"MODEL_PATH and LAST_CHECKPOINT_PATH are None.")
    
    logging.info(f"Loading model with dataloader {model.conf.MODEL_PATH}")

    if len(datasets_list)==3:
        # If data was splitted during training to train/val/test
        model.load_with_dataloader(datasets_list[0], datasets_list[1], datasets_list[2])
    elif len(datasets_list)==1:
        # If data was not used during training 
        model.load_with_dataloader(test_loader=datasets_list[0])
    else:
        logging.exception("[Generate Embeddings] Load model: List of datasets is not supported.")
    
    # Actual load of the model
    model.load_model()
    return model

def save_embeddings_and_labels(embedding_data, embeddings_folders, name):
    """_summary_

    Args:
        embedding_data (_type_): _description_
        embeddings_folders (string): the path to the embeddings folder (under model_outputs)
        name (string): "train/val/test/all"

    Returns:
        _type_: _description_
    """
    embeddings_folders_unique = np.unique(embeddings_folders)
    __dict_temp = {value: [index for index, item in enumerate(embeddings_folders) if item == value] for value in embeddings_folders_unique}
    for embeddings_folders_marker, marker_indexes in __dict_temp.items():
        # create folder if needed
        os.makedirs(embeddings_folders_marker, exist_ok=True) 
        # embeddings file name 
        embeddings_file_name = os.path.join(embeddings_folders_marker, name) + '_embeddings.npy'
        logging.info(f"Saving embeddings {name}. Path: {embeddings_file_name}")
        # save npy to relevant folder
        np.save(embeddings_file_name, embedding_data[marker_indexes])
    return None

def calc_embeddings(model, datasets_list, embeddings_folder, save=True, embeddings_layer='vqvec2'):

    # Parser to get the image's batch/cell_line/condition/rep/marker
    def final_save_path(full_path):
        path_list = full_path.split(os.sep)
        batch_cell_line_condition_rep_marker_list = [os.path.join(path_list[-1][:4],path_list[i]) if i==-2 else os.path.join(path_list[i]) for i in range(-5,-1)]
        batch_cell_line_condition_rep_marker = os.path.join(*batch_cell_line_condition_rep_marker_list)
        return os.path.join(embeddings_folder, embeddings_layer, batch_cell_line_condition_rep_marker)
    get_save_path = np.vectorize(final_save_path)
    
    def do_embeddings_inference(images_batch, dataset_type):
        save_path = get_save_path(images_batch['image_path'])
        # images_batch is torch.Tensor of size(n_tiles, n_channels, 100, 100)
        embedding_data = model.model.infer_embeddings(images_batch['image'].numpy(), output_layer=embeddings_layer)  
        if save: save_embeddings_and_labels(embedding_data, save_path, name=dataset_type+str(i))
        return None
    
    if len(datasets_list)==3:
        
        logging.info("Infer embeddings - train set")
        # compute the latent features of a batch of imgaes
        for i, images_batch in enumerate(datasets_list[0]):
            do_embeddings_inference(images_batch, dataset_type = 'trainset')
            
        logging.info("Infer embeddings - val set")
        for i, images_batch in enumerate(datasets_list[1]):
            do_embeddings_inference(images_batch, dataset_type = 'valset')
        
        logging.info("Infer embeddings - test set")
        for i, images_batch in enumerate(datasets_list[2]):
            do_embeddings_inference(images_batch, dataset_type = 'testset')
    
    elif len(datasets_list)==1:
        logging.info("Infer embeddings -  all data set")
        for i, images_batch in enumerate(datasets_list[0]):
            do_embeddings_inference(images_batch, dataset_type = 'all')
    else:
        logging.exception("[Generate Embeddings] Load model: List of datasets is not supported.")
    
    return None
        

###############################################################
# Utils for Generate spectral features (run from MOmaps/src/runables/generate_spectral_features.py)
################################################################ 

def calc_spectral_features(model, datasets_list, output_folder, save=True, output_layer = f'vqindhist1'):

    # Parser to get the image's batch/cell_line/condition/rep/marker
    def final_save_path(full_path):
        path_list = full_path.split(os.sep)
        # to create separate batch folders
        batch = path_list[-5] 
        # to create labels of each image
        batch_cell_line_condition_rep_marker_list = [os.path.join(path_list[-1][:4],path_list[i]) if i==-2 else os.path.join(path_list[i]) for i in range(-5,-1)]
        batch_cell_line_condition_rep_marker = os.path.join(*batch_cell_line_condition_rep_marker_list)
        return os.path.join(output_folder, output_layer, batch), batch_cell_line_condition_rep_marker
    get_save_path_and_labels = np.vectorize(final_save_path)
    
    
    def do_embeddings_inference(images_batch, dataset_type, 
                                images_spectral_features, images_labels, processed_images_path, save_paths):
        save_path, labels = get_save_path_and_labels(images_batch['image_path'])

        # images_batch is torch.Tensor of size(n_tiles, n_channels, 100, 100) - only because batch_size==1!!!!
        embedding_data = model.model.infer_embeddings(images_batch['image'].numpy(), output_layer=output_layer)
        before = len(images_labels)
        images_labels.extend(labels)
        logging.info(f"images_labels length before: {before}, adding labels length {len(labels)} = {len(images_labels)}")
        paths = [f'{path}_{n_tile}' for n_tile, path in enumerate(images_batch['image_path'])]
        processed_images_path.extend(paths)
        images_spectral_features.append(embedding_data)
        save_paths.extend(save_path)
        return images_spectral_features, images_labels, processed_images_path, save_paths
    
    def save(features, labels, paths, output_path, dataset_type):
            unique_output_paths = np.unique(output_path)
            __dict_temp = {value: [index for index, item in enumerate(output_path) if item == value] for value in unique_output_paths}
            for batch_save_path, batch_indexes in __dict_temp.items():
                # create folder if needed
                os.makedirs(batch_save_path, exist_ok=True)
                logging.info(f"Saving {len(batch_indexes)} ({dataset_type}) indhists in {batch_save_path}")
                np.save(os.path.join(batch_save_path, f'vqindhist1_{dataset_type}.npy'), features[batch_indexes])
                np.save(os.path.join(batch_save_path, f'vqindhist1_labels_{dataset_type}.npy'), np.array(labels)[batch_indexes])
                np.save(os.path.join(batch_save_path, f'vqindhist1_paths_{dataset_type}.npy'), np.array(paths)[batch_indexes])
            return None
    
    if len(datasets_list)==3:
        logging.info("Infer embeddings - test set")
        images_spectral_features, images_labels, processed_images_path, save_paths = [], [], [], []
        for i, images_batch in enumerate(datasets_list[2]):
            images_spectral_features, images_labels, processed_images_path, save_paths = do_embeddings_inference(images_batch, 'testset', images_spectral_features, images_labels, processed_images_path, save_paths)
        images_spectral_features = np.concatenate(images_spectral_features)
        save(images_spectral_features, images_labels, processed_images_path, save_paths, "testset")
        
        logging.info("Infer embeddings - train set")
        # compute the latent features of a batch of imgaes
        images_spectral_features, images_labels, processed_images_path, save_paths = [], [], [], []
        for i, images_batch in enumerate(datasets_list[0]):
            images_spectral_features, images_labels, processed_images_path, save_paths = do_embeddings_inference(images_batch, 'trainset', images_spectral_features, images_labels, processed_images_path, save_paths)
        images_spectral_features = np.concatenate(images_spectral_features)
        save(images_spectral_features, images_labels, processed_images_path, save_paths, "trainset")

        logging.info("Infer embeddings - val set")
        images_spectral_features, images_labels, processed_images_path, save_paths = [], [], [], []
        for i, images_batch in enumerate(datasets_list[1]):
            images_spectral_features, images_labels, processed_images_path, save_paths = do_embeddings_inference(images_batch, 'valset', images_spectral_features, images_labels, processed_images_path, save_paths)
        images_spectral_features = np.concatenate(images_spectral_features)
        save(images_spectral_features, images_labels, processed_images_path, save_paths, "valset")
        
    
    elif len(datasets_list)==1:
        logging.info("Infer embeddings -  all data set")
        images_spectral_features, images_labels, processed_images_path, save_paths = [], [], [], []
        for i, images_batch in enumerate(datasets_list[0]):
            images_spectral_features, images_labels, processed_images_path, save_paths = do_embeddings_inference(images_batch, 'all',images_spectral_features, images_labels, processed_images_path, save_paths)
        images_spectral_features = np.concatenate(images_spectral_features)
        save(images_spectral_features, images_labels, processed_images_path, save_paths, "all")
    else:
        logging.exception("[Generate spectral features] Load model: List of datasets is not supported.")
    
    return None



###############################################################
# Utils for Load Embeddings (callable function)
################################################################ 

def get_embeddings_subfolders_filtered(config_data, embeddings_main_folder, depth=4):
    """_summary_

    Args:
        config_data: Use the dataset config to decide which cell/condition/marker to filter out
        embeddings_main_folder (string): _description_
        depth (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """

    # Parse batch input folders 
    batch_names = [batch_name.split(os.sep)[-1] for batch_name in config_data.INPUT_FOLDERS]
    emb_batch_folders = [os.path.join(embeddings_main_folder, batch) for batch in batch_names]
    
    # For every marker in this batch, get (in lazy manner) list of files that pass filtration
    marker_folders_to_include = []
    for i, input_folder in enumerate(emb_batch_folders):
        # Get marker folders (last level in folder structure)
        marker_subfolders = find_marker_folders(input_folder, depth=depth, exclude_DAPI=False)
        logging.info(f"Input folder: {input_folder}, depth used: {depth}")
        
        for marker_folder in marker_subfolders:
                #####################################
                # Extract experimental settings from marker folder path (avoid multiple nested for loops..)
                marker_name = os.path.basename(marker_folder)
                rep =  marker_folder.split('/')[-2]
                condition = marker_folder.split('/')[-3]
                cell_line = marker_folder.split('/')[-4]
                #####################################
                # Filter: cell line
                if config_data.CELL_LINES is not None and cell_line not in config_data.CELL_LINES:
                    logging.info(f"Skipping cell line (not in cell lines list). {cell_line}")
                    continue
                # Filter: stress condition
                if config_data.CONDITIONS is not None and condition not in config_data.CONDITIONS:
                    logging.info(f"Skipping condition (not in conditions list). {condition}")
                    continue
                # Filter: rep
                if config_data.REPS is not None and rep not in config_data.REPS:
                    logging.info(f"Skipping rep (not in reps list). {rep}")
                    continue
                # Filter: marker to include
                if config_data.MARKERS is not None and marker_name not in config_data.MARKERS:
                    logging.info(f"Skipping marker (not in markers list). {marker_name}")
                    continue
                # Filter: marker to exclude
                if config_data.MARKERS_TO_EXCLUDE is not None and marker_name in config_data.MARKERS_TO_EXCLUDE:
                    logging.info(f"Skipping (in markers to exclude). {marker_name}")
                    continue
                #####################################
                marker_folders_to_include.append(marker_folder)

    if len(marker_folders_to_include) == 0:
        logging.warn("[get_embeddings_subfolders_filtered] Couldn't find any embeddings for your data")

    return marker_folders_to_include

def __handle_load_stored_embeddings(embeddings_type, experiment_type, config_data, config_model, embeddings_layer):
    def __get_embeddings_and_labels(embeddings_layer):
        embeddings_main_folder = os.path.join(config_model.MODEL_OUTPUT_FOLDER, 'embeddings', experiment_type, embeddings_layer)
        
        marker_folders_to_include = get_embeddings_subfolders_filtered(config_data, embeddings_main_folder)
        
        def __parallel_load(paths, embeddings_type, config_data):
            num_processes = multiprocessing.cpu_count()
            logging.info(f"[load_embeddings] Running in parallel: {num_processes} processes")
            __params = [(path, embeddings_type, config_data) for path in  paths]
            with multiprocessing.Pool(num_processes) as pool:
                results = pool.starmap(_load_stored_embeddings, __params)
            
            embeddings, labels = zip(*results)
            labels = flat_list_of_lists(list(labels))
            return list(embeddings), labels
        
        embedings_data_list, all_labels = __parallel_load(marker_folders_to_include, embeddings_type, config_data)
        all_labels = np.asarray(all_labels).reshape(-1,1)
        
        # Combine all markers to single numpy 
        if len(embedings_data_list) == 0:
            all_embedings_data = np.asarray([])  
            logging.warn('[__handle_load_stored_embeddings] 0 embeddings were loaded')  
        else:
            all_embedings_data = np.vstack(embedings_data_list)
            logging.info(f"[__handle_load_stored_embeddings] all_embedings_data: {all_embedings_data.shape} all_labels: {all_labels.shape}")
        
        return all_embedings_data, all_labels
    
    if embeddings_layer == 'vqvec_both':
        logging.info(f"embeddings_layer is {embeddings_layer}. Loading (and concatenating) both vqvec1 and vqvec2 embeddings...")
        logging.info("Loading vqvec1 embeddings...")
        emb_vq1, labels_vq1 = __get_embeddings_and_labels('vqvec1')
        logging.info("Loading vqvec2 embeddings...")
        emb_vq2, labels_vq2 = __get_embeddings_and_labels('vqvec2')
        
        assert all(labels_vq1 == labels_vq2), "Labels (vq1, vq2) mismatch"
        
        logging.info("Converting vqvec1 and vqvec2 from np arrays to tensors")
        emb_vq1 = torch.from_numpy(emb_vq1)
        emb_vq2 = torch.from_numpy(emb_vq2)
        logging.info("Resizing vqvec2 embeddings")
        emb_vq2 = resize(emb_vq2, config_model.EMB_SHAPES[0], interpolation=InterpolationMode.NEAREST)
        logging.info("Concatenating vqvec1 and vqvec2 embeddings")
        emb_vq_both = torch.cat([emb_vq2, emb_vq1], 1)
        labels_vq_both = labels_vq1
        
        logging.info("Converting final embeddings from tensor to np array")
        emb_vq_both = emb_vq_both.numpy()
        logging.info(f"emb_vq_both.shape = {emb_vq_both.shape}, labels_vq_both.shape = {labels_vq_both.shape}")
        
        return emb_vq_both, labels_vq_both
    
    return __get_embeddings_and_labels(embeddings_layer)

def _load_stored_embeddings(marker_folder, embeddings_type, config_data):
    """Load all pre-stored embeddings (npy files)

    Args:
        marker_folder (string): The full path to the marker folder
        embeddings_type (string): _description_

    Returns:
        embedings_data (ndarray): 
        labels (list):  
    """

    # List of all stored embedding npy files under this marker folder
    emb_filenames = sorted(os.listdir(marker_folder))

    # Filter npy files by "embeddings_type"
    filtered_emb_filenames = [emb_filename for emb_filename in emb_filenames if embeddings_type in emb_filename]
    
    # Filter 0 size npy files (corrupted emebddings...)
    filtered_emb_filenames = [emb_filename for emb_filename in filtered_emb_filenames if os.path.getsize(os.path.join(marker_folder, emb_filename))!=0]

    # Load all embeddings .npy files into a single numpy array
    embedings_data = np.vstack([np.load(os.path.join(marker_folder, emb_filename)) for emb_filename in filtered_emb_filenames])

    # Infer the label 
    path_list = marker_folder.split(os.sep)
    batch_cell_line_condition_rep_marker = '_'.join(path_list[-4-int(config_data.ADD_BATCH_TO_LABEL):])
    if not config_data.ADD_REP_TO_LABEL:
        pattern = re.compile(r'_rep\d+')
        batch_cell_line_condition_rep_marker = re.sub(pattern, '', batch_cell_line_condition_rep_marker)
        
    labels = [batch_cell_line_condition_rep_marker] * embedings_data.shape[0]
    
    logging.info(f"[_load_stored_embeddings] Loading stored embeddings of label {batch_cell_line_condition_rep_marker} of shape {embedings_data.shape} ")
    return embedings_data, labels
    

def load_embeddings(config_path_model=None, config_path_data=None,
                    config_model=None, config_data=None, embeddings_type='valset'):
    """Loads the embedding vectors 

    Args:
        config_path_model (string): full path to trained model config file 
        config_path_data (string): full path to dataset config file
        embeddings_type (string): which part of the dataset to fetch "train"/"test"/"val"/"all"
    """
    if config_path_model is None and config_model is None:
        raise ValueError("Invalid config (path). Must supply model config.")
    if config_path_data is None and config_data is None:
        raise ValueError("Invalid config (path). Must supply dataset config.")
    if embeddings_type not in ["trainset", "testset", "valset", "all"]:
        raise ValueError(f"Invalid embeddings_type. Must supply 'trainset' / 'testset' / 'valset' / 'all'. ")
    
    logging.info(f"[load_embeddings] Model: {config_path_model if config_path_model is not None else 'preloaded'}\
                    Dataset: {config_path_data if config_path_data is not None else 'preloaded'},\
                        embeddings_type: {embeddings_type}")
    
    # Get configs of model (trained model) 
    config_model = load_config_file(config_path_model, 'model') if config_model is None else config_model
    
    # Get dataset configs (as to be used in the desired UMAP)
    config_data = load_config_file(config_path_data, 'data') if config_data is None else config_data
    
    experiment_type = get_if_exists(config_data, 'EXPERIMENT_TYPE', None)
    assert experiment_type is not None, "EXPERIMENT_TYPE can't be None"
    logging.info(f"[load_embeddings] experiment_type = {experiment_type}")
    
    embeddings_layer = get_if_exists(config_data, 'EMBEDDINGS_LAYER', 'vqvec2')
    logging.info(f"[load_embeddings] embeddings_layer = {embeddings_layer}")
    
    all_embedings_data, all_labels = __handle_load_stored_embeddings(embeddings_type, experiment_type, config_data, config_model, embeddings_layer)
           
    return all_embedings_data, all_labels


# TODO: NANCY delete this after testing 
if __name__ == "__main__":
    
    #if len(sys.argv) != 3:
    #    raise ValueError("Invalid config path. Must supply model config and data config.")
    try:
        # Use case of data NOT used in training  (B6)
        # all_embedings_data, all_labels = load_embeddings(
        #   config_path_model='./src/models/neuroself/configs/model_config/NeuroselfB78TrainingConfig', 
        #   config_path_data='.src/datasets/configs/train_config/EmbeddingsB6DatasetConfig',
        #   embeddings_type='all')
        
        
        # Use case of data used in training  (B7+8)
        all_embedings_data, all_labels = load_embeddings(
            config_path_model='./src/models/neuroself/configs/model_config/NeuroselfB78TrainingConfig', 
            config_path_data='.src/datasets/configs/train_config/EmbeddingsB78DatasetConfig',
            embeddings_type='testset')

    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done!")



        