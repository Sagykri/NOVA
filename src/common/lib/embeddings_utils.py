import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np
import pandas as pd
import itertools  
import logging

from src.common.lib.image_sampling_utils import find_marker_folders
from src.common.lib.utils import load_config_file, init_logging
from src.common.lib.model import Model
from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_spd import DatasetSPD

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

def load_dataset_for_embeddings(config_path_data, batch_size):
    """Returns torch.utils.data.DataLoader objects 

    Use the dataset config (src.datasets.configs.train_config) to load the dataset that we want to calc embbedings for
    Init a DatasetSPD object (src.datasets.dataset_spd.DatasetSPD)
    If needed, returns the DataLoader with the original train/val/test split

    Args:
        config_path_data (string): Dataset config object (src.datasets.configs.train_config)
        batch_size (int): 

    Returns:
        torch.utils.data.DataLoader object/s 
    """
    
    # Get dataset configs (as used in trainig the model)
    config_data = load_config_file(config_path_data, 'data') 
    logging.info(f"Init datasets {config_data} from {config_path_data}")
    # Init dataset
    dataset = DatasetSPD(config_data)
    logging.info(f"Data shape: {dataset.X_paths.shape}, {dataset.y.shape}")
    
    # important! we don't want to get the augmented images
    dataset.flip, dataset.rot = False, False
    
    logging.info(f"Init dataloaders")
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

def save_embeddings_and_labels(embedding_data, labels, embeddings_folders, name):
    """_summary_

    Args:
        embedding_data (_type_): _description_
        labels (_type_): _description_
        embeddings_folders (string): the path to the embeddings folder (under model_outputs)
        name (string): "train/val/test/all"

    Returns:
        _type_: _description_
    """
    
    for i in range(embedding_data.shape[0]):
        # create folder if needed
        os.makedirs(embeddings_folders[i], exist_ok=True) 
        # embeddings file name 
        embeddings_file_name = os.path.join(embeddings_folders[i], name) + '_embeddings.npy'
        logging.info(f"Saving embeddings {name}. Path: {embeddings_file_name}")
        # save npy to relevant folder
        np.save(embeddings_file_name, embedding_data[i,:,:,:])
    return None

def calc_embeddings(model, datasets_list, embeddings_folder, save=True):

    # Parser to get the image's batch/cell_line/condition/marker
    def final_save_path(full_path):
        path_list = full_path.split(os.sep)
        batch_cell_line_condition_marker = os.path.join(*[os.path.join(path_list[i]) for i in range(-5,-1)])
        return os.path.join(embeddings_folder, batch_cell_line_condition_marker)
    get_save_path = np.vectorize(final_save_path)
    
    def do_embeddings_inference(images_batch, dataset_type):
        save_path = get_save_path(images_batch['image_path'])
        # images_batch is torch.Tensor of size(n_tiles, n_channels, 100, 100)
        embedding_data = model.model.infer_embeddings(images_batch['image'].numpy())  
        if save: save_embeddings_and_labels(embedding_data, images_batch['label'], save_path, name=dataset_type+str(i))

    if len(datasets_list)==3:
        
        logging.info("Infer embeddings - train set")
        # compute the latent features of a batch of imgaes
        for i, images_batch in enumerate(datasets_list[0]):
            do_embeddings_inference(images_batch, dataset_type = 'trainset')
            
        logging.info("Infer embeddings - val set")
        for i, images_batch in enumerate(datasets_list[1]):
            do_embeddings_inference(images_batch, dataset_type = 'valtest')
        
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
# Utils for Load Embeddings (callable function)
################################################################ 

def get_embeddings_subfolders_filtered(config_data, embeddings_main_folder, depth=3):
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
    for i, input_folder in enumerate(emb_batch_folders):
        # Get marker folders (last level in folder structure)
        marker_subfolders = find_marker_folders(input_folder, depth=depth, exclude_DAPI=False)
        logging.info(f"Input folder: {input_folder}, depth used: {depth}")
        
        marker_folders_to_include = []

        for marker_folder in marker_subfolders:
                #####################################
                # Extract experimental settings from marker folder path (avoid multiple nested for loops..)
                marker_name = os.path.basename(marker_folder)
                condition = marker_folder.split('/')[-2]
                cell_line = marker_folder.split('/')[-3]
                
                #####################################
                # Filter: cell line
                if config_data.CELL_LINES is not None and cell_line not in config_data.CELL_LINES:
                    logging.info(f"Skipping cell line (not in cell lines list). {cell_line}")
                    continue
                # Filter: stress condition
                if config_data.CONDITIONS is not None and condition not in config_data.CONDITIONS:
                    logging.info(f"Skipping condition (not in conditions list). {condition}")
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

        return marker_folders_to_include


def _load_stored_embeddings(marker_folder, embeddings_type):
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
    
    # Load all embeddings .npy files into a single numpy array
    embedings_data = np.array([np.load(os.path.join(marker_folder, emb_filename)) for emb_filename in filtered_emb_filenames])

    # Infer the label 
    path_list = marker_folder.split(os.sep)
    batch_cell_line_condition_marker = os.path.join(*[os.path.join(path_list[i]) for i in range(-5,-1)])
    labels = [batch_cell_line_condition_marker] * embedings_data.shape[0]
    
    logging.info(f"[_load_stored_embeddings] Loading stored embeddings of label {batch_cell_line_condition_marker} of shape {embedings_data.shape} ")
    return embedings_data, labels
    

def load_embeddings(config_path_model=None, config_path_data=None, embeddings_type='valset'):
    """Loads the embedding vectors 

    Args:
        config_path_model (string): full path to trained model config file 
        config_path_data (string): full path to dataset config file
        embeddings_type (string): which part of the dataset to fetch "train"/"test"/"val"/"all"
    """
    
    logging.info(f"[load_embeddings] Model: {config_path_model} Dataset: {config_path_data}")

    if config_path_model is None:
        raise ValueError("Invalid config path. Must supply model config.")
    if config_path_data is None:
        raise ValueError("Invalid config path. Must supply dataset config.")
    if embeddings_type not in ["trainset", "testset", "valset", "all"]:
        raise ValueError(f"Invalid embeddings_type. Must supply 'trainset' / 'testset' / 'valset' / 'all'. ")
    
    # Get configs of model (trained model) 
    config_model = load_config_file(config_path_model, 'model')
    embeddings_main_folder = os.path.join(config_model.MODEL_OUTPUT_FOLDER, 'embeddings')
    
    # Get dataset configs (as to be used in the desired UMAP)
    config_data = load_config_file(config_path_data, 'data')
    
    marker_folders_to_include = get_embeddings_subfolders_filtered(config_data, embeddings_main_folder)
    
    embedings_data_list, all_labels = [], []
    for marker_folder in marker_folders_to_include:
        
        embedings_data, labels = _load_stored_embeddings(marker_folder, embeddings_type)
        
        embedings_data_list.append(embedings_data)
        all_labels.extend(labels)

    # Combine all markers to single numpy 
    all_embedings_data = np.vstack(embedings_data_list)
    logging.info(f"[load_embeddings] all_embedings_data: {all_embedings_data.shape} all_labels: {len(all_labels)}")
           
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



        