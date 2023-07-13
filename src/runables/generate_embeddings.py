import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np
import pandas as pd
import itertools  
import logging
import  torch

from src.common.lib.utils import load_config_file, init_logging
from src.common.lib.model import Model
from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_spd import DatasetSPD

from src.common.configs.embeddings_config import EmbeddingsConfig

def load_trained_model_for_embedding():
    # Get model config path
    config_path_model = sys.argv[1]
    # Get configs of model (trained model) 
    config_model = load_config_file(config_path_model, 'model')
    logging.info("Init model {config_model}")
    model = Model(config_model)
    return model, config_model

def load_dataset_for_embedding(config_model):
    # Get dataset config path
    config_path_data = sys.argv[2]
    # Get configs of dataset (used in trainig the model)
    config_data = load_config_file(config_path_data, 'data', config_model.CONFIGS_USED_FOLDER) 
    logging.info(f"Init datasets {config_data} from {config_path_data}")
    dataset = DatasetSPD(config_data)
    dataset.flip, dataset.rot = False, False
    logging.info(f"Data shape: {dataset.X_paths.shape}, {dataset.y.shape}")
    
    # Dataloaders (actual load of the data)
    num_workers = 10
    config_model.BATCH_SIZE = 500
    
    logging.info(f"Init dataloaders")
    if config_data.SPLIT_DATA:
        logging.info("Get the data split that was used during training...")
        # Get numeric indexes of train, val and test sets
        train_indexes, val_indexes, test_indexes = dataset.split()
        # Get loaders
        dataloader_train, dataloader_val, dataloader_test = get_dataloader(dataset, config_model.BATCH_SIZE, indexes=train_indexes, num_workers=num_workers),\
                                                            get_dataloader(dataset, config_model.BATCH_SIZE, indexes=val_indexes, num_workers=num_workers),\
                                                            get_dataloader(dataset, config_model.BATCH_SIZE, indexes=test_indexes, num_workers=num_workers)
        
        print("\n\n\nNANXXXX", vars(dataset))
        
        return [dataloader_train, dataloader_val, dataloader_test], dataset.input_folders
    
    else:
        # Include all the data
        X_indexes, y = np.arange(len(dataset.X_paths)), dataset.y
        # Load the data
        dataloader = get_dataloader(dataset, config_model.BATCH_SIZE, indexes=X_indexes, num_workers=num_workers)    
    
        return [dataloader]

def load_embeddings_config(config_model): 
    # Load embeddings config
    embeddings_config = EmbeddingsConfig()
    # Set the output folder (where to save the embeddings)
    embeddings_config.EMBEDDINGS_FOLDER = os.path.join(config_model.MODEL_OUTPUT_FOLDER, 'embeddings')
    # Set logs folder 
    embeddings_config.LOGS_FOLDER = os.path.join(embeddings_config.EMBEDDINGS_FOLDER, 'logs')   
    logging.info(f"[{embeddings_config}] Init (log path: {embeddings_config.LOGS_FOLDER})")
    return embeddings_config
    
def load_model(model, datasets_list):
    logging.info("Loading model with dataloader")
    if len(datasets_list)==3:
        model.load_with_dataloader(datasets_list[0], datasets_list[1], datasets_list[2])
    elif len(datasets_list)==1:
        model.load_with_dataloader(test_loader=dataloader[0])
    else:
        logging.exception("[Generate Embeddings] Load model: List of datasets is not supported.")
    
    model.load_model()
    return model

def save_embeddings_and_labels(embedding_data, labels, embeddings_config, name):
    savepath_embeddings = embeddings_config.EMBEDDINGS_FOLDER
    logging.info("Saving embeddings {name}. Path: {savepath_embeddings}")
    np.save(name + '.npy', embedding_data)
    np.save(name + '_labels.npy', labels)
    return None

def calc_embeddings(model, datasets_list, embeddings_config, save=True):

    if len(datasets_list)==3:
        logging.info("Infer embeddings - train set")
        embedding_data_train = model.model.infer_embeddings(datasets_list[0])  
        print("\n\n\nNANXXXX", datasets_list[0], vars(datasets_list[0]))
        if save: save_embeddings_and_labels(embedding_data_train, datasets_list[0].y, embeddings_config, name='trainset')
        
        
        print("\n\nXXXX", embedding_data_train, type(embedding_data_train),datasets_list[0].y )
        
        logging.info("Infer embeddings - val set")
        embedding_data_val = model.model.infer_embeddings(datasets_list[1])  
        if save: save_embeddings_and_labels(embedding_data_val, datasets_list[1].y, embeddings_config, name='valtest')
        
        logging.info("Infer embeddings - test set")
        embedding_data_test = model.model.infer_embeddings(datasets_list[2])  
        if save: save_embeddings_and_labels(embedding_data_test, datasets_list[2].y, embeddings_config, name='testset')
        
        return [embedding_data_train, embedding_data_val, embedding_data_test]
    
    elif len(datasets_list)==1:
            logging.info("Infer embeddings -  all data")
            embedding_data = model.model.infer_embeddings(dataloader)  
            return [embedding_data]
    else:
        logging.exception("[Generate Embeddings] Load model: List of datasets is not supported.")
        
    
def generate_embeddings():
    
    
    if len(sys.argv) != 3:
        raise ValueError("Invalid config path. Must supply model config and data config.")
    
    model, config_model =  load_trained_model_for_embedding()
   
    datasets_list, dataset_input_folders = load_dataset_for_embedding(config_model)
   
    embeddings_config = load_embeddings_config(config_model)
   
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")

    model = load_model(model, datasets_list)
    
    embedding_data_list = calc_embeddings(model, datasets_list, embeddings_config, save=True)
    
    return embedding_data_list
    

if __name__ == "__main__":
    print("---------------Start---------------")
    print("Starting to generate VQVAE2 embeddings...")
    try:
        generate_embeddings()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done!")
    
# ./bash_commands/run_py.sh ./src/runables/generate_embeddings -g -m 10000 -b 10 -a ./src/models/neuroself/configs/model_test_config/NeuroselfB8PiecewiseTestingConfig ./src/datasets/configs/train_config/TrainB8PiecewiseDatasetConfig    
    
# /home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/outputs/models_outputs_batch8_piecewise/model_19.pt
    