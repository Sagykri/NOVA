import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler

# import tensorflow as tf
from copy import deepcopy
from src.datasets.dataset_spd import DatasetSPD
from src.datasets.dataset_conf import DatasetConf
from src.common.lib.utils import load_config_file
from src.common.lib.model import Model
from src.common.lib.data_loader import get_dataloader

def train_with_dataloader():
    
    # Importing customize config for this run
    config_path_model = sys.argv[1]
    is_one_config_supplied = len(sys.argv) == 3
    
    if is_one_config_supplied:
        config_path_train, config_path_val, config_path_test = sys.argv[2], sys.argv[2], sys.argv[2]
    elif len(sys.argv) == 5:
        config_path_train, config_path_val, config_path_test = sys.argv[2], sys.argv[3], sys.argv[4]
    else:
        raise ValueError(f"Invalid config paths. Must specify one or three config paths and training config. ({len(sys.argv)}: {sys.argv})")
    
    config_model = load_config_file(config_path_model, 'model')
    config_train, config_val, config_test = load_config_file(config_path_train, 'train', config_model.CONFIGS_USED_FOLDER),\
                                            load_config_file(config_path_val, 'val', config_model.CONFIGS_USED_FOLDER),\
                                            load_config_file(config_path_test, 'test', config_model.CONFIGS_USED_FOLDER)
    logging.info("init")    
    
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
    
    logging.info("Creating model")
    
    model = Model(config_model)
    
    logging.info("Init datasets")
    dataset_train = DatasetSPD(config_train)
    train_indexes, val_indexes, test_indexes = None, None, None
    dataset_val, dataset_test = None, None
    
    logging.info(f"Data shape: {dataset_train.X_paths.shape}, {dataset_train.y.shape}")
    
    if is_one_config_supplied:
        dataset_val, dataset_test = deepcopy(dataset_train), deepcopy(dataset_train) # the deepcopy is important. do not change. 
        dataset_test.flip, dataset_test.rot = False, False
        if config_train.SPLIT_DATA:
            logging.info("Split data...")
            train_indexes, val_indexes, test_indexes = dataset_train.split()
    else:
        dataset_val, dataset_test = DatasetSPD(config_val), DatasetSPD(config_test)
    
    batch_size = config_model.BATCH_SIZE
    num_workers = 2 #2 is advised by the warning log
    logging.info(f"Init dataloaders (batch_size: {batch_size}, num_workers: {num_workers})")
    dataloader_train, dataloader_val, dataloader_test = get_dataloader(dataset_train, batch_size, indexes=train_indexes, num_workers=num_workers),\
                                                        get_dataloader(dataset_val, batch_size, indexes=val_indexes, num_workers=num_workers),\
                                                        get_dataloader(dataset_test, batch_size, indexes=test_indexes, num_workers=num_workers)
    
    logging.info(f"\n\n\n\n\nBefore model.load_with_dataloader.. {len(train_indexes)}, {len(val_indexes)}, {len(test_indexes)}")
    
    model.load_with_dataloader(dataloader_train, dataloader_val, dataloader_test)

    logging.info("[Start] Training..")

    model.train_with_dataloader()
                
    logging.info("[End] Training..")


if __name__ == "__main__":    
    print("Calling the training func...")
    try:
        train_with_dataloader()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
    
    