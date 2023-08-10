import logging
import sys
import os

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import pandas as pd
import  torch

from src.common.lib.utils import load_config_file
from src.common.lib.model import Model
from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_spd import DatasetSPD
from src.common.lib.synthetic_multiplexing import multiplex

def eval_model():
    
    if len(sys.argv) != 3:
        raise ValueError("Invalid config path. Must supply model config and data config.")
    
    config_path_model = sys.argv[1]
    config_path_data = sys.argv[2]

    config_model = load_config_file(config_path_model, 'model')
    config_data = load_config_file(config_path_data, 'data', config_model.CONFIGS_USED_FOLDER)

    logging.info("init")
    
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
    
    logging.info("Init datasets")
    dataset = DatasetSPD(config_data)
    
    logging.info(f"Data shape: {dataset.X_paths.shape}, {dataset.y.shape}")
    
    dataset.flip, dataset.rot = False, False
    if config_data.SPLIT_DATA:
        logging.info("Split data...")
        _, _, indexes = dataset.split()
    else:
        indexes = None

    batch_size = config_model.BATCH_SIZE
    num_workers = 1 # No need in more since eval takes only the first 10 tiles
    logging.info(f"Init dataloaders (batch_size: {batch_size}, num_workers: {num_workers})")
    dataloader = get_dataloader(dataset, batch_size, indexes=indexes, num_workers=num_workers)    
    
    logging.info("Init model")
    model = Model(config_model)
    
    n_class = 225#1311#219#225
    logging.warning(f"NOTE! Setting len(unique_markers) to {n_class} !!!!")
    model.unique_markers = np.arange(n_class)
    
    logging.info("Loading model with dataloader")
    model.load_with_dataloader(test_loader=dataloader)
    
    logging.info(f"Loading model (Path: {config_model.MODEL_PATH})")
    model.load_model()
    
    logging.info("Multiplex!")
    multiplex(model)
    

if __name__ == "__main__":
    print("Testing synthetic multiplexing...")
    try:
        eval_model()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")