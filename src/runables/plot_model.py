import os
import sys




sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import pandas as pd
import logging
import  torch

from src.common.lib.utils import load_config_file
from src.common.lib.model import Model
from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_spd import DatasetSPD


def plot_model():
    """
    This function only creates the files to be open via tensorboard.
    You should run:
    ```
    tensorboard --logdir=PATH_TO_FOLDER --port=6008
    ```
    (The port is optional and you should set it to an open port number)
    """
    
    if len(sys.argv) != 2:
        raise ValueError("Invalid model config path.")
    
    config_path_model = sys.argv[1]
    
    config_model = load_config_file(config_path_model, 'model')
    
    logging.info("init")
    
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
    
    logging.info("Init model")
    model = Model(config_model)
    
    logging.info("Generate Viz")
    model.generate_model_visualization(26)
    
    

if __name__ == "__main__":
    print("Starting...")
    try:
        plot_model()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")