import datetime
import logging
import sys
import os

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import pandas as pd
import  torch

from src.common.lib.utils import get_if_exists, load_config_file
from src.common.lib.model import Model
from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_spd import DatasetSPD
from src.common.lib.synthetic_multiplexing import multiplex

def run_synthetic_multiplexing():
    
    if len(sys.argv) < 3:
        raise ValueError("Invalid config path. Must supply model config and data config.")
    
    config_path_model = sys.argv[1]
    config_path_data = sys.argv[2]
    output_folder_path = sys.argv[3] if len(sys.argv) > 3 else config_model.MODEL_OUTPUT_FOLDER

    assert os.path.isdir(output_folder_path) and os.path.exists(output_folder_path), f"{output_folder_path} is an invalid output folder path or doesn't exists"


    config_model = load_config_file(config_path_model, 'model')
    config_data = load_config_file(config_path_data, 'data', config_model.CONFIGS_USED_FOLDER)

    logging.info("init")
    logging.info("[Synthetic Multiplexing]")
    
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
    
    logging.info("Init datasets")
    dataset = DatasetSPD(config_data)
    
    __unique_labels_path = os.path.join(config_model.MODEL_OUTPUT_FOLDER, "unique_labels.npy")
    if os.path.exists(__unique_labels_path):
        logging.info(f"unique_labels.npy files has been detected - using it. ({__unique_labels_path})")
        dataset.unique_markers = np.load(__unique_labels_path)
    else:
        logging.warn(f"Couldn't find unique_labels file: {__unique_labels_path}")
    
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
    
    logging.info("Loading model with dataloader")
    model.load_with_dataloader(test_loader=dataloader)
    
    logging.info(f"Loading model (Path: {config_model.MODEL_PATH})")
    model.load_model()
    
    logging.info("Multiplex!")
    embeddings_type = get_if_exists(model.test_loader.dataset.conf,
                                    'EMBEDDINGS_TYPE_TO_LOAD',
                                    'testset' if config_data.SPLIT_DATA else 'all')


    title = f"{'_'.join([os.path.basename(f) for f in dataset.input_folders])}"
    savepath = os.path.join(output_folder_path,\
                            'SM_UMAPs',\
                                f'{datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")}_{os.path.splitext(os.path.basename(config_model.MODEL_PATH))[0]}',\
                                    f'{title}.png')
    
    __savepath_parent = os.path.dirname(savepath)
    if not os.path.exists(__savepath_parent):
        os.makedirs(__savepath_parent)

    multiplex(model,
              embeddings_type=embeddings_type,
              savepath=savepath)
    

if __name__ == "__main__":
    print("Running synthetic multiplexing...")
    try:
        run_synthetic_multiplexing()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")