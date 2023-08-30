import datetime
import os
import sys




sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import pandas as pd
import logging
import  torch

from src.common.lib.utils import get_if_exists, load_config_file
from src.common.lib.model import Model
from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_spd import DatasetSPD


def eval_model():
    
    if len(sys.argv) != 3:
        raise ValueError("Invalid config path. Must supply model config and data config.")
    
    config_path_model = sys.argv[1]
    config_path_data = sys.argv[2]

    config_model = load_config_file(config_path_model, 'model')
    config_data = load_config_file(config_path_data, 'data', config_model.CONFIGS_USED_FOLDER)

    logging.info("init")
    logging.info("[Eval Model]")
    
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
    
    logging.info("Init datasets")
    dataset = DatasetSPD(config_data)
    
    logging.info(f"Data shape: {dataset.X_paths.shape}, {dataset.y.shape}")
    
    __unique_labels_path = os.path.join(config_model.MODEL_OUTPUT_FOLDER, "unique_labels.npy")
    if os.path.exists(__unique_labels_path):
        logging.info(f"unique_labels.npy files has been detected - using it. ({__unique_labels_path})")
        dataset.unique_markers = np.load(__unique_labels_path)
    else:
        logging.warn(f"Couldn't find unique_labels file: {__unique_labels_path}")
    
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
    
    logging.info("Generate reconsturcted images")
    reconstructed_image_path = model.generate_reconstructed_image()
    logging.info(f"Image was saved to {reconstructed_image_path}")
        
    logging.info("Loading analytics..")
    model.load_analytics()
    logging.info("Plot umap..")
    model.plot_umap(colormap='Set1',
                    alpha=0.8,
                    s=0.8,
                    is_3d=False,
                    title=f"{'_'.join([os.path.basename(f) for f in config_data.INPUT_FOLDERS])}_{datetime.datetime.now().strftime('%d%m%y_%H%M%S_%f')}_{os.path.splitext(os.path.basename(config_model.MODEL_PATH))[0]}")
    

if __name__ == "__main__":
    print("Starting eval...")
    try:
        eval_model()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")