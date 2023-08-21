import os
import sys




sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import pandas as pd
import logging
import  torch
import datetime

from src.common.lib.dataset import Dataset
from src.common.lib.utils import get_if_exists, load_config_file
from src.common.lib.model import Model
from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_spd import DatasetSPD


def generate_umaps():
    
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
        dataset = Dataset.get_subset(dataset, indexes)
    else:
        indexes = None

    batch_size = config_model.BATCH_SIZE
    num_workers = 1 # No need in more since eval takes only the first 10 tiles
    
    logging.info("Init model")
    model = Model(config_model)
    
    n_class = 225
    logging.warning(f"NOTE! Setting len(unique_markers) to {n_class} !!!!")
    model.unique_markers = np.arange(n_class)
    
    logging.info(f"Loading model (Path: {config_model.MODEL_PATH})")
    model.load_model()
    
    __now = datetime.datetime.now()
    markers = np.unique([m.split('_')[0] if '_' in m else m for m in dataset.unique_markers]) 
    logging.info(f"Markers detected: {markers}")
    
    calc_embeddings = get_if_exists(config_data, 'CALCULATE_EMBEDDINGS', False)
    
    for c in markers:
        logging.info("Clearing cache")
        torch.cuda.empty_cache()
        
        logging.info(f"[{c}]")
        
        logging.info(f"[{c}] Selecting indexes of marker")
        c_indexes = np.where(np.char.startswith(dataset.y.astype(str), f"{c}_"))[0]
        logging.info(f"[{c}] {len(c_indexes)} indexes have been selected")
                        
        logging.info(f"[{c}] Init dataloaders (batch_size: {batch_size}, num_workers: {num_workers})")
        dataloader = get_dataloader(dataset, batch_size, indexes=c_indexes, num_workers=num_workers)    
        
        logging.info(f"[{c}] Loading model with dataloader")
        model.load_with_dataloader(test_loader=dataloader)
        
        logging.info(f"[{c}] Loading analytics..")
        model.load_analytics()
        
        logging.info(f"[{c}] Plot umap...")
        title = f"{'_'.join([os.path.basename(f) for f in dataset.input_folders])}_{c}"
        savepath = os.path.join(model.conf.MODEL_OUTPUT_FOLDER,\
                                'UMAPs',\
                                    f'{__now.strftime("%d%m%y_%H%M%S_%f")}_{os.path.splitext(os.path.basename(config_model.MODEL_PATH))[0]}',\
                                        f'{title}.png')
        
        __savepath_parent = os.path.dirname(savepath)
        if not os.path.exists(__savepath_parent):
            os.makedirs(__savepath_parent)
        
        model.plot_umap(title=title, savepath=savepath,
                        colormap='Set1',
                        alpha=0.7,
                        s=0.8,
                        calc_embeddings=calc_embeddings,
                        id2label=dataloader.dataset.id2label)
        
        logging.info(f"[{c}] UMAP saved successfully to {savepath}")

if __name__ == "__main__":
    print("Starting generating umaps...")
    try:
        generate_umaps()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")