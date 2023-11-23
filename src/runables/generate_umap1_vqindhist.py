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
    
    if len(sys.argv) < 3:
        raise ValueError("Invalid config path. Must supply model config and data config.")
    
    config_path_model = sys.argv[1]
    config_path_data = sys.argv[2]

    config_model = load_config_file(config_path_model, 'model')
    config_data = load_config_file(config_path_data, 'data', config_model.CONFIGS_USED_FOLDER)
    output_folder_path = sys.argv[3] if len(sys.argv) > 3 else config_model.MODEL_OUTPUT_FOLDER

    assert os.path.isdir(output_folder_path) and os.path.exists(output_folder_path), f"{output_folder_path} is an invalid output folder path or doesn't exists"

    logging.info("init")
    logging.info("[Gnerate UMAP1]")
    
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
        dataset = Dataset.get_subset(dataset, indexes)
    else:
        indexes = None
    
    logging.info("Init model")
    model = Model(config_model)
    
    logging.info(f"Loading model (Path: {config_model.MODEL_PATH})")
    model.load_model(num_fc_output_classes=len(dataset.unique_markers))
    
    __generate_with_load(config_model, config_data, dataset, model, output_folder_path)
    return None

def __generate_with_load(config_model, config_data, dataset, model, output_folder_path):
    logging.info("Clearing cache")
    torch.cuda.empty_cache()
    
    __now = datetime.datetime.now()
    
    model.generate_dummy_analytics()
    embeddings, labels = model.load_indhists(embeddings_type='testset' if config_data.SPLIT_DATA else 'all',
                                               config_data=config_data)
    logging.info(f'[__generate_with_load]: embeddings shape: {embeddings.shape}, labels shape: {labels.shape}')

        
    logging.info(f"Plot umap...")
    title = f"{'_'.join([os.path.basename(f) for f in dataset.input_folders])}"
    savepath = os.path.join(output_folder_path,\
                            'UMAPs',\
                            'UMAP1'
                                f'{__now.strftime("%d%m%y_%H%M%S_%f")}_{os.path.splitext(os.path.basename(config_model.MODEL_PATH))[0]}',\
                                    f'{title}.png')
        
    __savepath_parent = os.path.dirname(savepath)
    if not os.path.exists(__savepath_parent):
        os.makedirs(__savepath_parent)
        
    colormap = get_if_exists(config_data, 'COLORMAP', 'Set1')
    size = get_if_exists(config_data, 'SIZE', 0.8)
    alpha = get_if_exists(config_data, 'ALPHA', 0.7)
    map_labels_function = get_if_exists(config_data, 'MAP_LABELS_FUNCTION', None)
    if map_labels_function is not None:
        map_labels_function = eval(map_labels_function)(config_data)
    
    model.plot_umap(embedding_data=embeddings,
                    label_data=labels,
                    title=title,
                    savepath=savepath,
                    colormap=colormap,
                    alpha=alpha,
                    s=size,
                    reset_umap=True,
                    map_labels_function=map_labels_function)
    
    logging.info(f"UMAP saved successfully to {savepath}")
    return None

if __name__ == "__main__":
    print("Starting generating umaps...")
    try:
        generate_umaps()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")