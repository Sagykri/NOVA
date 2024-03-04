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
    
    jobid = os.getenv('LSB_JOBID')
    
    config_model = load_config_file(config_path_model, 'model')
    config_data = load_config_file(config_path_data, 'data', config_model.CONFIGS_USED_FOLDER)
    
    figure_output_folder = get_if_exists(config_data, 'FIGURE_OUTPUT_FOLDER', None)
    if len(sys.argv) > 3:
        output_folder_path = sys.argv[3]
    elif figure_output_folder is not None:
        output_folder_path = figure_output_folder
    else:
        output_folder_path = config_model.MODEL_OUTPUT_FOLDER
    
    if not os.path.exists(output_folder_path):
        logging.info(f"{output_folder_path} doesn't exists. Creating it")
        os.makedirs(output_folder_path)
    
    logging.info(f"init - jobid: {jobid}")
    logging.info("[Synthetic Multiplexing]")
    
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
    
    __unique_labels_path = os.path.join(config_model.MODEL_OUTPUT_FOLDER, "unique_labels.npy")
    if os.path.exists(__unique_labels_path):
        logging.info(f"unique_labels.npy files has been detected - using it. ({__unique_labels_path})")
        unique_markers = np.load(__unique_labels_path)
    else:
        logging.warn(f"Couldn't find unique_labels file: {__unique_labels_path}")
    
    logging.info("Init model")
    model = Model(config_model)
    
    logging.info("Multiplex!")
    embeddings_type = get_if_exists(config_data,
                                    'EMBEDDINGS_TYPE_TO_LOAD',
                                    'testset' if config_data.SPLIT_DATA else 'all')
    
    embeddings_layer = get_if_exists(config_data, 'EMBEDDINGS_LAYER', None)

    logging.info(f"Embeddings layer: {embeddings_layer}")
    
    colormap = get_if_exists(config_data, 'COLORMAP', 'Set1')
    size = get_if_exists(config_data, 'SIZE', 0.8)
    alpha = get_if_exists(config_data, 'ALPHA', 0.7)
    map_labels_function = get_if_exists(config_data, 'MAP_LABELS_FUNCTION', None)
    if map_labels_function is not None:
        map_labels_function = eval(map_labels_function)(config_data)


    title = f"{'_'.join([os.path.basename(f) for f in config_data.INPUT_FOLDERS])}"
    savepath = os.path.join(output_folder_path,\
                            'SM_UMAPs',\
                                f'{datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")}_{jobid}_{os.path.splitext(os.path.basename(config_model.MODEL_PATH))[0]}',\
                                    f'{title}')
    
    __savepath_parent = os.path.dirname(savepath)
    if not os.path.exists(__savepath_parent):
        os.makedirs(__savepath_parent)

    multiplex(model,
              embeddings_type=embeddings_type,
              savepath=savepath,
              colormap=colormap,
              output_layer=embeddings_layer,
              s=size,
              alpha=alpha,
              map_labels_function=map_labels_function,
              config_data=config_data)
    

if __name__ == "__main__":
    print("Running synthetic multiplexing...")
    try:
        run_synthetic_multiplexing()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
