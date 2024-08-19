import os
import sys




sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import pandas as pd
import logging
import  torch
import datetime

from src.common.lib.utils import get_if_exists, load_config_file
from src.common.lib.model import Model


def generate_umaps():
    
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

    logging.info(f"init (jobid: {jobid})")
    logging.info("[Generate UMAPs]")
    
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
        
    __unique_labels_path = os.path.join(config_model.MODEL_OUTPUT_FOLDER, "unique_labels.npy")
    if os.path.exists(__unique_labels_path):
        logging.info(f"unique_labels.npy files has been detected - using it. ({__unique_labels_path})")
        unique_markers = np.load(__unique_labels_path)
    else:
        logging.warn(f"Couldn't find unique_labels file: {__unique_labels_path}")
        raise Exception(f"Couldn't find unique_labels file: {__unique_labels_path}")
    
    logging.info("Init model")
    model = Model(config_model)
    
    __generate_with_load(config_model, config_data, model, output_folder_path)


def __generate_with_load(config_model, config_data, model, output_folder_path):
    logging.info("Clearing cache")
    torch.cuda.empty_cache()
    
    __now = datetime.datetime.now()
    
    model.generate_dummy_analytics()
    embeddings, labels = model.load_embeddings(embeddings_type='testset' if config_data.SPLIT_DATA else 'all',
                                               config_data=config_data)

    markers = np.unique([m.split('_')[-1] if '_' in m else m for m in np.unique(labels.reshape(-1,))]) 
    logging.info(f"Detected markers: {markers}")
    
    for c in markers:
        logging.info(f"Marker: {c}")
        logging.info(f"[{c}] Selecting indexes of marker")
        c_indexes = np.where(np.char.endswith(labels.astype(str), f"_{c}"))[0]
        logging.info(f"[{c}] {len(c_indexes)} indexes have been selected")

        if len(c_indexes) == 0:
            logging.info(f"[{c}] Not exists in embedding. Skipping to the next one")
            continue

        embeddings_c, labels_c = np.copy(embeddings[c_indexes]), np.copy(labels[c_indexes].reshape(-1,))
        
        logging.info(f"[{c}] Plot umap...")
        title = f"{'_'.join([os.path.basename(f) for f in config_data.INPUT_FOLDERS])}_{'_'.join(config_data.REPS)}_{config_data.EMBEDDINGS_LAYER}_{c}"
        savepath = os.path.join(output_folder_path,\
                                'UMAPs',\
                                    f'{__now.strftime("%d%m%y_%H%M%S_%f")}_{os.path.splitext(os.path.basename(config_model.MODEL_PATH))[0]}',\
                                        f'{title}') # NANCY
        
        __savepath_parent = os.path.dirname(savepath)
        if not os.path.exists(__savepath_parent):
            os.makedirs(__savepath_parent)
        
        colormap = get_if_exists(config_data, 'COLORMAP', 'tab20')
        size = get_if_exists(config_data, 'SIZE', 0.8)
        alpha = get_if_exists(config_data, 'ALPHA', 0.7)
        map_labels_function = get_if_exists(config_data, 'MAP_LABELS_FUNCTION', None)
        if map_labels_function is not None:
            map_labels_function = eval(map_labels_function)(config_data)
        
        model.plot_umap(embedding_data=embeddings_c,
                        label_data=labels_c,
                        title=title, #NANCY - comment this for fig 2
                        savepath=savepath,
                        colormap=colormap,
                        alpha=alpha,
                        s=size,
                        reset_umap=True,
                        map_labels_function=map_labels_function,
                        config_data=config_data)
        
        logging.info(f"[{c}] UMAP saved successfully to {savepath}")
        

if __name__ == "__main__":
    print("Starting generating umaps...")
    try:
        generate_umaps()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
