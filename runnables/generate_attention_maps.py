import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")
import logging
from src.models.architectures.NOVA_model import NOVAModel
from src.embeddings.embeddings_utils import load_embeddings
from src.common.utils import load_config_file
from src.datasets.dataset_config import DatasetConfig
from src.figures.plot_config import PlotConfig
from src.models.utils.consts import CHECKPOINT_BEST_FILENAME, CHECKPOINTS_FOLDERNAME
from typing import Dict, List, Optional, Tuple, Callable
from copy import deepcopy
import numpy as np
import torch
from src.attention_maps.attention_config import AttnConfig
from src.attention_maps.attention_maps_utils import generate_attn_maps, process_attn_maps, save_attn_maps


def generate_attn_maps_with_model(outputs_folder_path:str, config_path_data:str, config_path_attn:str, batch_size:int=700)->None:
    """
        For each sample in the data config - 
            - extracts the attention maps from the model 
            - saves the raw attention maps
            - process the attention maps according to the parameters in the attn config
            - saves the processed attn maps
    """
    # load configs
    config_data:DatasetConfig = load_config_file(config_path_data, "data")
    config_attn:AttnConfig = load_config_file(config_path_attn, "data")
    config_data.OUTPUTS_FOLDER = outputs_folder_path
    
    # load model
    chkp_path = os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)
    model = NOVAModel.load_from_checkpoint(chkp_path)

    # generate (extract from model) raw attention maps and save (if specified)
    attn_maps, labels, paths = generate_attn_maps(model, config_data, batch_size=batch_size)
    if config_attn.SAVE_RAW_ATTN:
        save_attn_maps(attn_maps, labels, paths, config_data, output_folder_path=os.path.join(outputs_folder_path, "attention_maps", "raw"))

    # process the raw attn_map and save 
    num_workers = min(config_attn.ATTN_NUM_WORKERS, os.cpu_count())
    processed_attn_maps = process_attn_maps(attn_maps, labels, config_data, config_attn, num_workers=num_workers)
    del attn_maps
    save_attn_maps(processed_attn_maps, labels, paths, config_data, output_folder_path=os.path.join(outputs_folder_path, "attention_maps", "processed"))


if __name__ == "__main__":
    print("Starting generate attention maps...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply model path, data config and attn config.")
        outputs_folder_path = sys.argv[1]
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder.")
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder, and inside a {CHECKPOINT_BEST_FILENAME} file.")
        
        config_path_data = sys.argv[2]
        config_path_attn = sys.argv[3]

        if len(sys.argv)==5:
            try:
                batch_size = int(sys.argv[4])
            except ValueError:
                raise ValueError("Invalid batch size, must be integer")
        else:
            batch_size = 200
        generate_attn_maps_with_model(outputs_folder_path, config_path_data, config_path_attn, batch_size)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
