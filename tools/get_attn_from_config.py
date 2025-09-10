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
from src.datasets.dataset_NOVA import DatasetNOVA
from src.datasets.data_loader import get_dataloader
from src.figures.plot_config import PlotConfig
from src.models.utils.consts import CHECKPOINT_BEST_FILENAME, CHECKPOINTS_FOLDERNAME
from typing import Dict, List, Optional, Tuple, Callable
from copy import deepcopy
import numpy as np
import torch
from src.attention_maps.attention_config import AttnConfig
from src.attention_maps.attention_maps_utils import generate_attn_maps, process_attn_maps, save_attn_maps
from src.figures.attention_maps_plotting import plot_attn_maps
from src.analysis.analyzer_attention_correlation import AnalyzerAttnCorr


# arguments: model, dataset config, attn_config, plot_attn_config 

# load configs
# load model 
# get attn matrix from model (generate using infernce)
# process attn matrix (process attn maps)
# plot - create fig 

MAX_SAMPLES = 100
def generate_attn_maps_with_model(outputs_folder_path:str, config_path_data:str, config_path_attn:str,config_path_plot:str, batch_size:int=10)->None:


    # load configs
    config_data:DatasetConfig = load_config_file(config_path_data, "data")
    config_attn:AttnConfig = load_config_file(config_path_attn, "data")
    config_data.OUTPUTS_FOLDER = outputs_folder_path
    config_plot:PlotAttnMapConfig = load_config_file(config_path_plot, "plot")
    
    # load model
    chkp_path = os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)
    model = NOVAModel.load_from_checkpoint(chkp_path)

    temp_output_path = os.path.join("attn_by_paths", "FUS_for_sagy")
    # create dataset
    dataset = DatasetNOVA(config_data)
    batch_size = len(dataset) // 3
        
    # generate (extract from model) raw attention maps and save
    attn_maps, labels, paths = __generate_attn_maps_with_paths_dataloader(
            dataset=dataset, model=model, batch_size=batch_size, num_workers=3)

    num_samples = min(len(labels), MAX_SAMPLES)
    attn_maps, labels, paths = attn_maps[:num_samples], labels[:num_samples], paths[:num_samples]
    # process the raw attn_map and save 
    processed_attn_maps = process_attn_maps([attn_maps], [labels], config_data, config_attn)

    plot_attn_maps(processed_attn_maps, [labels], [paths], config_data, config_plot, output_folder_path=temp_output_path,corr_data =  None,corr_method = "")

def __generate_attn_maps_with_paths_dataloader(dataset:DatasetNOVA, model:NOVAModel, batch_size:int=700, 
                                          num_workers:int=6)->Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]:
    data_loader = get_dataloader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    logging.info(f"[generate_attn_maps_with_dataloader] Data loaded: there are {len(dataset)} images.")
    
    attn_maps, labels, paths = model.gen_attn_maps(data_loader) # (num_samples, num_layers, num_heads, num_patches, num_patches)
    logging.info(f'[generate_attn_maps_with_dataloader] total attn_maps: {attn_maps.shape}')
    
    return attn_maps, labels, paths


if __name__ == "__main__":
    print("Starting generate attention maps...")
    

    try:
        if len(sys.argv) < 5:
            raise ValueError("Invalid arguments. Must supply model path, data config, attn config, plot_attn_config")
        outputs_folder_path = sys.argv[1]
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder.")
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder, and inside a {CHECKPOINT_BEST_FILENAME} file.")
        
        config_path_data = sys.argv[2]
        config_path_attn = sys.argv[3]
        config_path_plot = sys.argv[4]

        if len(sys.argv)==6:
            try:
                batch_size = int(sys.argv[5])
            except ValueError:
                raise ValueError("Invalid batch size, must be integer")
        else:
            batch_size = 10
        generate_attn_maps_with_model(outputs_folder_path, config_path_data, config_path_attn, config_path_plot, batch_size)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
