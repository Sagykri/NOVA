import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")
import logging
from src.models.architectures.NOVA_model import NOVAModel
from src.embeddings.embeddings_utils import load_embeddings
from src.datasets.dataset_NOVA import DatasetNOVA
from tools.attn_maps_plotting.dataset_PATHS import DatasetFromPaths
from src.datasets.data_loader import get_dataloader
from src.models.utils.consts import CHECKPOINT_BEST_FILENAME, CHECKPOINTS_FOLDERNAME
from typing import Dict, List, Optional, Tuple, Callable
from copy import deepcopy
import numpy as np
import torch
from src.attention_maps.attention_maps_utils import generate_attn_maps, process_attn_maps, save_attn_maps
from tools.attn_maps_plotting.attention_maps_plotting import plot_attn_maps
from src.analysis.analyzer_attention_scores import AnalyzerAttnScore
import re


# arguments: model, dataset config, attn_config, plot_attn_config 

# load configs
# load model 
# get attn matrix from model (generate using infernce)
# process attn matrix (process attn maps)
# plot - create fig 

def generate_attn_maps_with_model(paths_by_type, outputs_folder_path:str, config_data, 
                                config_attn, config_plot, 
                                 config_corr = None ,batch_size:int=50)->None:
        """
            For each sample in the data config - 
                - extracts the attention maps from the model 
                - saves the raw attention maps
                - process the attention maps according to the parameters in the attn config
                - saves the processed attn maps
        """


        num_workers = min(config_plot.PLOT_ATTN_NUM_WORKERS, os.cpu_count())

        # load model
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder.")
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder, and inside a {CHECKPOINT_BEST_FILENAME} file.")
        chkp_path = os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)
        model = NOVAModel.load_from_checkpoint(chkp_path)

        # load attn score analyzer if specified
        if config_corr is not None:
            d = AnalyzerAttnScore(config_data, outputs_folder_path, config_corr)
            corr_method = config_corr.CORR_METHOD
        else:
            d = None
            corr_method = None
            corr_data = None

        attn_maps_output_folder = os.path.join(outputs_folder_path, 'figures', config_data.EXPERIMENT_TYPE, "attention_maps")

        # iteratre on paths_by_type dictionary
        for description, original_paths in paths_by_type.items():
            batch_size = min(batch_size, len(original_paths))

            # remove tiles from paths
            no_tiles_paths = [re.sub(r"/\d+$", "", path) for path in original_paths]
            no_tiles_paths = list(set(no_tiles_paths))
            
            # create dataset
            dataset = DatasetFromPaths(config_data, no_tiles_paths)

            
            # generate (extract from model) raw attention maps and save
            attn_maps, labels, paths = __generate_attn_maps_with_paths_dataloader(
                dataset=dataset, model=model, batch_size=batch_size, num_workers=1)
            
            # keep only samples with matching tiles of original paths
            tiles_idx = np.isin(paths, np.array(original_paths))
            paths = paths[tiles_idx]
            attn_maps = attn_maps[tiles_idx]
            labels = labels[tiles_idx]

            
            print("paths", paths.shape)
            print("attn_maps", attn_maps.shape)
            print("labels", labels.shape)

            # process the raw attn_map and save 
            processed_attn_maps = process_attn_maps([attn_maps], [labels], config_data, config_attn)

            if d is not None:
                corr_data = d.calculate(processed_attn_maps, [labels], [paths])

            temp_output_path = os.path.join(attn_maps_output_folder, description)

            plot_attn_maps(processed_attn_maps, [labels], [paths], config_data, config_plot, output_folder_path=temp_output_path, num_workers = num_workers, corr_data =  corr_data,corr_method = corr_method)



def __generate_attn_maps_with_paths_dataloader(dataset:DatasetNOVA, model:NOVAModel, batch_size:int=700, 
                                          num_workers:int=6)->Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]:
    data_loader = get_dataloader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    logging.info(f"[generate_attn_maps_with_dataloader] Data loaded: there are {len(dataset)} images.")
    
    attn_maps, labels, paths = model.gen_attn_maps(data_loader) # (num_samples, num_layers, num_heads, num_patches, num_patches)
    logging.info(f'[generate_attn_maps_with_dataloader] total attn_maps: {attn_maps.shape}')
    
    return attn_maps, labels, paths

