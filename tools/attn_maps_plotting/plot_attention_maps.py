import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))


import logging
from src.embeddings.embeddings_utils import load_embeddings
from src.common.utils import load_config_file
from src.datasets.dataset_config import DatasetConfig
from tools.attn_maps_plotting.plot_attention_config import PlotAttnMapConfig
from src.models.utils.consts import CHECKPOINT_BEST_FILENAME, CHECKPOINTS_FOLDERNAME
from typing import Dict, List, Optional, Tuple, Callable
from copy import deepcopy
import numpy as np
import torch
from tools.attn_maps_plotting.attention_maps_plotting import plot_attn_maps
from src.datasets.label_utils import get_batches_from_input_folders
from tools.load_data_from_npy import __extract_indices_to_plot, __extract_samples_to_plot
from src.analysis.analyzer_attention_scores import AnalyzerAttnScore
from tools.attn_maps_plotting.analyzer_pair_wise_distances import AnalyzerPairwiseDistances
from src.datasets.label_utils import get_unique_parts_from_labels, get_markers_from_labels


def load_and_plot_attn_maps(outputs_folder_path:str, config_path_data:str, config_path_plot:str, config_path_corr:str = None):
    config_data:DatasetConfig = load_config_file(config_path_data, "data")
    config_data.OUTPUTS_FOLDER = outputs_folder_path
    config_plot:PlotAttnMapConfig = load_config_file(config_path_plot, "plot")

    if config_path_corr is not None:
        config_corr = load_config_file(config_path_corr, "data")
        corr_method = config_corr.CORR_METHOD
        features_names = config_corr.FEATURES_NAMES

    # load processed attn maps
    processed_attn_maps, labels, paths = load_embeddings(os.path.join(outputs_folder_path, "attention_maps"), config_data, emb_folder_name = "processed")
    processed_attn_maps, labels, paths = [processed_attn_maps], [labels], [paths] #TODO: fix, needed for settypes
    
    
    d = AnalyzerAttnScore(config_data, output_folder_path, config_corr) #TODO: decied if to instancize it - here only for the output dir

    # load correlation data if needed
    if config_plot.SHOW_CORR_SCORES:
        d.load()
        corr_data = d.features
    else:
        corr_method = ""
        corr_data = None
    
    num_workers = min(config_plot.PLOT_ATTN_NUM_WORKERS, os.cpu_count())
    # filter for subsets if needed
    if config_plot.FILTER_SAMPLES_BY_FOLDER_PATHS:
        marker_names = get_unique_parts_from_labels(labels, get_markers_from_labels)
        pair_wise_output_folder = d.get_saving_folder(feature_type='pairwise_distances', main_folder = 'figures')
        attn_maps_output_folder = d.get_saving_folder(feature_type="attention_maps", main_folder = 'figures')
        for marker in marker_names:
            keep_samples_dirs = [os.path.join(pair_wise_output_folder, f"{marker}_paths.npy")]
            samples_indices = extract_indices(keep_samples_dirs=keep_samples_dirs, paths = paths, data_config = config_data)
            marker_processed_attn_maps = __extract_samples_to_plot(processed_attn_maps, samples_indices, data_config = config_data)
            marker_labels = __extract_samples_to_plot(labels, samples_indices, data_config = config_data)
            marker_paths = __extract_samples_to_plot(paths, samples_indices, data_config = config_data)
            if corr_data is not None:
                 marker_corr_data = __extract_samples_to_plot(corr_data, samples_indices, data_config = config_data)
            else:
                marker_corr_data = None
            save_path = os.path.join(attn_maps_output_folder,marker)
            plot_attn_maps(marker_processed_attn_maps, marker_labels, marker_paths, 
                            config_data, config_plot, 
                            output_folder_path=save_path, num_workers = num_workers, 
                            corr_data =  marker_corr_data,corr_method = corr_method)
    
    # TODO: keep seperation by settype?
    else:
        plot_attn_maps(processed_attn_maps, labels, paths, 
                        config_data, config_plot, 
                        output_folder_path=d.get_saving_folder(feature_type="attention_maps"),
                        num_workers = num_workers, 
                        corr_data =  corr_data,corr_method = corr_method)

def extract_indices(keep_samples_dirs: list[str], paths: np.ndarray, data_config: DatasetConfig):
    if data_config.SPLIT_DATA:
        data_set_types = ['trainset', 'valset', 'testset']
    else:
        data_set_types = ['testset']

    all_samples_indices = []

    for i, set_type in enumerate(data_set_types):
        cur_paths = paths[i]  # NumPy array of strings

        # Accumulate all keep_paths from all dirs
        combined_keep_paths = set()
        for dir_path in keep_samples_dirs:
            keep_paths_array = np.load(dir_path, allow_pickle=True)
            combined_keep_paths.update(keep_paths_array.tolist())

        # Find indices in cur_paths that match any in combined_keep_paths
        samples_indices = [j for j, p in enumerate(cur_paths) if p in combined_keep_paths]
        all_samples_indices.append(samples_indices)

    return all_samples_indices
    
        

if __name__ == "__main__":
    print("Starting plotting attention maps...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply output folder path, data config and path config!")
        output_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]
        config_path_plot = sys.argv[3]

        if len(sys.argv) == 5:
            config_path_corr = sys.argv[4]
        else:
            config_path_corr = None
        

        load_and_plot_attn_maps(output_folder_path, config_path_data, config_path_plot, config_path_corr)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
